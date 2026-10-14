"""Depth-preserving slice LFQ tokenizer: paper equations (1)-(7)."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def transformer(dim, heads, layers):
    block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)


class LFQ(nn.Module):
    def __init__(self, bits, temperature=1.):
        super().__init__()
        self.bits, self.temperature = bits, temperature
        self.register_buffer('powers', 2 ** torch.arange(bits, dtype=torch.long))

    def to_bits(self, ids):
        return ((ids[..., None].long() & self.powers) != 0).float() * 2 - 1

    def forward(self, u):
        hard = torch.where(u >= 0, 1., -1.)
        binary = u + (hard - u).detach()
        ids = ((hard > 0).long() * self.powers).sum(-1)
        p = (u.float() / self.temperature).sigmoid().clamp(1e-6, 1-1e-6)
        entropy = lambda v: -(v * v.log() + (1-v) * (1-v).log())
        loss = entropy(p).mean() - entropy(p.mean(tuple(range(p.ndim-1)))).mean()
        return binary, ids, loss

    def straight_through_ids(self, logits, ids):
        """Hard tokens forward; expected binary codes backward (documented estimator)."""
        vocabulary = self.to_bits(torch.arange(2 ** self.bits, device=logits.device))
        soft = logits.softmax(-1) @ vocabulary.to(logits.dtype)
        hard = self.to_bits(ids).to(logits.dtype)
        return soft + (hard - soft).detach()


class CardiacTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.dim, cfg.patch
        self.grid = cfg.image_size // p
        self.patch_embed = nn.Conv2d(1, d, p, stride=p)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.grid**2, d) * .02)
        self.spatial = transformer(d, cfg.heads, cfg.spatial_layers)
        self.time_pos = nn.Parameter(torch.randn(1, cfg.phases, d) * .02)
        self.temporal = transformer(d, cfg.heads, cfg.temporal_layers)
        self.to_binary = nn.Linear(d, cfg.bits)
        self.lfq = LFQ(cfg.bits, cfg.token_temperature)
        self.from_binary = nn.Linear(cfg.bits, d)
        self.decode_temporal = transformer(d, cfg.heads, cfg.temporal_layers)
        self.decode_spatial = transformer(d, cfg.heads, cfg.spatial_layers)
        self.to_pixels = nn.Linear(d, p*p)

    def _run(self, module, x, **kwargs):
        if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(lambda v: module(v, **kwargs), x, use_reentrant=False)
        return module(x, **kwargs)

    def encode(self, x):
        # x is B,T,Z,H,W; depth NEVER shares the temporal axis.
        b, t, z, h, w = x.shape
        if (h, w) != (self.cfg.image_size,)*2 or t > self.cfg.phases or z != self.cfg.slices:
            raise ValueError(f'Unexpected cine shape {tuple(x.shape)}')
        slices = x.reshape(b*t*z, 1, h, w)
        pooled = []
        for chunk in slices.split(self.cfg.slice_chunk):
            tokens = self.patch_embed(chunk).flatten(2).transpose(1, 2)
            pooled.append(self._run(self.spatial, tokens + self.spatial_pos).mean(1))
        s = torch.cat(pooled).reshape(b, t, z, -1).permute(0, 2, 1, 3).reshape(b*z, t, -1)
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        latent = self._run(self.temporal, s + self.time_pos[:, :t], mask=causal)
        u = self.to_binary(latent).reshape(b, z, t, -1).transpose(1, 2)
        return self.lfq(u)

    def decode_bits(self, bits):
        b, t, z, _ = bits.shape
        h = self.from_binary(bits).transpose(1, 2).reshape(b*z, t, -1)
        causal = torch.ones(t, t, device=bits.device, dtype=torch.bool).triu(1)
        h = self._run(self.decode_temporal, h + self.time_pos[:, :t], mask=causal)
        h = h.reshape(b, z, t, -1).transpose(1, 2).reshape(b*t*z, -1)
        pixels = []
        for chunk in h.split(self.cfg.slice_chunk):
            patches = chunk[:, None] + self.spatial_pos
            pixels.append(self.to_pixels(self._run(self.decode_spatial, patches)))
        p, g = self.cfg.patch, self.grid
        out = torch.cat(pixels).reshape(b*t*z, g, g, p, p)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, t, z, g*p, g*p)
        return out.sigmoid()

    def decode(self, ids):
        return self.decode_bits(self.lfq.to_bits(ids))

    def forward(self, x):
        bits, ids, regularizer = self.encode(x)
        return self.decode_bits(bits), ids, regularizer


class FrozenPerceptual(nn.Module):
    """Eq. (7): selected frozen VGG16 layers; encoder choice unspecified by paper.

    Requires an explicit local torchvision VGG16 state dict. No random encoder
    or silent download is substituted for pretrained perceptual features.
    """
    def __init__(self, checkpoint_path, layers=(3, 8, 15), chunk=32):
        super().__init__()
        from torchvision.models import vgg16
        model = vgg16(weights=None)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        self.features = model.features[:max(layers)+1].eval().requires_grad_(False)
        self.layers, self.chunk = layers, chunk
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, pred, target):
        a, b = pred.flatten(0, 2)[:, None], target.flatten(0, 2)[:, None]
        total = pred.new_zeros(())
        for aa, bb in zip(a.split(self.chunk), b.split(self.chunk)):
            aa = (aa.expand(-1, 3, -1, -1) - self.mean) / self.std
            bb = (bb.expand(-1, 3, -1, -1) - self.mean) / self.std
            for i, layer in enumerate(self.features):
                aa, bb = layer(aa), layer(bb)
                if i in self.layers:
                    total = total + F.l1_loss(aa, bb) * len(aa) / len(a) / len(self.layers)
        return total
