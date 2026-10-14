"""Independent aligned cine/deviation streams, equations (19)-(26)."""
import torch
from torch import nn
from torch.nn import functional as F
from .tokenizer import transformer


class CineEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch = cfg.classifier_patch
        pt, pz, ph, pw = self.patch
        n = (cfg.phases//pt)*(cfg.slices//pz)*(cfg.image_size//ph)*(cfg.image_size//pw)
        self.embed = nn.Linear(pt*pz*ph*pw, cfg.dim)
        self.pos = nn.Parameter(torch.randn(1, n, cfg.dim)*.02)
        self.encoder = transformer(cfg.dim, cfg.heads, cfg.classifier_encoder_layers)

    def forward(self, x):
        b, t, z, h, w = x.shape
        pt, pz, ph, pw = self.patch
        x = x.reshape(b, t//pt, pt, z//pz, pz, h//ph, ph, w//pw, pw)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).reshape(b, -1, pt*pz*ph*pw)
        return self.encoder(self.embed(x) + self.pos)


class DiseaseClassifier(nn.Module):
    def __init__(self, cfg, classes=5, variant='full'):
        super().__init__()
        if variant not in ('full', 'pathological_only', 'no_reweighting', 'constant_weighting', 'no_contrastive', 'no_cross_attention'):
            raise ValueError(f'Unknown classifier variant {variant}')
        self.cfg, self.variant = cfg, variant
        self.pathological = CineEncoder(cfg)
        self.difference = CineEncoder(cfg)
        self.weight_p = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        self.weight_d = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        self.proj_p, self.proj_d = nn.Linear(cfg.dim, cfg.dim), nn.Linear(cfg.dim, cfg.dim)
        self.cross = nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.dim)*.02)
        self.encoder = transformer(cfg.dim, cfg.heads, cfg.classifier_layers)
        self.head = nn.Linear(cfg.dim, classes)

    def reweight(self, z, mlp):
        if self.variant == 'no_reweighting':
            return z
        if self.variant == 'constant_weighting':
            return z / z.shape[1]
        return z * mlp(z).softmax(dim=1)

    def forward(self, observed, reference=None, difference=None):
        p = self.reweight(self.pathological(observed), self.weight_p)
        if self.variant == 'pathological_only':
            tokens, ud, up = p, None, None
        else:
            if reference is None or reference.shape != observed.shape:
                raise ValueError('Aligned healthy reference with identical shape required')
            # Signed difference, not absolute error: Delta X = Xp - Xh.
            deviation = observed-reference.detach() if difference is None else difference.detach()
            if deviation.shape != observed.shape:
                raise ValueError('Difference input must match observed cine shape')
            d = self.reweight(self.difference(deviation), self.weight_d)
            ud = F.normalize(self.proj_d(d.mean(1)), dim=-1)
            up = F.normalize(self.proj_p(p.mean(1)), dim=-1)
            fused = d if self.variant == 'no_cross_attention' else d + self.cross(d, p, p, need_weights=False)[0]
            tokens = torch.cat([fused, p], dim=1)
        tokens = torch.cat([self.cls.expand(len(observed), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0]), ud, up

    def objective(self, observed, reference, labels, alpha=None, difference=None):
        logits, ud, up = self(observed, reference, difference)
        logp = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
        weights = torch.ones_like(logp) if alpha is None else alpha[labels]
        # Paper writes sums over subjects for both losses.
        focal = -(weights * (1-logp.exp()).pow(self.cfg.focal_gamma) * logp).sum()
        contrast = logits.sum()*0
        if ud is not None and self.variant != 'no_contrastive':
            log_sim = (ud @ up.T / self.cfg.contrast_temperature).log_softmax(-1)
            positive = labels[:, None] == labels[None, :]  # includes cross-stream same subject
            contrast = -((log_sim*positive).sum(1)/positive.sum(1)).sum()
        return focal+self.cfg.contrast_weight*contrast, {'focal': focal, 'con': contrast}
