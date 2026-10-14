"""Position-aligned AR and selective one-pass correction, equations (8)-(18)."""
import torch
from torch import nn
from torch.nn import functional as F
from .tokenizer import transformer


def context_mask(t, z, corrector=False, device=None):
    """True = disallowed. Queries never see their own token (even through layers).

    Keys/values are RAW token embeddings, never self-attended. This avoids
    indirect target/future leakage through a context token at another depth.
    """
    times = torch.arange(t, device=device).repeat_interleave(z)
    depths = torch.arange(z, device=device).repeat(t)
    qt, kt = times[:, None], times[None, :]
    qz, kz = depths[:, None], depths[None, :]
    allowed = (kt < qt) & (kz == qz)
    if corrector:
        allowed |= ((kt == qt+1) & (kz == qz)) | ((kt == qt) & ((kz-qz).abs() == 1))
    else:
        allowed |= (kt == qt) & (kz < qz)
    # Global anatomy memory is always visible.
    return torch.cat([torch.zeros(t*z, 1, dtype=torch.bool, device=device), ~allowed], dim=1)


class ContextBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, query, memory, mask):
        q = self.norm(query)
        query = query + self.attn(q, memory, memory, attn_mask=mask, need_weights=False)[0]
        return query + self.ff(query)


class PositionAlignedPredictor(nn.Module):
    def __init__(self, cfg, corrector=False):
        super().__init__()
        self.cfg, self.corrector = cfg, corrector
        self.embedding = nn.Embedding(2**cfg.bits, cfg.dim)
        self.time_pos = nn.Embedding(cfg.phases, cfg.dim)
        self.depth_pos = nn.Embedding(cfg.slices, cfg.dim)
        self.anatomy = transformer(cfg.dim, cfg.heads, 1)
        layers = cfg.corrector_layers if corrector else cfg.ar_layers
        self.blocks = nn.ModuleList([ContextBlock(cfg.dim, cfg.heads) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 2**cfg.bits))

    def forward(self, ids, positions=None):
        b, t, z = ids.shape
        times = torch.arange(t, device=ids.device).repeat_interleave(z)
        depths = torch.arange(z, device=ids.device).repeat(t)
        pos = self.time_pos(times) + self.depth_pos(depths)
        # Eq. (8), pooled first-volume tokens -> lightweight Transformer.
        anatomy = self.anatomy(self.embedding(ids[:, 0]).mean(1, keepdim=True))
        memory = torch.cat([anatomy, self.embedding(ids.flatten(1)) + pos], dim=1)
        query = pos[None].expand(b, -1, -1)
        mask = context_mask(t, z, self.corrector, ids.device)
        if positions is not None:
            query, mask = query[:, positions], mask[positions]
        for block in self.blocks:
            query = block(query, memory, mask)
        logits = self.head(query)
        return logits if positions is not None else logits.reshape(b, t, z, -1)


def second_difference(h, axis):
    """Second-order interior; forward/backward second-order boundary stencil."""
    x = h.movedim(axis, 1)
    if x.shape[1] < 3:
        return torch.zeros_like(h[..., 0])
    interior = x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]
    out = torch.cat([interior[:, :1], interior, interior[:, -1:]], dim=1)
    return out.abs().sum(-1).movedim(1, axis)


class HealthyGenerator(nn.Module):
    def __init__(self, cfg, consistency=True, resampling=True):
        super().__init__()
        self.cfg, self.consistency, self.resampling = cfg, consistency, resampling
        self.ar = PositionAlignedPredictor(cfg)
        self.corrector = PositionAlignedPredictor(cfg, corrector=True)
        self.detector = nn.Sequential(nn.Linear(3, cfg.dim//2), nn.GELU(), nn.Linear(cfg.dim//2, 1))

    @torch.no_grad()
    def rollout(self, first_ids, phases=None, greedy=False):
        b, z = first_ids.shape
        t = phases or self.cfg.phases
        ids = first_ids.new_zeros(b, t, z)
        ids[:, 0] = first_ids
        uncertainty = torch.zeros(b, t, z, device=ids.device)
        # Only the current query is evaluated. Unknown zero tokens are masked out.
        for phase in range(1, t):
            for depth in range(z):
                logits = self.ar(ids, [phase*z+depth])[:, 0]
                prob = (logits / self.cfg.sample_temperature).softmax(-1)
                token = prob.argmax(-1) if greedy else torch.multinomial(prob, 1).squeeze(-1)
                ids[:, phase, depth] = token
                uncertainty[:, phase, depth] = 1-logits.softmax(-1).amax(-1)
        return ids, uncertainty

    def risk_logits(self, ids, uncertainty):
        h = self.ar.embedding(ids)
        t, z = ids.shape[1:]
        dt = second_difference(h, 1) / ((t-1)*z)
        dz = second_difference(h, 2) / ((z-1)*t)
        return self.detector(torch.stack([uncertainty, dt, dz], dim=-1)).squeeze(-1)

    def select(self, risk):
        b, t, z = risk.shape
        scores = risk.sigmoid().clone()
        scores[:, 0] = -1  # first-volume condition is immutable
        flat = scores.flatten(1)
        n = min(self.cfg.max_corrections, (t-1)*z)
        selected = torch.zeros_like(flat, dtype=torch.bool)
        if n:
            vals, idx = flat.topk(n, dim=1)
            selected.scatter_(1, idx, vals > self.cfg.risk_threshold)
        return selected.reshape(b, t, z)

    @torch.no_grad()
    def sample(self, tokenizer, first_volume, greedy=False):
        # This API accepts ONLY B,1,Z,H,W; future observed volumes cannot leak.
        if first_volume.shape[1] != 1:
            raise ValueError('Provide only the first 3D volume')
        _, first, _ = tokenizer.encode(first_volume)
        ids, uncertainty = self.rollout(first[:, 0], greedy=greedy)
        selected = torch.zeros_like(ids, dtype=torch.bool)
        if self.resampling:
            selected = self.select(self.risk_logits(ids, uncertainty))
            # All corrections use the same original rollout; each site changed once.
            positions = selected.flatten(1).any(0).nonzero().flatten()
            if len(positions):
                replacements = self.corrector(ids, positions).argmax(-1)
                flat = ids.flatten(1).clone()
                flat[:, positions] = torch.where(selected.flatten(1)[:, positions], replacements, flat[:, positions])
                ids = flat.reshape_as(ids)
        out = tokenizer.decode(ids)
        out[:, :1] = first_volume
        return out, ids, selected

    def objective(self, tokenizer, x):
        """Eq. LGen. Frozen decoder still propagates gradients to token logits.

        Rollout is discrete/no-grad. Recomputed conditional distributions carry a
        straight-through estimator into voxel losses, not gradients through sampling.
        """
        with torch.no_grad():
            _, target, _ = tokenizer.encode(x)
            rollout, uncertainty = self.rollout(target[:, 0])
        teacher = self.ar(target)
        losses = {'ar': F.cross_entropy(teacher[:, 1:].reshape(-1, teacher.shape[-1]), target[:, 1:].reshape(-1))}
        logits = self.ar(rollout)
        hard_bits = tokenizer.lfq.straight_through_ids(logits, rollout)
        if self.resampling:
            risk = self.risk_logits(rollout, uncertainty)
            errors = rollout != target
            losses['det'] = F.binary_cross_entropy_with_logits(risk[:, 1:], errors[:, 1:].float())
            supervised = errors | (torch.rand_like(risk) < self.cfg.random_mask_rate)
            supervised[:, 0] = False
            corrected_logits = self.corrector(rollout)  # self token excluded structurally
            losses['res'] = (F.cross_entropy(corrected_logits[supervised], target[supervised])
                             if supervised.any() else corrected_logits.sum()*0)
            selected = self.select(risk)
            replacement = tokenizer.lfq.straight_through_ids(corrected_logits, corrected_logits.argmax(-1))
            hard_bits = torch.where(selected[..., None], replacement, hard_bits)
        else:
            losses.update(det=logits.sum()*0, res=logits.sum()*0)
        hard_bits = torch.cat([tokenizer.lfq.to_bits(target[:, :1]), hard_bits[:, 1:]], dim=1)
        recon = tokenizer.decode_bits(hard_bits)
        recon = torch.cat([x[:, :1], recon[:, 1:]], dim=1)
        losses['mae'] = F.l1_loss(recon, x)
        # Literal Eq. (18) and Ltemp/Ldepth: sum H,W, average B,T,Z.
        losses['temp'] = (recon.diff(dim=1)-x.diff(dim=1)).abs().sum((-2, -1)).mean()
        losses['depth'] = (recon.diff(dim=2)-x.diff(dim=2)).abs().sum((-2, -1)).mean()
        losses['cyc'] = (recon[:, -1]-x[:, 0]).abs().sum((-2, -1)).mean()
        total = sum(v * (0. if k in ('temp', 'depth') and not self.consistency else self.cfg.generator_weights[k])
                    for k, v in losses.items())
        return total, losses
