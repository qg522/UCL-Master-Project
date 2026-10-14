"""Reported settings are separated from unspecified implementation choices."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    # Explicitly reported in Experimental Setup (pp. 5-6).
    phases: int = 30
    slices: int = 10
    image_size: int = 240
    epochs: tuple = (200, 300, 50)
    batch_sizes: tuple = (32, 32, 8)
    lr: float = 1e-4
    views: int = 5
    shots: int = 4
    seeds: tuple = (0, 1, 2, 3, 4)  # five runs reported, actual seeds unspecified
    # Engineering defaults: these values are NOT specified in the supplied PDF.
    dim: int = 128
    heads: int = 4
    bits: int = 10
    patch: int = 30
    spatial_layers: int = 2
    temporal_layers: int = 2
    ar_layers: int = 4
    corrector_layers: int = 2
    classifier_layers: int = 4  # four ViT layers in Fig. 1
    classifier_patch: tuple = (5, 2, 48, 48)
    classifier_encoder_layers: int = 2
    token_temperature: float = 1.0
    sample_temperature: float = 1.0
    risk_threshold: float = 0.5
    max_corrections: int = 30
    random_mask_rate: float = 0.15
    contrast_temperature: float = 0.1
    focal_gamma: float = 2.0
    contrast_weight: float = 0.1
    weight_decay: float = 0.01
    slice_chunk: int = 32
    gradient_checkpointing: bool = True
    tokenizer_weights: dict = field(default_factory=lambda: {'rec': 1., 'per': .1, 'lfq': .1})
    generator_weights: dict = field(default_factory=lambda: {
        'ar': 1., 'det': 1., 'res': 1., 'mae': 1., 'temp': 1., 'depth': 1., 'cyc': 1.})

    def validate(self):
        if self.image_size % self.patch or self.dim % self.heads:
            raise ValueError('image_size must divide by patch; dim must divide by heads')
        if self.phases < 3 or self.slices < 3 or not 1 <= self.bits <= 16:
            raise ValueError('Require phases/slices >=3 and 1<=bits<=16')
        if any(n % p for n, p in zip(
                (self.phases, self.slices, self.image_size, self.image_size), self.classifier_patch)):
            raise ValueError('Classifier patches must divide T,Z,H,W')
        if not 0 <= self.random_mask_rate <= 1 or self.sample_temperature <= 0:
            raise ValueError('Invalid mask rate / sampling temperature')
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path=None):
        return cls(**(json.loads(Path(path).read_text(encoding='utf-8')) if path else {})).validate()
