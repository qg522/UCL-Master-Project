"""Load only the packaged test partition, with shared geometry and signed differences."""
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class TestCineDataset(Dataset):
    def __init__(self, root, cfg, classes):
        self.root = Path(root).resolve()
        self.classes = tuple(classes)
        self.shape = (cfg.image_size, cfg.image_size, cfg.slices, cfg.phases)
        with (self.root / 'test.csv').open(encoding='utf-8-sig', newline='') as f:
            self.rows = list(csv.DictReader(f))
        ids = json.loads((self.root / 'test_ids.json').read_text())
        keys = [r['patient_key'] for r in self.rows]
        if not keys or len(keys) != len(set(keys)) or keys != ids:
            raise ValueError('Manifest must contain exactly the ordered packaged test IDs')
        for row in self.rows:
            if row['label'] not in self.classes:
                raise ValueError('Unknown class: ' + row['label'])
            for name in ('original_path', 'reference_path'):
                path = (self.root / row[name]).resolve()
                if self.root not in path.parents or not path.is_file():
                    raise ValueError('Missing or invalid data path: ' + str(path))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = [nib.load(self.root / row[k]) for k in ('original_path', 'reference_path')]
        if any(im.shape != self.shape for im in images):
            raise ValueError('Expected aligned H,W,Z,T shape ' + str(self.shape))
        if not np.allclose(images[0].affine, images[1].affine):
            raise ValueError('Observed/reference affine mismatch')
        arrays = [im.get_fdata(dtype=np.float32).transpose(3, 2, 0, 1) for im in images]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Non-finite input')
        observed, reference = [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]
        return dict(cine=observed, reference=reference, difference=observed-reference,
                    key=row['patient_key'], label=self.classes.index(row['label']))
