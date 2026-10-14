"""Classification inputs from materialized external original/reference/difference triples."""
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def audit_paired_data(root, rows, ids, cfg, classes):
    root = Path(root).resolve()
    with (root/'pairs.csv').open(encoding='utf-8-sig', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    index = {r['patient_key']: r for r in pairs}
    if len(index) != len(pairs):
        raise ValueError('Duplicate paired patient keys')
    manifest = {r['key']: r for r in rows}
    shape = (cfg.image_size, cfg.image_size, cfg.slices, cfg.phases)
    for key in ids:
        if key not in index or key not in manifest:
            raise ValueError('Missing paired patient: '+key)
        pair, row = index[key], manifest[key]
        if pair['label'] != row['group'] or int(pair['label_id']) != classes.index(row['group']):
            raise ValueError('Paired diagnosis mismatch: '+key)
        images = [nib.load(root/pair[k]) for k in ('original_path', 'generated_path', 'difference_path')]
        if any(im.shape != shape for im in images):
            raise ValueError('Paired shape does not match classifier config: '+key)
        if any(not np.allclose(im.affine, images[0].affine) for im in images[1:]):
            raise ValueError('Paired affine mismatch: '+key)
    audit = json.loads((root/'audit.json').read_text(encoding='utf-8'))
    return {'method': 'external_synthesis_all/acdc', 'paired_root': str(root),
            'alignment_protocol': audit['alignment_protocol'],
            'reference_training_provenance': 'unknown',
            'paper_reproduction_certified': False}, index


class PairedCineDataset(Dataset):
    def __init__(self, rows, ids, cfg, root, classes, cache=False):
        self.root = Path(root).resolve()
        self.classes = tuple(classes)
        self.provenance, index = audit_paired_data(root, rows, ids, cfg, self.classes)
        self.pairs = [index[key] for key in ids]
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        if self.cache is not None and i in self.cache:
            item = self.cache[i]
            return {**item, 'difference': item['cine']-item['reference']}
        pair = self.pairs[i]
        arrays = [nib.load(self.root/pair[k]).get_fdata(dtype=np.float32).transpose(3, 2, 0, 1)
                  for k in ('original_path', 'generated_path', 'difference_path')]
        original, reference, difference = arrays
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Non-finite paired data: '+pair['patient_key'])
        if not np.allclose(difference, original-reference, atol=1e-6, rtol=1e-5):
            raise ValueError('Stored signed difference must equal original - generated')
        item = {'cine': torch.from_numpy(np.ascontiguousarray(original)),
                'reference': torch.from_numpy(np.ascontiguousarray(reference)),
                'difference': torch.from_numpy(np.ascontiguousarray(difference)),
                'label': int(pair['label_id']), 'key': pair['patient_key']}
        if self.cache is not None:
            self.cache[i] = {k: v for k, v in item.items() if k != 'difference'}
        return item
