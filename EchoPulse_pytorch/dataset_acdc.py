"""ACDC 4D cine-MRI dataset for ECHOPulse migration.

Output video shape: (C=1, T*D=310, H=192, W=192)
  - Per-time-step ordering: [t=0,d=0..9], [t=1,d=0..9], ...
  - So the first D=10 tokens form the *first volume* used for prime-frame conditioning.
ECG: constant ones(12, 2250) -- shape matches the original paper; no z-score.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
import nibabel as nib
from scipy.ndimage import zoom

ACDC_GROUPS = ('NOR', 'DCM', 'HCM', 'MINF', 'RV')


def _load_labels(root):
    """Concat training + testing labels with absolute patient folder paths."""
    root = Path(root)
    rows = []
    for split, csv_dir, sub_dir in (
        ('train', 'training label/training_labels.csv', 'training'),
        ('test',  'testing label/testing_labels.csv',   'testing'),
    ):
        df = pd.read_csv(root / csv_dir)
        df['orig_split'] = split
        df['folder'] = df['patient_id'].apply(
            lambda pid: str(root / 'database' / sub_dir / pid))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _load_4d_volume(folder, pid, target_shape=(31, 10, 192, 192)):
    """Load patientXXX_4d.nii.gz and resample to target (T, D, H, W).

    nibabel returns (H_raw, W_raw, D_raw, T_raw); we transpose to (T, D, H, W)
    and trilinear-resample with scipy.ndimage.zoom.
    Per-volume min-max normalised to [0, 1].
    """
    path = Path(folder) / f'{pid}_4d.nii.gz'
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    # (H_raw, W_raw, D_raw, T_raw) -> (T, D, H, W)
    vol = np.transpose(vol, (3, 2, 0, 1))
    T_, D_, H_, W_ = vol.shape
    Tt, Dt, Ht, Wt = target_shape
    if (T_, D_, H_, W_) != (Tt, Dt, Ht, Wt):
        vol = zoom(vol, (Tt / T_, Dt / D_, Ht / H_, Wt / W_), order=1)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    return vol  # (T, D, H, W) in [0, 1]


class ACDCDataset(Dataset):
    """ACDC dataset.

    split:
      - 'train': ALL NOR patients (train+test merged) = 30 cases
      - 'test':  ALL non-NOR patients (train+test merged) = 120 cases
      - 'all':   everyone
    Returns:
      video: (1, T*D, H, W) float in [0, 1]
      ekg:   (12, 2250) all-ones float
      group: str (one of ACDC_GROUPS)  -- useful for evaluation
    """

    def __init__(
        self,
        root,
        split='train',
        T=31,
        D=10,
        H=192,
        W=192,
        groups=None,
        ekg_len=2250,
        ekg_channels=12,
        cache_in_ram=True,    # pre-decode all volumes once -> avoid I/O bottleneck
    ):
        super().__init__()
        labels = _load_labels(root)
        if split == 'train':
            labels = labels[labels.group == 'NOR']
        elif split == 'test':
            labels = labels[labels.group != 'NOR']
        elif split == 'all':
            pass
        else:
            raise ValueError(f'unknown split {split}')
        if groups is not None:
            labels = labels[labels.group.isin(groups)]
        self.items = labels.reset_index(drop=True).to_dict('records')
        self.target_shape = (T, D, H, W)
        self.ekg_len = ekg_len
        self.ekg_channels = ekg_channels
        self._ekg_cached = torch.ones(ekg_channels, ekg_len, dtype=torch.float32)

        self._cache = None
        if cache_in_ram:
            # decode all 4D volumes once at init; cheap for ACDC (30 NOR -> ~700 MB).
            print(f'[ACDCDataset/{split}] pre-caching {len(self.items)} volumes...')
            self._cache = []
            for rec in self.items:
                vol = _load_4d_volume(rec['folder'], rec['patient_id'], self.target_shape)
                T_, D_, H_, W_ = vol.shape
                flat = vol.reshape(T_ * D_, H_, W_)
                self._cache.append(torch.from_numpy(flat).float().unsqueeze(0))
            print(f'[ACDCDataset/{split}] cache done '
                  f'(~{sum(v.element_size()*v.nelement() for v in self._cache)/1e9:.2f} GB)')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        if self._cache is not None:
            video = self._cache[idx]
        else:
            vol = _load_4d_volume(rec['folder'], rec['patient_id'], self.target_shape)
            T_, D_, H_, W_ = vol.shape
            flat = vol.reshape(T_ * D_, H_, W_)
            video = torch.from_numpy(flat).float().unsqueeze(0)
        return video, self._ekg_cached, rec['group'], rec['patient_id']


def _collate(batch):
    videos, ekgs, groups, pids = zip(*batch)
    return (
        torch.stack(videos, dim=0),
        torch.stack(ekgs, dim=0),
        list(groups),
        list(pids),
    )


def _collate_train(batch):
    """For training we only need (video, ekg) so the existing trainer's
    `data_tuple_to_kwargs` (which uses DATASET_FIELD_TYPE_CONFIG) still works."""
    videos, ekgs, _, _ = zip(*batch)
    return torch.stack(videos, dim=0), torch.stack(ekgs, dim=0)


def DataLoader(dataset, *, batch_size, num_workers=0, pin_memory=True, train=True, **kwargs):
    return PytorchDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_train if train else _collate,
        **kwargs,
    )


if __name__ == '__main__':
    ds = ACDCDataset(root=os.path.join(os.path.dirname(__file__), '..', 'ACDC'),
                     split='train')
    print(f'#NOR train samples: {len(ds)}')
    v, e, g, pid = ds[0]
    print(f'video {v.shape} dtype={v.dtype} range=[{v.min():.3f},{v.max():.3f}]')
    print(f'ekg   {e.shape} dtype={e.dtype} unique={e.unique().tolist()}')
    print(f'group={g} pid={pid}')
