"""Patient manifests, ED-aligned preprocessing and sequence-coherent augmentation."""
import csv
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, zoom
import torch
from torch.utils.data import Dataset

CLASSES = ('NOR', 'DCM', 'HCM', 'MINF', 'RV')


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        for key in ('patient_id', 'dataset', 'path', 'group', 'ed_index', 'tokenizer_only'):
            if key not in r:
                raise ValueError(f'Manifest missing {key}')
        r['path'] = str((path.parent / r['path']).resolve())
        r['key'] = r['dataset'] + ':' + r['patient_id']
        if r['key'] in seen:
            raise ValueError('Duplicate patient: '+r['key'])
        seen.add(r['key'])
        r['ed_index'] = int(r['ed_index'])  # explicitly zero based
        r['tokenizer_only'] = str(r['tokenizer_only']).lower() in ('1', 'true', 'yes')
        if not r['group'] and not r['tokenizer_only']:
            raise ValueError('Unlabeled subjects must be tokenizer_only')
    return rows


def acdc_manifest(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    rows = []
    for folder in sorted((root/'database').glob('*/patient*')):
        info = {}
        for line in (folder/'Info.cfg').read_text().splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
        rows.append(dict(patient_id=folder.name, dataset='ACDC',
                         path=str(folder/(folder.name+'_4d.nii.gz')), group=info['Group'],
                         ed_index=int(info['ED'])-1, tokenizer_only=0))
    if len(rows) != 150:
        raise ValueError(f'Expected 150 ACDC patients, found {len(rows)}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_splits(rows, dataset, seed=0, classes=CLASSES):
    """Five disjoint test folds; ACDC NOR = exactly 20 train / 4 val / 6 test.

    Classification partition ratios other than four-shot are unspecified in the
    PDF. We use stratified five folds, four validation cases/class, and four
    support cases/class. Generator NOR train/val/test shares those partitions.
    """
    rows = [r for r in rows if r['dataset'] == dataset and not r['tokenizer_only']]
    rng = np.random.default_rng(seed)
    by_class = {}
    for group in classes:
        ids = sorted(r['key'] for r in rows if r['group'] == group)
        if len(ids) < 10:
            raise ValueError(f'{group}: need enough subjects for disjoint 4-shot/val/test')
        rng.shuffle(ids)
        by_class[group] = list(np.array_split(ids, 5))
    folds = []
    for fold in range(5):
        support, validation, test = [], [], []
        gen = {}
        for group, chunks in by_class.items():
            heldout = list(chunks[fold])
            rest = [str(v) for j, c in enumerate(chunks) if j != fold for v in c]
            rng.shuffle(rest)
            val, train = rest[:4], rest[4:]
            if len(train) < 4:
                raise ValueError('Not enough four-shot support patients')
            support.extend(train[:4]); validation.extend(val); test.extend(heldout)
            if group == 'NOR':
                gen = {'train': train, 'val': val, 'test': heldout}
                if dataset == 'ACDC' and list(map(len, (train, val, heldout))) != [20, 4, 6]:
                    raise ValueError('ACDC requires exactly 30 NOR subjects')
        folds.append({'generator': gen, 'classifier': {'train': support, 'val': validation, 'test': test}})
    return {'seed': seed, 'dataset': dataset, 'classes': list(classes), 'folds': folds,
            'classifier_partition_assumption': 'stratified 5-fold, 4 val and 4 support per class'}


def center_crop_pad(arr, hw):
    h, w = arr.shape[-2:]
    arr = arr[..., max(0, (h-hw)//2):max(0, (h-hw)//2)+min(hw, h),
              max(0, (w-hw)//2):max(0, (w-hw)//2)+min(hw, w)]
    dh, dw = hw-arr.shape[-2], hw-arr.shape[-1]
    return np.pad(arr, [(0, 0)]*(arr.ndim-2)+[(dh//2, dh-dh//2), (dw//2, dw-dw//2)])


def load_cine(record, shape):
    """Returns T,Z,H,W; H/W are center cropped/padded, never resized.

    Intensity limits come ONLY from ED, so generating a reference cannot obtain
    patient-specific information from future frame intensity extrema.
    """
    if str(record['path']).endswith('.npy'):
        # BU-4DFE has no voxelization recipe in the PDF. Accept explicitly
        # prepared T,Z,H,W voxel arrays, never reinterpret raw meshes as MRI.
        array = np.load(record['path'], allow_pickle=False).astype(np.float32)
        if array.ndim != 4:
            raise ValueError('Prepared BU4DFE .npy must be T,Z,H,W')
        img = nib.Nifti1Image(array.transpose(2, 3, 1, 0), np.eye(4))
    else:
        img = nib.load(record['path'])
    source = img.get_fdata(dtype=np.float32)
    if source.ndim != 4 or not np.isfinite(source).all():
        raise ValueError(f'Invalid 4D cine: {record["path"]}')
    ed = record['ed_index']
    if not 0 <= ed < source.shape[-1]:
        raise ValueError('ed_index outside cine')
    x = np.roll(source.transpose(3, 2, 0, 1), -ed, axis=0)
    lo, hi = float(x[0].min()), float(x[0].max())
    x = np.clip((x-lo)/max(hi-lo, 1e-6), 0, 1)
    t, z, h, w = shape
    x = zoom(x, (t/x.shape[0], z/x.shape[1], 1, 1), order=1, prefilter=False)
    x = center_crop_pad(x, h)
    if h != w:
        raise ValueError('Square target required')
    # scipy zoom aligns voxel centres at endpoints; preserve spatial geometry.
    transform = np.eye(4)
    for axis, size in enumerate(source.shape[:2]):
        transform[axis, 3] = max(0, (size-h)//2)-max(0, (h-size)//2)
    transform[2, 2] = (source.shape[2]-1)/max(z-1, 1)
    meta = {'affine': (img.affine @ transform).tolist(), 'source_shape': list(source.shape),
            'source_path': record['path'], 'ed_index': ed, 'intensity_limits': [lo, hi],
            'temporal_spacing': float(img.header.get_zooms()[3])*(source.shape[3]-1)/max(t-1, 1),
            'xyzt_units': list(img.header.get_xyzt_units())}
    return np.ascontiguousarray(x, dtype=np.float32), meta


def augment_sequence(x, rng):
    """Five views use fixed per-subject/view RNG; transforms shared across T,Z."""
    angle = rng.uniform(-5, 5)*np.pi/180
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s], [s, c]])
    center = (np.array(x.shape[-2:])-1)/2
    offset = center-matrix@center+rng.uniform(-4, 4, 2)
    out = np.empty_like(x)
    for t in range(len(x)):
        for z in range(x.shape[1]):
            out[t, z] = affine_transform(x[t, z], matrix, offset, order=1, mode='constant')
    out = np.clip(out*rng.uniform(.9, 1.1), 0, 1)**rng.uniform(.9, 1.1)
    sigma = rng.uniform(0, .02)
    out = np.sqrt((out+rng.normal(0, sigma, out.shape))**2+rng.normal(0, sigma, out.shape)**2)
    # Monotone phase warp with endpoints fixed (ED remains first).
    coord = (np.linspace(0, 1, len(x))**rng.uniform(.9, 1.1))*(len(x)-1)
    low = np.floor(coord).astype(int); high = np.minimum(low+1, len(x)-1)
    weight = (coord-low)[:, None, None, None]
    return np.clip(out[low]*(1-weight)+out[high]*weight, 0, 1).astype(np.float32)


class CineDataset(Dataset):
    def __init__(self, rows, ids, cfg, views=1, seed=0, reference_dir=None, classes=CLASSES):
        index = {r['key']: r for r in rows}
        if len(set(ids)) != len(ids) or any(i not in index for i in ids):
            raise ValueError('Duplicate or missing manifest IDs')
        self.rows = [index[i] for i in ids]
        self.cfg, self.views, self.seed = cfg, views, seed
        self.reference_dir, self.classes = reference_dir, tuple(classes)

    def __len__(self):
        return len(self.rows)*self.views

    def __getitem__(self, i):
        row_idx, view = divmod(i, self.views)
        r = self.rows[row_idx]
        x, _ = load_cine(r, (self.cfg.phases, self.cfg.slices, self.cfg.image_size, self.cfg.image_size))
        if self.views > 1:
            x = augment_sequence(x, np.random.default_rng(self.seed+row_idx*1009+view))
        item = {'cine': torch.from_numpy(x), 'key': r['key'],
                'label': self.classes.index(r['group']) if r['group'] in self.classes else -1}
        if self.reference_dir:
            file = Path(self.reference_dir)/r['dataset']/(r['patient_id']+'.npz')
            with np.load(file, allow_pickle=False) as f:
                reference = f['cine'].astype(np.float32)
                meta = json.loads(str(f['metadata']))
            if meta['patient_key'] != r['key'] or reference.shape != x.shape:
                raise ValueError(f'Reference alignment mismatch: {file}')
            if r['key'] in meta.get('generator_train_ids', [])+meta.get('generator_val_ids', []):
                raise ValueError(f'Healthy reference must be out of fold: {r["key"]}')
            if meta.get('seed') != self.seed:
                raise ValueError('Reference seed does not match support split')
            if r['key'] in meta.get('tokenizer_seen_ids', []):
                raise ValueError('Reference tokenizer saw this evaluation patient')
            if not np.allclose(reference[0], x[0], atol=1e-5):
                raise ValueError('Reference must preserve the observed ED volume')
            item['reference'] = torch.from_numpy(reference)
        return item
