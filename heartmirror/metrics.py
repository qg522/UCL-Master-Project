"""Patient-level generation and classification metrics; never synthetic scores."""
import numpy as np
from scipy.linalg import sqrtm
from scipy.ndimage import uniform_filter


def structural_similarity(a, b, data_range=1., win_size=7):
    """2D SSIM with uniform 7x7 window and unbiased sample covariance.

    Matches skimage's default single-channel SSIM, without an additional binary
    dependency. Excludes a window-radius border from the spatial mean.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < win_size:
        raise ValueError('SSIM expects same-shape 2D images >= window size')
    ua, ub = uniform_filter(a, win_size), uniform_filter(b, win_size)
    correction = win_size**2/(win_size**2-1)
    va = correction*(uniform_filter(a*a, win_size)-ua*ua)
    vb = correction*(uniform_filter(b*b, win_size)-ub*ub)
    cov = correction*(uniform_filter(a*b, win_size)-ua*ub)
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    score = ((2*ua*ub+c1)*(2*cov+c2))/((ua**2+ub**2+c1)*(va+vb+c2))
    pad = win_size//2
    return float(score[pad:-pad, pad:-pad].mean())


def paired_metrics(real, generated):
    if real.shape != generated.shape or real.ndim != 4:
        raise ValueError('Expected aligned T,Z,H,W')
    # Conditioning ED volume is explicitly excluded, p. 6.
    real, generated = real[1:], generated[1:]
    values = [structural_similarity(a, b, data_range=1.)
              for a, b in zip(real.reshape(-1, *real.shape[-2:]), generated.reshape(-1, *generated.shape[-2:]))]
    return {'ssim': float(np.mean(values)), 'mse': float(np.mean((real-generated)**2))}


def frechet(real_features, generated_features):
    a, b = np.asarray(real_features, dtype=np.float64), np.asarray(generated_features, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(len(a), len(b)) < 2:
        raise ValueError('Frechet distance needs >=2 feature samples per distribution')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite features')
    c1, c2 = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    root = sqrtm(c1@c2)
    if not np.isfinite(root).all():
        eye = np.eye(a.shape[1])*1e-6
        root = sqrtm((c1+eye)@(c2+eye))
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag)) > 1e-3:
            raise ValueError('Unstable Frechet covariance square root')
        root = root.real
    return max(0., float(np.sum((a.mean(0)-b.mean(0))**2)+np.trace(c1+c2-2*root)))


def classification_metrics(labels, probabilities, classes):
    y, p = np.asarray(labels), np.asarray(probabilities)
    n = len(classes)
    if p.shape != (len(y), n) or not len(y) or not np.isfinite(p).all():
        raise ValueError('Invalid classification predictions')
    pred = p.argmax(1)
    confusion = np.zeros((n, n), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    tp = confusion.diagonal()
    f1 = 2*tp/np.maximum(confusion.sum(0)+confusion.sum(1), 1)
    aucs = []
    for i in range(n):
        pos, neg = p[y == i, i], p[y != i, i]
        aucs.append(float(((pos[:, None] > neg).sum()+.5*(pos[:, None] == neg).sum())/(len(pos)*len(neg)))
                    if len(pos) and len(neg) else None)
    rv = next((classes.index(c) for c in ('RV', 'ARV') if c in classes), None)
    return {'accuracy': float(np.mean(y == pred)), 'macro_f1': float(f1.mean()),
            'macro_auc': float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
            'rv_recall': float(tp[rv]/confusion[rv].sum()) if rv is not None and confusion[rv].sum() else None,
            'confusion_matrix': confusion.tolist(), 'auc_by_class': dict(zip(classes, aucs)),
            'units': 'fraction (multiply by 100 for paper tables)'}


def aggregate_runs(reports):
    """Require five independent seeds; do not count five folds as five runs."""
    seeds = [r['seed'] for r in reports]
    if len(reports) != 5 or len(set(seeds)) != 5:
        raise ValueError('Paper comparison requires five distinct seeds, not five folds')
    names = set.intersection(*(set(r['metrics']) for r in reports))
    result = {}
    for name in names:
        values = [r['metrics'][name] for r in reports]
        if all(isinstance(v, (int, float)) for v in values):
            result[name] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))}
    return {'seeds': seeds, 'metrics': result}
