"""Explicit source-to-export transforms inferred from ACDC conditioning volumes."""
import numpy as np
import torch
from torch.nn import functional as F


def spatial_transform(source, hw=128, depth=16):
    """H,W,Z,T -> T,Z,H,W; bilinear H/W resize and centered Z crop/pad."""
    x = torch.from_numpy(np.ascontiguousarray(source.transpose(3, 2, 0, 1)))
    t, z, h, w = x.shape
    x = F.interpolate(x.reshape(t*z, 1, h, w), size=(hw, hw),
                      mode='bilinear', align_corners=False)[:, 0].reshape(t, z, hw, hw)
    if z > depth:
        start = (z-depth)//2
        x = x[:, start:start+depth]
    elif z < depth:
        before = (depth-z)//2
        x = F.pad(x, (0, 0, 0, 0, before, depth-z-before))
    return x.numpy()


def temporal_resample(x, phases):
    """Linear endpoint-preserving phase grid, with no motion registration."""
    coordinates = np.linspace(0, len(x)-1, phases)
    low = np.floor(coordinates).astype(int)
    high = np.minimum(low+1, len(x)-1)
    weight = (coordinates-low).astype(np.float32)[:, None, None, None]
    return np.ascontiguousarray(x[low]*(1-weight)+x[high]*weight, dtype=np.float32)


def aligned_affine(original_affine, original_shape, hw=128, depth=16):
    transform = np.eye(4)
    for axis in (0, 1):
        scale = original_shape[axis]/hw
        transform[axis, axis] = scale
        transform[axis, 3] = (scale-1)/2
    z = original_shape[2]
    transform[2, 3] = max(0, (z-depth)//2)-max(0, (depth-z)//2)
    return original_affine @ transform


def align_pair(raw, generated, ed_index, phases=30, hw=128, depth=16):
    if generated.shape != (hw, hw, depth, phases):
        raise ValueError('Unexpected generated grid')
    if not np.isfinite(raw).all() or not np.isfinite(generated).all():
        raise ValueError('Non-finite source voxels')
    original = spatial_transform(raw, hw, depth)
    lo, hi = float(original.min()), float(original.max())
    if hi <= lo:
        raise ValueError('Constant source cine')
    original = (original-lo)/(hi-lo)
    # Exported generated H/W are swapped; restore original image orientation.
    reference = np.ascontiguousarray(generated.transpose(3, 2, 1, 0))
    correlation = float(np.corrcoef(original[0].ravel(), reference[0].ravel())[0, 1])
    first_mae = float(np.abs(original[0]-reference[0]).mean())
    if not np.isfinite(correlation) or correlation < .995:
        raise ValueError(f'Conditioning-volume transform failed: correlation={correlation}')
    original = temporal_resample(original, phases)
    # Both exports start at raw frame 0, not necessarily ED (patient090 ED=3).
    mapped_ed = int(np.floor(ed_index*(phases-1)/max(raw.shape[-1]-1, 1)+.5))
    original = np.ascontiguousarray(np.roll(original, -mapped_ed, axis=0))
    reference = np.ascontiguousarray(np.roll(reference, -mapped_ed, axis=0))
    difference = original-reference
    return original, reference, difference, dict(
        conditioning_volume_correlation=correlation, conditioning_volume_mae=first_mae,
        source_intensity_limits=[lo, hi], source_ed_index=ed_index, phase_roll=mapped_ed,
        spatial_alignment='verified_by_conditioning_volume',
        temporal_alignment='endpoint_grid_assumption_not_exact_generator_recipe',
        intensity_alignment='source_resized_cine_minmax_inferred',
        difference_definition='original - generated')
