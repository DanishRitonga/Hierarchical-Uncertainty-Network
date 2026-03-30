"""Polygon YOLOv26 — Loss Operations

Loss functions for polygon training.
Includes both NumPy (validation) and PyTorch (training) variants.
"""

import numpy as np

# =============================================================================
# NUMPY VARIANT (for validation/debugging)
# =============================================================================


def angular_smoothness_loss(rays: np.ndarray) -> float:
    """Compute circular first-difference penalty on ray distances.

    This encourages smooth polygon boundaries by penalizing large
    differences between consecutive rays.

    The loss is circular: the difference between ray 31 and ray 0 is included.

    Formula:
        L_smooth = (1/32) × Σ|d_{i+1} - d_i|  for i in [0, 31]
        where d_{32} = d_0 (circular)

    Args:
        rays: Ray distances, shape (32,) or (N, 32)

    Returns:
        loss: Smoothness penalty (scalar or shape (N,))
    """
    rays = np.asarray(rays, dtype=np.float64)

    if rays.ndim == 1:
        # Single prediction
        diff = np.diff(rays, append=rays[0])  # Circular difference
        return np.mean(np.abs(diff))
    else:
        # Batch of predictions
        # np.diff with append for circular
        diff = np.diff(rays, axis=1, append=rays[:, :1])
        return np.mean(np.abs(diff), axis=1)


# =============================================================================
# PYTORCH VARIANT (for training)
# =============================================================================


def angular_smoothness_loss_torch(rays):
    """PyTorch version of angular_smoothness_loss.

    Args:
        rays: Tensor of shape (..., 32)

    Returns:
        loss: Tensor of shape (...) - smoothness penalty
    """
    import torch

    # Circular difference: append first element at end
    rays_shifted = torch.roll(rays, shifts=-1, dims=-1)
    diff = rays_shifted - rays

    return torch.mean(torch.abs(diff), dim=-1)


def decode_pred_xy(xy_sigmoid, anchor_grid, stride):
    """Decode Sigmoid xy outputs to absolute normalised coordinates.

    The head outputs grid-cell-relative Sigmoid offsets in [0, 1].
    GT centroids from DataLoader are absolute normalised coordinates in [0, 1].
    This function converts predictions to absolute normalised space.

    Args:
        xy_sigmoid: Tensor of shape (B, N_anchors, 2) - Sigmoid outputs
        anchor_grid: Tensor of shape (N_anchors, 2) - anchor grid positions (normalized)
        stride: Scalar or Tensor - stride for this prediction level

    Returns:
        xy_absolute: Tensor of shape (B, N_anchors, 2) - absolute normalized coords
    """
    # Anchor grid positions are the center of each grid cell in normalized space
    # xy_sigmoid is the offset within the grid cell
    # Absolute position = anchor_center + (sigmoid - 0.5) * cell_size
    #
    # But typically in YOLO, anchor_grid is already in pixel space at each stride level,
    # and the formula is: xy_absolute = (anchor_grid + xy_sigmoid) * stride / img_size
    #
    # Simpler interpretation: anchor_grid contains grid cell indices or positions,
    # and we add the sigmoid offset, then normalize by image size.

    # Assuming anchor_grid is in normalized [0, 1] space:
    # xy_absolute = anchor_grid + (xy_sigmoid - 0.5) / (image_size / stride)
    #
    # But the standard YOLO convention is:
    # xy_pred = (grid_xy + xy_sigmoid) / grid_size
    # where grid_xy is the integer grid index

    # For simplicity, assume anchor_grid contains the grid cell centers in [0, 1]:
    return anchor_grid.unsqueeze(0) + (xy_sigmoid - 0.5) * (stride / 640.0)  # Assuming 640 image size
