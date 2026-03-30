"""Polygon YOLOv26 — Polar IoU Operations

Polar-IoU computation for star-convex polygons.
Includes both NumPy (ETL) and PyTorch (training) variants.

Formula:
    PolarIoU = Σ min(d_pred_i, d_gt_i)² / Σ max(d_pred_i, d_gt_i)²

This is a sector-area approximation from PolarMask.
"""

import numpy as np

from ..utils.constants import POLAR_IOU_EPS

# =============================================================================
# NUMPY VARIANTS (for ETL)
# =============================================================================


def polar_iou(
    d_pred: np.ndarray,
    d_gt: np.ndarray,
    eps: float = POLAR_IOU_EPS,
) -> float:
    """Compute Polar-IoU between two ray sets.

    Polar-IoU approximates polygon IoU by treating each ray as defining
    a circular sector. This is exact for circles and tight for near-circular shapes.

    Formula:
        PolarIoU = Σ min(d_pred_i, d_gt_i)² / Σ max(d_pred_i, d_gt_i)²

    This is a sector-area approximation where each sector has equal angular width.

    Args:
        d_pred: Predicted ray distances, shape (32,)
        d_gt: Ground truth ray distances, shape (32,)
        eps: Small constant to prevent division by zero

    Returns:
        iou: Polar-IoU value in [0, 1]
    """
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_gt = np.asarray(d_gt, dtype=np.float64)

    # Sector areas are proportional to d² (sector_area = 0.5 * r² * θ)
    # The constant 0.5 * θ cancels in the ratio
    pred_sq = d_pred**2
    gt_sq = d_gt**2

    intersection = np.sum(np.minimum(pred_sq, gt_sq))
    union = np.sum(np.maximum(pred_sq, gt_sq))

    return intersection / (union + eps)


def polar_iou_pairwise(
    d_pred: np.ndarray,
    d_gt: np.ndarray,
    eps: float = POLAR_IOU_EPS,
) -> np.ndarray:
    """Compute pairwise Polar-IoU between sets of predictions and ground truths.

    Args:
        d_pred: Predicted ray distances, shape (N_pred, 32)
        d_gt: Ground truth ray distances, shape (N_gt, 32)
        eps: Small constant to prevent division by zero

    Returns:
        iou_matrix: IoU matrix, shape (N_pred, N_gt)
    """
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_gt = np.asarray(d_gt, dtype=np.float64)

    n_pred = d_pred.shape[0]
    n_gt = d_gt.shape[0]

    # Expand for broadcasting: (n_pred, 1, 32) and (1, n_gt, 32)
    pred_sq = d_pred[:, :, np.newaxis] ** 2 if d_pred.ndim == 2 else d_pred**2
    gt_sq = d_gt[np.newaxis, :, :] ** 2 if d_gt.ndim == 2 else d_gt**2

    # This is memory-intensive; use the flat version for large arrays
    pred_sq = d_pred[:, np.newaxis, :] ** 2  # (N_pred, 1, 32)
    gt_sq = d_gt[np.newaxis, :, :] ** 2  # (1, N_gt, 32)

    intersection = np.sum(np.minimum(pred_sq, gt_sq), axis=2)  # (N_pred, N_gt)
    union = np.sum(np.maximum(pred_sq, gt_sq), axis=2)  # (N_pred, N_gt)

    return intersection / (union + eps)


def polar_iou_pairwise_flat(
    d_pred: np.ndarray,
    d_gt: np.ndarray,
    eps: float = POLAR_IOU_EPS,
) -> np.ndarray:
    """Memory-efficient pairwise Polar-IoU for flattened inputs.

    This is the version used by the assigner, where d_pred has already been
    filtered to only candidate anchors.

    Args:
        d_pred: Predicted ray distances, shape (N_cand, 32)
        d_gt: Ground truth ray distances, shape (N_gt, 32)
        eps: Small constant to prevent division by zero

    Returns:
        iou_matrix: IoU matrix, shape (N_cand, N_gt)
    """
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_gt = np.asarray(d_gt, dtype=np.float64)

    n_cand = d_pred.shape[0]
    n_gt = d_gt.shape[0]

    if n_cand == 0 or n_gt == 0:
        return np.zeros((n_cand, n_gt), dtype=np.float64)

    # Square the ray distances
    pred_sq = d_pred**2  # (N_cand, 32)
    gt_sq = d_gt**2  # (N_gt, 32)

    # Compute pairwise using broadcasting
    # This creates (N_cand, N_gt, 32) intermediate
    # For memory efficiency with large arrays, could loop over chunks

    intersection = np.sum(np.minimum(pred_sq[:, np.newaxis, :], gt_sq[np.newaxis, :, :]), axis=2)  # (N_cand, N_gt)

    union = np.sum(np.maximum(pred_sq[:, np.newaxis, :], gt_sq[np.newaxis, :, :]), axis=2)  # (N_cand, N_gt)

    return intersection / (union + eps)


# =============================================================================
# PYTORCH VARIANTS (for training)
# =============================================================================


def polar_iou_torch(d_pred, d_gt, eps=POLAR_IOU_EPS):
    """PyTorch version of polar_iou for use in loss function.

    Args:
        d_pred: Tensor of shape (..., 32) - predicted rays
        d_gt: Tensor of shape (..., 32) - ground truth rays
        eps: Small constant to prevent division by zero

    Returns:
        iou: Tensor of shape (...) - Polar-IoU
    """
    import torch

    pred_sq = d_pred**2
    gt_sq = d_gt**2

    intersection = torch.sum(torch.minimum(pred_sq, gt_sq), dim=-1)
    union = torch.sum(torch.maximum(pred_sq, gt_sq), dim=-1)

    return intersection / (union + eps)


def polar_iou_pairwise_torch(d_pred, d_gt, eps=POLAR_IOU_EPS):
    """PyTorch version of polar_iou_pairwise for use in assigner.

    Memory-efficient implementation using chunking if needed.

    Args:
        d_pred: Tensor of shape (N_pred, 32)
        d_gt: Tensor of shape (N_gt, 32)
        eps: Small constant to prevent division by zero

    Returns:
        iou_matrix: Tensor of shape (N_pred, N_gt)
    """
    import torch

    n_pred = d_pred.shape[0]
    n_gt = d_gt.shape[0]

    if n_pred == 0 or n_gt == 0:
        return torch.zeros((n_pred, n_gt), device=d_pred.device, dtype=d_pred.dtype)

    pred_sq = d_pred**2  # (N_pred, 32)
    gt_sq = d_gt**2  # (N_gt, 32)

    # Expand for broadcasting
    # pred_sq: (N_pred, 1, 32)
    # gt_sq: (1, N_gt, 32)
    intersection = torch.sum(torch.minimum(pred_sq.unsqueeze(1), gt_sq.unsqueeze(0)), dim=2)  # (N_pred, N_gt)

    union = torch.sum(torch.maximum(pred_sq.unsqueeze(1), gt_sq.unsqueeze(0)), dim=2)  # (N_pred, N_gt)

    return intersection / (union + eps)
