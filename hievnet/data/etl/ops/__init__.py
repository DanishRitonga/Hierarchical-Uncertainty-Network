"""Polygon YOLOv26 — Operations Module

Single source of truth for all polygon/raycast geometry logic.
This module is imported by:
    - hievnet/data/ingestors/*.py (NumPy, offline ETL)
    - hievnet/data/transform/*.py (NumPy, offline ETL)
    - hievnet/data/loader/polygon_dataset.py (NumPy, online DataLoader)
    - ultralytics/utils/loss.py (PyTorch, training)
    - ultralytics/utils/tal.py (PyTorch, assignment)
    - ultralytics/models/yolo/detect/predict.py (PyTorch, inference)

PyTorch functions use LAZY IMPORTS (import torch inside function body)
so this module can be imported in ETL environments without PyTorch installed.

Module Structure:
    - convert: Polygon ↔ raycast conversion
    - filter: Annotation filtering and clipping
    - iou: Polar-IoU computation (NumPy + PyTorch)
    - loss: Loss functions (NumPy + PyTorch)
    - augment: Geometric augmentations
    - utils: Validation and quality monitoring
"""

# Conversion operations
# Augmentation operations
from .augment import (
    flip_horizontal,
    flip_vertical,
    rotate_90,
)
from .convert import (
    decode_to_vertices,
    polygon_to_raycast,
    raycast_to_annotation,
    raycast_to_polygon,
)

# Filtering operations
from .filter import filter_and_clip_annotations

# IoU operations (NumPy + PyTorch)
from .iou import (
    polar_iou,
    polar_iou_pairwise,
    polar_iou_pairwise_flat,
    polar_iou_pairwise_torch,
    polar_iou_torch,
)

# Loss operations (NumPy + PyTorch)
from .loss import (
    angular_smoothness_loss,
    angular_smoothness_loss_torch,
    decode_pred_xy,
)

# Utility operations
from .utils import (
    count_zero_rays,
    validate_annotation_format,
)

__all__ = [
    # Convert
    'polygon_to_raycast',
    'raycast_to_annotation',
    'raycast_to_polygon',
    'decode_to_vertices',
    # Filter
    'filter_and_clip_annotations',
    # IoU
    'polar_iou',
    'polar_iou_pairwise',
    'polar_iou_pairwise_flat',
    'polar_iou_torch',
    'polar_iou_pairwise_torch',
    # Loss
    'angular_smoothness_loss',
    'angular_smoothness_loss_torch',
    'decode_pred_xy',
    # Augment
    'flip_horizontal',
    'flip_vertical',
    'rotate_90',
    # Utils
    'count_zero_rays',
    'validate_annotation_format',
]
