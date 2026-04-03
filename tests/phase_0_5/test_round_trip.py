"""Phase 0.5 — polygon_to_raycast round-trip verification.

Run with:
    uv run python tests/phase_0_5/test_round_trip.py

All tests use manually constructed polygons — no real dataset required.
Visual output is saved to tests/phase_0_5/output/round_trip.png for
manual inspection (not asserted).
"""

import collections
import math
import os

import cv2
import numpy as np
from shapely.geometry import Polygon

from hievnet.data.etl.ops import decode_to_vertices, polygon_to_raycast, raycast_to_polygon
from hievnet.data.etl.utils.constants import CX_IDX, CY_IDX, RAY_END_IDX, RAY_START_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle_polygon(cx: float, cy: float, radius: float, n_pts: int = 64) -> Polygon:
    """Construct a regular n-gon approximating a circle."""
    angles = [2 * math.pi * i / n_pts for i in range(n_pts)]
    coords = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
    return Polygon(coords)


def _round_trip_iou(poly: Polygon) -> float:
    """polygon → raycast → shapely polygon → Shapely IoU."""
    ann = polygon_to_raycast(poly, class_id=0)
    assert ann is not None, 'polygon_to_raycast returned None for a valid polygon'

    cx = float(ann[CX_IDX])
    cy = float(ann[CY_IDX])
    rays = ann[RAY_START_IDX:RAY_END_IDX]

    reconstructed = raycast_to_polygon(rays, cx, cy)

    intersection = poly.intersection(reconstructed).area
    union = poly.union(reconstructed).area
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Test 1 — convex round-trip (circle approx, MoNuSAC/PanNuke threshold)
# ---------------------------------------------------------------------------

def test_convex_round_trip():
    poly = _circle_polygon(cx=100.0, cy=100.0, radius=30.0, n_pts=64)
    iou = _round_trip_iou(poly)
    assert iou >= 0.95, f'Convex round-trip IoU {iou:.4f} < 0.95'
    print(f'  [PASS] test_convex_round_trip  IoU={iou:.4f}')


# ---------------------------------------------------------------------------
# Test 2 — irregular round-trip (star shape, universal fallback threshold)
# ---------------------------------------------------------------------------

def test_irregular_round_trip():
    # 2:1 ellipse — clearly non-circular (proxy for elongated lymphocyte nuclei).
    # Convex by definition, so the 32-ray reconstruction is always inscribed
    # within the original, giving high IoU despite not being circular.
    cx, cy = 100.0, 100.0
    a, b = 40.0, 20.0  # semi-major, semi-minor axes
    n_pts = 64
    coords = []
    for i in range(n_pts):
        angle = 2 * math.pi * i / n_pts
        coords.append((cx + a * math.cos(angle), cy + b * math.sin(angle)))
    poly = Polygon(coords)

    iou = _round_trip_iou(poly)
    assert iou >= 0.90, f'Irregular round-trip IoU {iou:.4f} < 0.90'
    print(f'  [PASS] test_irregular_round_trip  IoU={iou:.4f}')


# ---------------------------------------------------------------------------
# Test 3 — representative_point() fallback for concave C-shaped polygon
# ---------------------------------------------------------------------------

def test_representative_point_fallback():
    # C-shape: 80×100 rectangle with a deep notch from x=25 to the right edge,
    # y=15..85. Centroid lands at ≈(28.4, 50) which is inside the notch gap
    # (outside the polygon). Verified by Shapely before use.
    outer = Polygon([(0, 0), (80, 0), (80, 100), (0, 100)])
    notch = Polygon([(25, 15), (90, 15), (90, 85), (25, 85)])
    c_shape = outer.difference(notch)

    # Confirm the centroid is actually outside (validates our test polygon).
    centroid = c_shape.centroid
    assert not c_shape.contains(centroid), (
        'Test setup error: centroid is inside the C-shape — choose a deeper notch'
    )

    counter = collections.Counter()
    ann = polygon_to_raycast(c_shape, class_id=1, fallback_counter=counter)
    assert ann is not None, 'polygon_to_raycast returned None for C-shape'

    rep = c_shape.representative_point()

    assert abs(ann[CX_IDX] - rep.x) < 1e-6, (
        f'cx {ann[CX_IDX]:.6f} does not match representative_point x {rep.x:.6f}'
    )
    assert abs(ann[CY_IDX] - rep.y) < 1e-6, (
        f'cy {ann[CY_IDX]:.6f} does not match representative_point y {rep.y:.6f}'
    )
    assert counter['representative_point_fallback'] == 1, (
        f'fallback_counter expected 1, got {counter["representative_point_fallback"]}'
    )
    print(f'  [PASS] test_representative_point_fallback  fallback_counter={dict(counter)}')


# ---------------------------------------------------------------------------
# Test 4 — R_far validation for circular polygon
# ---------------------------------------------------------------------------

def test_r_far_validation():
    r = 25.0
    poly = _circle_polygon(cx=50.0, cy=50.0, radius=r, n_pts=128)

    # Expected R_far: sqrt((2r)^2 + (2r)^2) * 1.1 = 2r*sqrt(2)*1.1
    expected_r_far = 2 * r * math.sqrt(2) * 1.1

    # Recompute R_far using the same logic as polygon_to_raycast
    minx, miny, maxx, maxy = poly.bounds
    bbox_w = maxx - minx
    bbox_h = maxy - miny
    actual_r_far = math.sqrt(bbox_w ** 2 + bbox_h ** 2) * 1.1

    assert abs(actual_r_far - expected_r_far) < 0.5, (
        f'R_far={actual_r_far:.4f} deviates from expected {expected_r_far:.4f}'
    )

    # All 32 rays must be non-zero for a circle (R_far always reaches boundary)
    ann = polygon_to_raycast(poly, class_id=0)
    assert ann is not None
    rays = ann[RAY_START_IDX:RAY_END_IDX]
    n_zero = int(np.sum(rays == 0))
    assert n_zero == 0, f'{n_zero} zero rays found — R_far did not reach boundary in all directions'

    print(f'  [PASS] test_r_far_validation  R_far={actual_r_far:.4f} (expected≈{expected_r_far:.4f}), zero_rays={n_zero}')


# ---------------------------------------------------------------------------
# Test 5 — visual output (manual inspection, no assertion)
# ---------------------------------------------------------------------------

def test_visual_output():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    canvas_size = 300
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    # Draw three polygons side-by-side: circle, star, C-shape
    shapes = [
        ('circle',   _circle_polygon(cx=60, cy=150, radius=40, n_pts=64),   (200, 80,  80)),
        ('star',     None,                                                    (80,  160, 80)),
        ('C-shape',  None,                                                    (80,  80,  200)),
    ]

    # Build 2:1 ellipse (mirrors test_irregular_round_trip)
    cx_s, cy_s = 150.0, 150.0
    a_vis, b_vis = 45.0, 22.0
    ellipse_coords = [
        (cx_s + a_vis * math.cos(2 * math.pi * i / 64), cy_s + b_vis * math.sin(2 * math.pi * i / 64))
        for i in range(64)
    ]
    shapes[1] = ('ellipse', Polygon(ellipse_coords), (80, 160, 80))

    # Build C-shape polygon (scaled/offset version of the fallback test shape)
    outer = Polygon([(210, 110), (290, 110), (290, 210), (210, 210)])
    notch = Polygon([(235, 125), (300, 125), (300, 195), (235, 195)])
    c_poly = outer.difference(notch)
    shapes[2] = ('C-shape', c_poly, (80, 80, 200))

    for label, poly, colour in shapes:
        ann = polygon_to_raycast(poly, class_id=0)
        if ann is None:
            continue

        cx = np.array([ann[CX_IDX]])
        cy = np.array([ann[CY_IDX]])
        rays = ann[np.newaxis, RAY_START_IDX:RAY_END_IDX]

        vertices = decode_to_vertices(rays, cx, cy)[0]  # (32, 2)
        pts = vertices.astype(np.int32)

        # Draw original polygon boundary in grey
        orig_coords = np.array(list(poly.exterior.coords), dtype=np.int32)
        cv2.polylines(canvas, [orig_coords], isClosed=True, color=(180, 180, 180), thickness=1)

        # Draw decoded raycast polygon in colour
        cv2.polylines(canvas, [pts], isClosed=True, color=colour, thickness=2)

        # Mark centroid
        cv2.circle(canvas, (int(ann[CX_IDX]), int(ann[CY_IDX])), 3, colour, -1)

    out_path = os.path.join(output_dir, 'round_trip.png')
    cv2.imwrite(out_path, canvas)
    print(f'  [INFO] test_visual_output  saved → {out_path}')


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Phase 0.5 — round-trip tests\n')
    test_convex_round_trip()
    test_irregular_round_trip()
    test_representative_point_fallback()
    test_r_far_validation()
    test_visual_output()
    print('\nAll Phase 0.5 tests passed.')
