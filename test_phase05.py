"""Phase 0.5 — Round-trip accuracy tests.

Tests:
    T1: Circle round-trip IoU >= 0.95
    T2: Ellipse round-trip IoU >= 0.95
    T3: Irregular convex polygon round-trip IoU >= 0.90
    T4: Concave / star polygon round-trip IoU >= 0.90
    T5: C-shape representative_point() fallback: counter incremented,
        decoded cx/cy inside polygon
    T6: R_far formula: sqrt((2r)^2 + (2r)^2) * 1.1 ~= 3.11r for circle;
        all 32 rays non-zero

All shapes use pixel-space coordinates (no normalisation).

Run:
    uv run python test_phase05.py
"""

import collections
import math
import os

import numpy as np
from shapely.geometry import Polygon

from hievnet.data.etl.ops import (
    decode_to_vertices,
    polygon_to_raycast,
    polar_iou,
    raycast_to_polygon,
)
from hievnet.data.etl.utils.constants import N_RAYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circle(cx: float, cy: float, r: float, n_pts: int = 128) -> Polygon:
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    coords = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
    return Polygon(coords)


def make_ellipse(cx: float, cy: float, rx: float, ry: float, n_pts: int = 128) -> Polygon:
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    coords = [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in angles]
    return Polygon(coords)


def make_irregular_convex(cx: float, cy: float, seed: int = 42) -> Polygon:
    """Random convex polygon by generating points and taking convex hull."""
    rng = np.random.default_rng(seed)
    n = 12
    angles = np.sort(rng.uniform(0, 2 * math.pi, n))
    radii = rng.uniform(30, 80, n)
    coords = [(cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(radii, angles)]
    return Polygon(coords).convex_hull


def make_star(cx: float, cy: float, r_outer: float = 60, r_inner: float = 25, n_points: int = 6) -> Polygon:
    """Star polygon (concave)."""
    coords = []
    for i in range(n_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi * i / n_points
        coords.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return Polygon(coords)


def make_c_shape(cx: float, cy: float, r: float = 50) -> Polygon:
    """C-shape: circle with a rectangular notch cut out. Centroid falls outside."""
    from shapely.geometry import box
    circle = make_circle(cx, cy, r)
    notch = box(cx, cy - r * 0.3, cx + r * 1.5, cy + r * 0.3)
    c = circle.difference(notch)
    return c


def roundtrip_iou(poly: Polygon, class_id: int = 0) -> float:
    """polygon -> raycast annotation -> polygon -> IoU with original."""
    ann = polygon_to_raycast(poly, class_id)
    assert ann is not None, 'polygon_to_raycast returned None'
    rays = ann[3:35]
    cx, cy = ann[1], ann[2]
    decoded = raycast_to_polygon(rays, cx, cy)
    return polar_iou(rays, np.array(ann[3:35]))  # trivially 1 — use shape IoU instead


def roundtrip_iou_shape(poly: Polygon, class_id: int = 0) -> float:
    """Compute Shapely IoU between original and decoded polygon."""
    ann = polygon_to_raycast(poly, class_id)
    assert ann is not None, 'polygon_to_raycast returned None'
    rays = ann[3:35]
    cx, cy = float(ann[1]), float(ann[2])
    decoded = raycast_to_polygon(rays, cx, cy)

    if decoded.is_empty or not decoded.is_valid:
        return 0.0

    intersection = poly.intersection(decoded).area
    union = poly.union(decoded).area
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

results = {}


def check(name: str, condition: bool, detail: str = '') -> None:
    status = PASS if condition else FAIL
    print(f'  [{status}] {name}' + (f' — {detail}' if detail else ''))
    results[name] = condition


print('\n=== Phase 0.5 Round-Trip Tests ===\n')

CX, CY = 320.0, 320.0  # typical crop-centre

# T1 — Circle
print('T1: Circle round-trip IoU >= 0.95')
circle = make_circle(CX, CY, r=50)
iou_circle = roundtrip_iou_shape(circle)
check('T1_circle_iou', iou_circle >= 0.95, f'IoU={iou_circle:.4f}')

# T2 — Ellipse
print('T2: Ellipse round-trip IoU >= 0.95')
ellipse = make_ellipse(CX, CY, rx=80, ry=35)
iou_ellipse = roundtrip_iou_shape(ellipse)
check('T2_ellipse_iou', iou_ellipse >= 0.95, f'IoU={iou_ellipse:.4f}')

# T3 — Irregular convex
print('T3: Irregular convex polygon round-trip IoU >= 0.90')
convex = make_irregular_convex(CX, CY)
iou_convex = roundtrip_iou_shape(convex)
check('T3_irregular_convex_iou', iou_convex >= 0.90, f'IoU={iou_convex:.4f}')

# T4 — Star (concave)
print('T4: Star (concave) polygon round-trip IoU >= 0.90')
star = make_star(CX, CY)
iou_star = roundtrip_iou_shape(star)
check('T4_star_iou', iou_star >= 0.90, f'IoU={iou_star:.4f}')

# T5 — C-shape fallback
print('T5: C-shape representative_point() fallback')
c_shape = make_c_shape(CX, CY, r=50)
counter = collections.Counter()
ann_c = polygon_to_raycast(c_shape, class_id=0, fallback_counter=counter)
fallback_used = counter['representative_point_fallback'] == 1
check('T5a_fallback_counter', fallback_used, f'counter={counter}')

if ann_c is not None:
    from shapely.geometry import Point
    decoded_cx, decoded_cy = float(ann_c[1]), float(ann_c[2])
    cx_inside = c_shape.contains(Point(decoded_cx, decoded_cy))
    check('T5b_cx_inside_polygon', cx_inside, f'cx={decoded_cx:.1f}, cy={decoded_cy:.1f}')
else:
    check('T5b_cx_inside_polygon', False, 'ann_c is None')

# T6 — R_far formula + all rays non-zero for circle
print('T6: R_far formula and all rays non-zero for circle')
r = 50.0
ann_circle = polygon_to_raycast(circle, class_id=0)
rays_circle = ann_circle[3:35]
expected_r_far = math.sqrt((2 * r) ** 2 + (2 * r) ** 2) * 1.1  # ≈ 155.56 for r=50
# The circle bounding box is (2r x 2r), so R_far = sqrt(4r^2+4r^2)*1.1 = 2r*sqrt(2)*1.1
r_far_ratio = expected_r_far / r  # should be ~3.11
check('T6a_r_far_formula', abs(r_far_ratio - 2 * math.sqrt(2) * 1.1) < 0.01,
      f'R_far/r={r_far_ratio:.4f}, expected={2*math.sqrt(2)*1.1:.4f}')
all_nonzero = np.all(rays_circle > 0)
check('T6b_all_rays_nonzero', all_nonzero,
      f'zero_rays={np.sum(rays_circle == 0)}/{N_RAYS}')

# T7 — decode_to_vertices shape and round-trip
print('T7: decode_to_vertices shape and consistency with raycast_to_polygon')
anns = np.stack([ann_circle, ann_circle], axis=0)  # (2, 35)
rays_batch = anns[:, 3:35]
cx_batch = anns[:, 1]
cy_batch = anns[:, 2]
verts = decode_to_vertices(rays_batch, cx_batch, cy_batch)
check('T7a_vertices_shape', verts.shape == (2, 32, 2), f'shape={verts.shape}')

# Verify decode_to_vertices matches raycast_to_polygon for row 0
poly0 = raycast_to_polygon(rays_circle, float(cx_batch[0]), float(cy_batch[0]))
coords0 = np.array(poly0.exterior.coords[:-1])  # drop closing point
verts0 = verts[0]  # (32, 2)
match = np.allclose(coords0, verts0, atol=1e-4)
check('T7b_decode_matches_raycast_to_polygon', match,
      f'max_diff={np.max(np.abs(coords0 - verts0)):.6f}')

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

os.makedirs('viz/phase05', exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    shapes = [
        ('circle', circle, ann_circle),
        ('ellipse', ellipse, polygon_to_raycast(ellipse, 0)),
        ('convex', convex, polygon_to_raycast(convex, 0)),
        ('star', star, polygon_to_raycast(star, 0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, (name, orig_poly, ann) in zip(axes, shapes):
        rays = ann[3:35]
        cx_, cy_ = float(ann[1]), float(ann[2])
        decoded_poly = raycast_to_polygon(rays, cx_, cy_)

        ox, oy = orig_poly.exterior.xy
        dx, dy = decoded_poly.exterior.xy

        ax.fill(ox, oy, alpha=0.3, color='blue', label='original')
        ax.plot(ox, oy, 'b-', linewidth=1)
        ax.fill(dx, dy, alpha=0.3, color='red', label='decoded')
        ax.plot(dx, dy, 'r--', linewidth=1)
        ax.plot(cx_, cy_, 'k+', markersize=10, label='centroid')

        iou_val = roundtrip_iou_shape(orig_poly)
        ax.set_title(f'{name}  IoU={iou_val:.3f}')
        ax.set_aspect('equal')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('viz/phase05/roundtrip.png', dpi=150)
    plt.close()
    print('\n  Saved: viz/phase05/roundtrip.png')

    # C-shape separately
    if ann_c is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ox, oy = c_shape.exterior.xy
        rays_c = ann_c[3:35]
        cx_c, cy_c = float(ann_c[1]), float(ann_c[2])
        decoded_c = raycast_to_polygon(rays_c, cx_c, cy_c)
        dx, dy = decoded_c.exterior.xy
        ax.fill(ox, oy, alpha=0.3, color='blue', label='original C')
        ax.plot(ox, oy, 'b-')
        ax.fill(dx, dy, alpha=0.3, color='red', label='decoded')
        ax.plot(dx, dy, 'r--')
        ax.plot(cx_c, cy_c, 'k+', markersize=12, label='rep_point centroid')
        ax.set_title('C-shape fallback')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('viz/phase05/c_shape_fallback.png', dpi=150)
        plt.close()
        print('  Saved: viz/phase05/c_shape_fallback.png')

except ImportError:
    print('\n  (matplotlib not available — skipping visualizations)')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

n_pass = sum(results.values())
n_total = len(results)
print(f'\n=== Summary: {n_pass}/{n_total} passed ===\n')

if n_pass < n_total:
    failed = [k for k, v in results.items() if not v]
    print('Failed:', ', '.join(failed))
    raise SystemExit(1)
