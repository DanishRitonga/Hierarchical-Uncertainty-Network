"""Polygon YOLOv26 — Conversion Operations

Core conversion functions between polygon and raycast representations.
These are used during ETL ingestion and inference decoding.
"""

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from ..utils.constants import (
    CLASS_IDX,
    CX_IDX,
    CY_IDX,
    N_RAYS,
    RAY_ANGLES,
    RAY_COS,
    RAY_END_IDX,
    RAY_SIN,
    RAY_START_IDX,
)


def polygon_to_raycast(
    polygon: Polygon,
    class_id: int,
    use_representative_point_fallback: bool = True,
) -> tuple[float, float, np.ndarray, int]:
    """Convert a Shapely polygon to raycast representation.

    Args:
        polygon: Shapely Polygon object (must be valid)
        class_id: Class ID for this annotation
        use_representative_point_fallback: If True, use representative_point()
            when centroid falls outside polygon (for highly concave shapes)

    Returns:
        cx: Centroid x-coordinate (pixel space)
        cy: Centroid y-coordinate (pixel space)
        rays: Array of 32 ray distances (pixel space), shape (32,)
        class_id: Class ID (unchanged)

    Raises:
        ValueError: If polygon is invalid or empty
    """
    # Validate polygon
    if polygon is None or polygon.is_empty:
        raise ValueError('Polygon is None or empty')

    if not polygon.is_valid:
        # Attempt to fix with buffer(0)
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            raise ValueError('Polygon could not be made valid with buffer(0)')

    # Compute centroid
    centroid = polygon.centroid
    cx, cy = centroid.x, centroid.y

    # Check if centroid is inside polygon (for highly concave shapes)
    if use_representative_point_fallback:
        if not polygon.contains(Point(cx, cy)):
            # Fall back to representative point (guaranteed inside)
            rep_point = polygon.representative_point()
            cx, cy = rep_point.x, rep_point.y

    # Prepare polygon for faster ray intersection queries
    prepared_poly = prep(polygon)

    # Get bounding box for dynamic R_far
    minx, miny, maxx, maxy = polygon.bounds
    bbox_width = maxx - minx
    bbox_height = maxy - miny
    # R_far should be large enough to extend beyond polygon boundary
    R_far = max(bbox_width, bbox_height) * 2.0

    # Cast 32 rays from centroid
    rays = np.zeros(N_RAYS, dtype=np.float32)

    for i in range(N_RAYS):
        angle = RAY_ANGLES[i]
        cos_a = RAY_COS[i]
        sin_a = RAY_SIN[i]

        # Create ray line from centroid extending in direction `angle`
        # End point is far enough to cross any boundary
        end_x = cx + R_far * cos_a
        end_y = cy + R_far * sin_a

        ray_line = LineString([(cx, cy), (end_x, end_y)])

        # Find intersection with polygon boundary
        if prepared_poly.intersects(ray_line):
            intersection = polygon.boundary.intersection(ray_line)

            if intersection.is_empty:
                rays[i] = 0.0
            elif intersection.geom_type == 'Point':
                # Single intersection point
                dist = np.sqrt((intersection.x - cx) ** 2 + (intersection.y - cy) ** 2)
                rays[i] = dist
            elif intersection.geom_type == 'MultiPoint':
                # Multiple intersections (non-convex polygon)
                # Take the nearest one to centroid
                min_dist = np.inf
                for pt in intersection.geoms:
                    dist = np.sqrt((pt.x - cx) ** 2 + (pt.y - cy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                rays[i] = min_dist if min_dist < np.inf else 0.0
            elif intersection.geom_type == 'LineString':
                # Ray goes along the boundary (tangent)
                # Take the nearest point on the line
                coords = list(intersection.coords)
                min_dist = np.inf
                for coord in coords:
                    dist = np.sqrt((coord[0] - cx) ** 2 + (coord[1] - cy) ** 2)
                    if dist < min_dist and dist > 1e-6:  # Exclude centroid itself
                        min_dist = dist
                rays[i] = min_dist if min_dist < np.inf else 0.0
            elif intersection.geom_type == 'GeometryCollection':
                # Mixed geometry types
                min_dist = np.inf
                for geom in intersection.geoms:
                    if geom.geom_type == 'Point':
                        dist = np.sqrt((geom.x - cx) ** 2 + (geom.y - cy) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                rays[i] = min_dist if min_dist < np.inf else 0.0
            else:
                rays[i] = 0.0
        else:
            rays[i] = 0.0

    return cx, cy, rays, class_id


def raycast_to_annotation(
    cx: float,
    cy: float,
    rays: np.ndarray,
    class_id: int,
) -> np.ndarray:
    """Package raycast data into ETL internal format array.

    Args:
        cx: Centroid x-coordinate
        cy: Centroid y-coordinate
        rays: Array of 32 ray distances
        class_id: Class ID

    Returns:
        annotation: Array of shape (35,) in ETL format
            [cx, cy, d_1, ..., d_32, class_id]
    """
    annotation = np.zeros(35, dtype=np.float32)
    annotation[CX_IDX] = cx
    annotation[CY_IDX] = cy
    annotation[RAY_START_IDX:RAY_END_IDX] = rays
    annotation[CLASS_IDX] = class_id
    return annotation


def raycast_to_polygon(
    rays: np.ndarray,
    cx: float,
    cy: float,
) -> Polygon:
    """Convert raycast representation to a Shapely Polygon.

    Args:
        rays: Ray distances, shape (32,)
        cx: Centroid x-coordinate
        cy: Centroid y-coordinate

    Returns:
        polygon: Shapely Polygon object
    """
    rays = np.asarray(rays, dtype=np.float64)

    # Compute vertices
    vertex_x = cx + rays * RAY_COS
    vertex_y = cy + rays * (-RAY_SIN)  # Image coordinates

    # Create polygon from vertices
    coords = list(zip(vertex_x, vertex_y))

    return Polygon(coords)


def decode_to_vertices(
    rays: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> np.ndarray:
    """Decode ray distances to Cartesian polygon vertices.

    Args:
        rays: Ray distances, shape (N, 32) in pixel space
        cx: Centroid x-coordinates, shape (N,)
        cy: Centroid y-coordinates, shape (N,)

    Returns:
        vertices: Polygon vertices, shape (N, 32, 2) in pixel space
            vertices[i, j, 0] = x-coordinate of j-th vertex of i-th polygon
            vertices[i, j, 1] = y-coordinate of j-th vertex of i-th polygon
    """
    rays = np.asarray(rays, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)

    n_polygons = rays.shape[0]

    # Compute vertex coordinates
    # vertex_x = cx + d_i * cos(θ_i)
    # vertex_y = cy + d_i * sin(θ_i)
    # Note: Using image coordinates where Y points down

    # rays: (N, 32)
    # RAY_COS, RAY_SIN: (32,)
    # cx, cy: (N,)

    # Broadcast: (N, 32) * (32,) = (N, 32)
    vertex_x = cx[:, np.newaxis] + rays * RAY_COS[np.newaxis, :]
    vertex_y = cy[:, np.newaxis] + rays * (-RAY_SIN[np.newaxis, :])  # Negate for image coords

    # Stack to (N, 32, 2)
    vertices = np.stack([vertex_x, vertex_y], axis=2)

    return vertices.astype(np.float32)
