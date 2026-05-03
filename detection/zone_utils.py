import numpy as np
import cv2 as cv
import supervision as sv
from typing import List


def scale_polygons(polygons: List[np.ndarray], scale: float) -> List[np.ndarray]:
    return [(poly * scale).astype(np.int32) for poly in polygons]


def is_inside_zones(xyxy: np.ndarray, polygons: List[np.ndarray]) -> bool:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    check_points = [
        (x1, y1), (x2, y2), (x1, y2), (x2, y1), (cx, cy)
    ]

    for poly in polygons:
        poly_cv = np.asarray(poly, dtype=np.int32)
        if poly_cv.ndim == 2:
            poly_cv = poly_cv.reshape(-1, 1, 2)

        for pt in check_points:
            if cv.pointPolygonTest(poly_cv, pt, False) >= 0:
                return True
    return False


def filter_detections(
        detections: sv.Detections,
        class_ids: List[int],
        polygons: List[np.ndarray]
) -> sv.Detections:
    if not polygons and not class_ids:
        return detections
    if len(detections.class_id) == 0:
        return sv.Detections.empty()

    class_mask = np.isin(detections.class_id, class_ids) if class_ids else np.ones(len(detections), dtype=bool)
    filtered = detections[class_mask]

    if not polygons or len(filtered) == 0:
        return filtered

    zone_mask = np.array([
        is_inside_zones(xyxy, polygons)
        for xyxy in filtered.xyxy
    ], dtype=bool)

    return filtered[zone_mask]