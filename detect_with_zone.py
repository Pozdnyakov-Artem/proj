import numpy as np
import cv2 as cv
import time, sys, os
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import torch
from PIL import Image

in_file = r"C:\Users\gegan\Videos\cam25-1.avi"
create_out_file = True
out_file = 'out.avi'

scale = 0.5
threshold = 0.5

PERSON_CLASS_ID = 1

ZONE_POLYGONS = [
    np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300],
        [50,200]
    ], dtype=np.int32),
    np.array([
        [400, 200],
        [600, 200],
        [600, 400],
        [400, 400]
    ], dtype=np.int32)
]

ZONE_COLOR = (0, 255, 255)
ZONE_ALPHA = 0.3

def segments_intersect(p1, p2, p3, p4):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
            return True
        return False

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False


def is_inside_zones(xyxy, polygons):
    x1, y1, x2, y2 = xyxy

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    bbox_corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max)
    ]

    bbox_edges = [
        (bbox_corners[0], bbox_corners[1]),
        (bbox_corners[1], bbox_corners[2]),
        (bbox_corners[2], bbox_corners[3]),
        (bbox_corners[3], bbox_corners[0])
    ]

    for poly in polygons:
        for corner in bbox_corners:
            dist = cv.pointPolygonTest(poly, corner, False)
            if dist >= 0:
                return True

        for vertex in poly:
            vx, vy = vertex[0], vertex[1]
            if x_min <= vx <= x_max and y_min <= vy <= y_max:
                return True

        poly_vertices = poly.tolist() if hasattr(poly, 'tolist') else poly
        n = len(poly_vertices)

        for i in range(n):
            p1 = tuple(poly_vertices[i])
            p2 = tuple(poly_vertices[(i + 1) % n])
            poly_edge = (p1, p2)

            for bbox_edge in bbox_edges:
                if segments_intersect(bbox_edge[0], bbox_edge[1], poly_edge[0], poly_edge[1]):
                    return True

    return False


def filter_detections(detections, class_id, polygons):
    if len(detections.class_id) == 0:
        return detections

    class_mask = detections.class_id == class_id

    if not np.any(class_mask):
        return sv.Detections.empty()

    people_detections = detections[class_mask]

    if len(people_detections) == 0:
        return sv.Detections.empty()

    zone_mask = []
    for xyxy in people_detections.xyxy:
        zone_mask.append(is_inside_zones(xyxy, polygons))

    zone_mask = np.array(zone_mask, dtype=bool)

    final_detections = people_detections[zone_mask]

    return final_detections


def main():
    cv.setNumThreads(1)

    model = RFDETRBase()
    model.optimize_for_inference()

    video_capture = cv.VideoCapture(in_file)
    if not video_capture.isOpened():
        print('Could not open video')
        sys.exit()

    w = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv.CAP_PROP_FPS)

    wh = int(w * scale)
    hh = int(h * scale)

    print(f"Original: {w}x{h}, Scaled: {wh}x{hh}")

    if create_out_file:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_out = cv.VideoWriter(out_file, fourcc, fps, (wh, hh))

    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
        "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(wh, hh))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(wh, hh))

    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        smart_position=True
    )

    fn = 0
    paused = False

    zone_overlay = np.zeros((hh, wh, 3), dtype=np.uint8)
    if len(ZONE_POLYGONS) > 0:
        cv.fillPoly(zone_overlay, ZONE_POLYGONS, ZONE_COLOR)

    while True:
        t0 = time.perf_counter()

        if not paused:
            ret, frame = video_capture.read()
            if not ret:
                break
            fn += 1
            frameh = cv.resize(frame, None, fx=scale, fy=scale)
            frame_rgb = cv.cvtColor(frameh, cv.COLOR_BGR2RGB)
        else:
            t_show_start = time.perf_counter()
            key = cv.waitKey(100)

            if key == ord('p'):
                paused = False
            continue

        dt0 = time.perf_counter() - t0

        t1 = time.perf_counter()

        image_pil = Image.fromarray(frame_rgb)

        all_detections = model.predict(image_pil, threshold=threshold)

        dt1 = time.perf_counter() - t1

        t2 = time.perf_counter()

        detections = filter_detections(all_detections, PERSON_CLASS_ID, ZONE_POLYGONS)

        nob = len(detections.class_id)
        print(f'Frame {fn} people in zones: {nob}')

        labels = [
            f"{COCO_CLASSES[class_id]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated_image = frameh.copy()

        if len(ZONE_POLYGONS) > 0:
            cv.addWeighted(zone_overlay, ZONE_ALPHA, annotated_image, 1 - ZONE_ALPHA, 0, annotated_image)
            cv.polylines(annotated_image, ZONE_POLYGONS, True, ZONE_COLOR, 2)

        if nob > 0:
            annotated_image = bbox_annotator.annotate(annotated_image, detections)
            annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        dt2 = time.perf_counter() - t2

        t_show_start = time.perf_counter()
        cv.imshow('Real-time Detection (Zones Active)', annotated_image)
        key = cv.waitKey(1)
        dt3 = time.perf_counter() - t_show_start

        if create_out_file:
            video_out.write(annotated_image)

        tt = dt0 + dt1 + dt2 + dt3

        if fn % 10 == 0:
            print(
                f'{fn} | Pre: {dt0:.4f}s | Detect: {dt1:.4f}s | Annot: {dt2:.4f}s | Total: {tt:.4f}s')

        if key == ord('q'):
            break
        if key == ord('p'):
            paused = True

    cv.destroyAllWindows()
    video_capture.release()

    if create_out_file:
        video_out.release()


if __name__ == '__main__':
    main()