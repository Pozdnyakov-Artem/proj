import numpy as np
import cv2 as cv
import time, sys, os
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import torch

vfile = 'path_to_your_input_video.avi'
createovf = True
ovfile = 'out.avi'
scale = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
threshold = 0.5
max_display_objs = 10


def main():
    cv.setNumThreads(1)

    print("Loading RFDETR model...")
    model = RFDETRBase()

    if device == 'cuda':
        model.model.to(device)  # Перенос модели на CUDA

    vco = cv.VideoCapture(vfile)
    if not vco.isOpened():
        print('Could not open video. Exiting.')
        sys.exit()

    w = int(vco.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(vco.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = vco.get(cv.CAP_PROP_FPS)
    print(f'Video width={w}, height={h}, fps={fps}')

    wh = int(w * scale)
    hh = int(h * scale)

    if createovf:
        fourcc = cv.VideoWriter_fourcc(*'XVID')  # Кодек для AVI
        vwo = cv.VideoWriter(ovfile, fourcc, fps, (wh, hh))

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

    print("Starting detection. Press 'q' to quit, 'p' to pause/resume.")

    while True:
        t0 = time.perf_counter()

        if not paused:
            ret, frame = vco.read()
            if not ret:
                print('Video is finished or cannot be read further')
                break
            fn += 1
            frameh = cv.resize(frame, None, fx=scale, fy=scale)
            frame_rgb = cv.cvtColor(frameh, cv.COLOR_BGR2RGB)
        else:
            key = cv.waitKey(100)
            if key == ord('p'):
                paused = False
            continue

        dt0 = time.perf_counter() - t0

        t1 = time.perf_counter()

        from PIL import Image
        image_pil = Image.fromarray(frame_rgb)
        detections = model.predict(image_pil, threshold=threshold)

        dt1 = time.perf_counter() - t1

        t2 = time.perf_counter()

        nob = len(detections.class_id)
        print(f'Frame {fn} objects detected: {nob}')

        if nob > max_display_objs:
            sorted_idx = np.argsort(detections.confidence)[::-1][:max_display_objs]
            detections.class_id = detections.class_id[sorted_idx]
            detections.confidence = detections.confidence[sorted_idx]
            detections.xyxy = detections.xyxy[sorted_idx]
            nob = max_display_objs

        labels = [
            f"{COCO_CLASSES[class_id]} {conf:.2f}"
            for class_id, conf
            in zip(detections.class_id, detections.confidence)
        ]

        annotated_image = frameh.copy()
        if nob > 0:
            annotated_image = bbox_annotator.annotate(annotated_image, detections)
            annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        dt2 = time.perf_counter() - t2

        t3 = time.perf_counter()

        cv.imshow('RFDETR Real-time Detection', annotated_image)

        if createovf:
            vwo.write(annotated_image)

        dt3 = time.perf_counter() - t3
        tt = dt0 + dt1 + dt2 + dt3

        # Вывод статистики по времени (каждые 10 кадров, чтобы не засорять консоль)
        if fn % 10 == 0:
            print(
                f'{fn} | Pre: {dt0:.4f}s | Detect: {dt1:.4f}s | Annot: {dt2:.4f}s | Show: {dt3:.4f}s | Total: {tt:.4f}s')

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = True
            print("Paused. Press 'p' to resume.")

    cv.destroyAllWindows()
    vco.release()
    if createovf:
        vwo.release()
    print(f"Processing complete! {fn} frames processed.")
    print(f"Output saved to: {ovfile}" if createovf else "No output file created.")


if __name__ == '__main__':
    main()