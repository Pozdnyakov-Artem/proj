import numpy as np
import cv2 as cv
import time, sys, os
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import torch
from PIL import Image

in_file = r"C:\Users\gegan\Videos\cam25-1.avi"
create_out_file = True # сохранять или нет обработку
out_file = 'out.avi'

scale = 0.5 #уменьшение размеров для повышения скорости,
# нужно посмотреть при каких значения находит всех людей, при 0,5 в дали через кадр находит людей

threshold = 0.5 # граница после которой детектит объект

#перенос на gpu не нужен моделька сама детектит и переносит

def main():
    cv.setNumThreads(1) # ограничение на 1 поток

    model = RFDETRBase()

    video_capture = cv.VideoCapture(in_file)
    if not video_capture.isOpened():
        print('Could not open video')
        sys.exit()

    w = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv.CAP_PROP_FPS)

    # размеры после scale
    wh = int(w * scale)
    hh = int(h * scale)

    # создание объекта для записи видео
    if create_out_file:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_out = cv.VideoWriter(out_file, fourcc, fps, (wh, hh))

    # настройка цветов для рамок
    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
        "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])

    # подгон текста и толщины линии под кадр
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(wh, hh))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(wh, hh))

    # для отрисовки bbox
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        smart_position=True
    )

    fn = 0  # счётчик кадров
    paused = False

    while True:
        t0 = time.perf_counter()    # начало захвата кадра

        if not paused:
            ret, frame = video_capture.read()
            if not ret:
                break
            fn += 1
            frameh = cv.resize(frame, None, fx=scale, fy=scale) # уменьшение как указано в scale
            frame_rgb = cv.cvtColor(frameh, cv.COLOR_BGR2RGB)
        else:
            key = cv.waitKey(100)
            if key == ord('p'):
                paused = False
            continue

        dt0 = time.perf_counter() - t0 # конец захвата кадра

        t1 = time.perf_counter()    # начало детекции

        image_pil = Image.fromarray(frame_rgb)
        detections = model.predict(image_pil, threshold=threshold)

        dt1 = time.perf_counter() - t1 # конец детекции

        t2 = time.perf_counter()    # начало обработки

        nob = len(detections.class_id) # количество найденных объектов
        print(f'Frame {fn} objects detected: {nob}')

        if nob > 0: # тут просто сортировка по conf сначала в которых наиболее уверена модель
            sorted_idx = np.argsort(detections.confidence)[::-1]
            detections.class_id = detections.class_id[sorted_idx]
            detections.confidence = detections.confidence[sorted_idx]
            detections.xyxy = detections.xyxy[sorted_idx]

        # создание подписей
        labels = [
            f"{class_id}  {COCO_CLASSES[class_id]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        # наложение рамок и текста
        annotated_image = frameh.copy()
        if nob > 0:
            annotated_image = bbox_annotator.annotate(annotated_image, detections)
            annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        dt2 = time.perf_counter() - t2  # конец обработки

        cv.imshow('Real-time Detection', annotated_image)

        if create_out_file:
            video_out.write(annotated_image)

        tt = dt0 + dt1 + dt2 # общее время

        # каждые 10 кадров выводиться статистика
        if fn % 10 == 0:
            print(
                f'{fn} | Pre: {dt0:.4f}s | Detect: {dt1:.4f}s | Annot: {dt2:.4f}s | Total: {tt:.4f}s')

        key = cv.waitKey(1)
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