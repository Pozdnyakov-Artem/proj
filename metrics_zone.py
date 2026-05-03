import numpy as np
import cv2 as cv
import time, sys, os
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import torch
from PIL import Image
from collections import defaultdict

# ================= НАСТРОЙКИ =================
in_file = r"/home/julia/Рабочий стол/snow.mp4"
labels_dir = (
    r"/home/julia/Рабочий стол/runs/detect/track-9/labels"  # Папка с аннотациями
)
create_out_file = True
out_file = "out.avi"
metrics_file = "snow_metrics_report.txt"

scale = 0.5
threshold = 0.5
iou_threshold = 0.5  # Порог IoU для совпадения детекции с GT

PERSON_CLASS_ID = 1  # COCO class_id для person
# ===============================================


def load_ground_truth(frame_idx, w, h):
    """
    Загрузка GT аннотаций для кадра.
    Формат: class cx cy w h track_id (нормализованные координаты)
    Возвращает: список словарей {xyxy, class_id, track_id}
    """
    gt_path = os.path.join(labels_dir, f"snow_{frame_idx}.txt")
    gt_boxes = []

    if not os.path.exists(gt_path):
        return gt_boxes

    try:
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cls, cx, cy, bw, bh, track_id = map(float, parts[:6])

                # Денормализация координат
                cx_px, cy_px = cx * w, cy * h
                bw_px, bh_px = bw * w, bh * h
                x1 = cx_px - bw_px / 2
                y1 = cy_px - bh_px / 2
                x2 = cx_px + bw_px / 2
                y2 = cy_px + bh_px / 2
                xyxy = np.array([x1, y1, x2, y2])

                # Убрана проверка на полигон, добавляем все GT объекты
                gt_boxes.append(
                    {"xyxy": xyxy, "class_id": int(cls), "track_id": int(track_id)}
                )
    except Exception as e:
        print(f"Warning: Error reading {gt_path}: {e}")

    return gt_boxes


def calculate_iou(box1, box2):
    """IoU между двумя bbox [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_dice(box1, box2):
    """Dice коэффициент между двумя bbox"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return 2 * intersection / (area1 + area2) if (area1 + area2) > 0 else 0


def match_detections_to_gt(pred_detections, gt_boxes, iou_thresh=0.5):
    """
    Сопоставление предсказаний с GT.
    Приоритет: совпадение track_id + IoU > threshold.
    Возвращает: TP, FP, FN, list_of_iou_scores, list_of_dice_scores
    """
    if len(pred_detections) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, [], []

    # Группируем предсказания по track_id (если есть)
    pred_by_track = {}
    for i, (xyxy, conf, cls) in enumerate(
        zip(pred_detections.xyxy, pred_detections.confidence, pred_detections.class_id)
    ):
        track_id = (
            int(pred_detections.tracker_id[i])
            if pred_detections.tracker_id is not None
            else -i - 1
        )
        pred_by_track[track_id] = {
            "xyxy": xyxy,
            "class_id": cls,
            "conf": conf,
            "idx": i,
        }

    matched_pred = set()
    matched_gt = set()
    iou_scores = []
    dice_scores = []

    # Сначала пытаемся сопоставить по track_id
    for gt in gt_boxes:
        gt_track = gt["track_id"]
        if gt_track in pred_by_track and gt_track not in [
            p["track_id"]
            for _, p in zip(matched_gt, gt_boxes)
            if hasattr(p, "track_id")
        ]:
            pred = pred_by_track[gt_track]
            iou = calculate_iou(gt["xyxy"], pred["xyxy"])
            if iou >= iou_thresh:
                matched_gt.add(gt_track)
                matched_pred.add(gt_track)
                iou_scores.append(iou)
                dice_scores.append(calculate_dice(gt["xyxy"], pred["xyxy"]))

    # Затем сопоставляем оставшиеся по максимальному IoU
    for i, gt in enumerate(gt_boxes):
        if i in matched_gt:
            continue
        best_iou = 0
        best_pred_idx = None
        for j, (xyxy, conf, cls) in enumerate(
            zip(
                pred_detections.xyxy,
                pred_detections.confidence,
                pred_detections.class_id,
            )
        ):
            if j in matched_pred:
                continue
            iou = calculate_iou(gt["xyxy"], xyxy)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j

        if best_iou >= iou_thresh and best_pred_idx is not None:
            matched_gt.add(i)
            matched_pred.add(best_pred_idx)
            iou_scores.append(best_iou)
            dice_scores.append(
                calculate_dice(gt["xyxy"], pred_detections.xyxy[best_pred_idx])
            )

    tp = len(matched_gt)
    fp = len(pred_detections) - len([p for p in matched_pred])
    fn = len(gt_boxes) - len(matched_gt)

    return tp, max(0, fp), max(0, fn), iou_scores, dice_scores


def filter_detections(detections, class_id):
    """Фильтрация детекций только по классу (без полигона)"""
    if len(detections.class_id) == 0:
        return detections
    class_mask = detections.class_id == class_id
    if not np.any(class_mask):
        return sv.Detections.empty()
    return detections[class_mask]


def main():
    cv.setNumThreads(1)

    # Инициализация метрик
    total_tp, total_fp, total_fn = 0, 0, 0
    all_iou_scores, all_dice_scores = [], []
    frame_metrics = []

    model = RFDETRBase()
    model.optimize_for_inference()

    video_capture = cv.VideoCapture(in_file)
    if not video_capture.isOpened():
        print("Could not open video")
        sys.exit()

    w = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv.CAP_PROP_FPS)
    wh, hh = int(w * scale), int(h * scale)

    print(f"Original: {w}x{h}, Scaled: {wh}x{hh}")

    if create_out_file:
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        video_out = cv.VideoWriter(out_file, fourcc, fps, (wh, hh))

    color = sv.ColorPalette.from_hex(
        [
            "#ffff00",
            "#ff9b00",
            "#ff8080",
            "#ff66b2",
            "#ff66ff",
            "#b266ff",
            "#9999ff",
            "#3399ff",
            "#66ffff",
            "#33ff99",
            "#66ff66",
            "#99ff00",
        ]
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(wh, hh))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(wh, hh))
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        smart_position=True,
    )

    fn = 0
    paused = False
    # Убрано создание zone_overlay

    print(f"\n{'='*60}")
    print(
        f"Начало тестирования. Порог детекции: {threshold}, IoU matching: {iou_threshold}"
    )
    print(f"{'='*60}\n")

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
            key = cv.waitKey(100)
            if key == ord("p"):
                paused = False
            continue

        dt0 = time.perf_counter() - t0
        t1 = time.perf_counter()

        # Инференс модели
        image_pil = Image.fromarray(frame_rgb)
        all_detections = model.predict(image_pil, threshold=threshold)
        dt1 = time.perf_counter() - t1
        t2 = time.perf_counter()

        # Фильтрация детекций только по классу
        detections = filter_detections(all_detections, PERSON_CLASS_ID)

        # Загрузка GT для текущего кадра
        gt_boxes = load_ground_truth(fn, wh, hh)

        # Сопоставление и подсчёт метрик
        tp, fp, fn_gt, iou_scores, dice_scores = match_detections_to_gt(
            detections, gt_boxes, iou_threshold
        )

        # Накопление метрик
        total_tp += tp
        total_fp += fp
        total_fn += fn_gt
        all_iou_scores.extend(iou_scores)
        all_dice_scores.extend(dice_scores)

        # Метрики для текущего кадра
        prec_frame = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_frame = tp / (tp + fn_gt) if (tp + fn_gt) > 0 else 0
        iou_frame = np.mean(iou_scores) if iou_scores else 0
        dice_frame = np.mean(dice_scores) if dice_scores else 0

        frame_metrics.append(
            {
                "frame": fn,
                "tp": tp,
                "fp": fp,
                "fn": fn_gt,
                "precision": prec_frame,
                "recall": rec_frame,
                "iou": iou_frame,
                "dice": dice_frame,
            }
        )

        nob = len(detections.class_id)
        print(
            f"Frame {fn} | Pred: {nob} | GT: {len(gt_boxes)} | TP: {tp} | FP: {fp} | FN: {fn_gt}"
        )

        # Визуализация
        labels = [
            f"{COCO_CLASSES[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated_image = frameh.copy()

        # Убрана отрисовка полигона и наложение зоны

        if len(detections) > 0:
            annotated_image = bbox_annotator.annotate(annotated_image, detections)
            annotated_image = label_annotator.annotate(
                annotated_image, detections, labels
            )

        dt2 = time.perf_counter() - t2
        t_show_start = time.perf_counter()
        cv.imshow("Real-time Detection", annotated_image)
        key = cv.waitKey(1)
        dt3 = time.perf_counter() - t_show_start

        if create_out_file:
            video_out.write(annotated_image)

        tt = dt0 + dt1 + dt2 + dt3
        if fn % 10 == 0:
            print(
                f"{fn} | Pre: {dt0:.4f}s | Detect: {dt1:.4f}s | Annot: {dt2:.4f}s | Total: {tt:.4f}s"
            )

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = True

    # ================= ФИНАЛЬНЫЕ МЕТРИКИ =================
    cv.destroyAllWindows()
    video_capture.release()
    if create_out_file:
        video_out.release()

    # Расчёт агрегированных метрик
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    mean_iou = np.mean(all_iou_scores) if all_iou_scores else 0
    mean_dice = np.mean(all_dice_scores) if all_dice_scores else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Вывод в консоль
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ (полный кадр)")
    print(f"{'='*60}")
    print(f"Всего кадров обработано: {fn}")
    print(f"Всего предсказаний: {total_tp + total_fp}")
    print(f"Всего GT объектов: {total_tp + total_fn}")
    print(f"\nМетрики:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1_score:.4f}")
    print(f"  Mean IoU:  {mean_iou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"{'='*60}\n")

    # Сохранение в файл
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"RF-DETR Testing Report - Full Frame Evaluation\n")
        f.write(f"Video: {in_file}\n")
        f.write(f"Labels: {labels_dir}\n")
        f.write(f"Threshold: {threshold}, IoU match: {iou_threshold}\n")
        f.write(f"Frames processed: {fn}\n\n")
        f.write(f"AGGREGATED METRICS:\n")
        f.write(f"  Precision: {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"  F1-Score:  {f1_score:.4f}\n")
        f.write(f"  Mean IoU:  {mean_iou:.4f}\n")
        f.write(f"  Mean Dice: {mean_dice:.4f}\n\n")
        f.write(f"CONFUSION:\n")
        f.write(f"  True Positives:  {total_tp}\n")
        f.write(f"  False Positives: {total_fp}\n")
        f.write(f"  False Negatives: {total_fn}\n\n")
    print(f"Метрики сохранены в файл: {metrics_file}")


if __name__ == "__main__":
    main()
