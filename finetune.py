import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rfdetr import RFDETRBase
from PIL import Image

VIDEO_DIR = "/home/julia/Desktop/code/detect-2"
RUNS_DIR = "/home/julia/Desktop/code/detect-2/ann"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (640, 640)


class VideoDataset(Dataset):
    def __init__(self, video_paths, annotation_files, transform=None):
        self.video_paths = video_paths
        self.annotation_files = annotation_files
        self.transform = transform
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for video_path, ann_files in zip(self.video_paths, self.annotation_files):
            if not ann_files:
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARN] Не удалось открыть видео: {video_path}")
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for frame_num, txt_path in sorted(ann_files, key=lambda x: x[0]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if not ret:
                    print(
                        f"[WARN] Кадр {frame_num} пропущен в {os.path.basename(video_path)}"
                    )
                    continue

                boxes, labels = [], []
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])

                            x1 = (xc - bw / 2) * w
                            y1 = (yc - bh / 2) * h
                            x2 = (xc + bw / 2) * w
                            y2 = (yc + bh / 2) * h

                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls_id)
                        except ValueError:
                            continue

                if boxes:
                    self.samples.append(
                        (
                            frame,
                            torch.tensor(boxes, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long),
                        )
                    )
            cap.release()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, boxes, labels = self.samples[idx]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        orig_w, orig_h = image.size

        if self.transform:
            image = self.transform(image)

        target_h, target_w = image.shape[1], image.shape[2]

        boxes = boxes.clone()
        boxes[:, 0::2] *= target_w / orig_w
        boxes[:, 1::2] *= target_h / orig_h

        boxes[:, 0::2] /= target_w
        boxes[:, 1::2] /= target_h
        boxes = torch.clamp(boxes, min=0.0, max=1.0)

        return image, {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor([target_h, target_w], dtype=torch.long),
        }


def find_data(video_dir, runs_dir):
    video_paths, ann_files_per_video = [], []
    video_map = {}

    if os.path.isdir(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith((".mp4", ".avi", ".mkv")):
                video_map[os.path.splitext(f)[0]] = os.path.join(video_dir, f)

    for track in os.listdir(runs_dir):
        track_path = os.path.join(runs_dir, track)
        labels_dir = os.path.join(track_path, "labels")
        if not os.path.isdir(labels_dir):
            continue

        current_anns = {}
        for ann_file in os.listdir(labels_dir):
            if not ann_file.endswith(".txt"):
                continue
            stem = os.path.splitext(ann_file)[0]
            last_underscore = stem.rfind("_")
            if last_underscore == -1:
                continue
            vid_name = stem[:last_underscore]
            try:
                frame_num = int(stem[last_underscore + 1 :])
            except ValueError:
                continue
            current_anns.setdefault(vid_name, []).append(
                (frame_num, os.path.join(labels_dir, ann_file))
            )

        for vid_name, anns in current_anns.items():
            vid_path = video_map.get(vid_name)
            if vid_path is None:
                alt_path = os.path.join(track_path, f"{vid_name}.mp4")
                if os.path.exists(alt_path):
                    vid_path = alt_path
            if vid_path:
                video_paths.append(vid_path)
                ann_files_per_video.append(anns)
                print(f"[OK] {vid_name}: загружено {len(anns)} кадров")
            else:
                print(f"[WARN] Видео '{vid_name}.mp4' не найдено. Аннотации пропущены.")

    return video_paths, ann_files_per_video


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def extract_nn_module(model):
    module = model
    visited = set()
    while not isinstance(module, torch.nn.Module) or type(module).__name__ in [
        "RFDETRBase",
        "ModelContext",
    ]:
        if id(module) in visited:
            break
        visited.add(id(module))
        next_module = (
            getattr(module, "model", None)
            or getattr(module, "net", None)
            or getattr(module, "module", None)
        )
        if next_module is None:
            break
        module = next_module
    return module if isinstance(module, torch.nn.Module) else None


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print("Поиск данных...")
    video_paths, ann_files_per_video = find_data(VIDEO_DIR, RUNS_DIR)

    if not video_paths:
        print("[ERROR] Видеофайлы или аннотации не найдены.")
        exit()

    TARGET_SIZE = (672, 672)
    transform = transforms.Compose(
        [
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
        ]
    )

    dataset = VideoDataset(video_paths, ann_files_per_video, transform=transform)
    print(f"[INFO] Датасет сформирован: {len(dataset)} семплов")

    if len(dataset) == 0:
        print("[ERROR] Датасет пуст.")
        exit()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    print("Инициализация модели...")
    model = RFDETRBase(device=str(DEVICE))

    backbone = extract_nn_module(model)
    if backbone is None:
        raise RuntimeError("Не удалось извлечь nn.Module из RFDETRBase")

    backbone.to(DEVICE)

    optimizer = optim.AdamW(backbone.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 10
    print(f"Начало обучения на устройстве: {DEVICE}")

    for epoch in range(num_epochs):
        backbone.train()
        epoch_loss = 0.0

        for i, (images, targets) in enumerate(dataloader):
            images_tensor = torch.stack(images).to(DEVICE, non_blocking=True)
            targets = [
                {k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad()

            try:
                outputs = backbone(images_tensor, targets)
            except TypeError:
                outputs = backbone(images_tensor, targets=targets)

            if isinstance(outputs, dict):
                loss = outputs.get("loss")
                if loss is None:
                    scalars = [
                        v
                        for v in outputs.values()
                        if isinstance(v, torch.Tensor)
                        and v.dim() == 0
                        and v.requires_grad
                    ]
                    if scalars:
                        loss = sum(scalars)
                    else:
                        print(
                            f"[WARN] Скалярный loss не найден. Доступные ключи: {list(outputs.keys())}"
                        )
                        continue
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                print(f"[WARN] Неожиданный формат outputs: {type(outputs)}")
                continue

            if loss is None:
                print("[WARN] Loss is None — проверьте формат targets")
                continue

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"[INFO] Epoch {epoch+1} завершён. Средний Loss: {avg_loss:.4f}")

    torch.save(backbone.state_dict(), "rf-detr-finetuned.pth")
    print("Обучение завершено. Модель сохранена в rf-detr-finetuned.pth")
