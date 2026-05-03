import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rfdetr import RFDETRBase
from PIL import Image
import numpy as np

video_dir = "/videos"
runs_dir = "/runs"


class VideoDataset(Dataset):
    def __init__(self, video_paths, annotation_files, transform=None):
        self.video_paths = video_paths
        self.annotation_files = annotation_files
        self.transform = transform
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for video_path, ann_files in zip(self.video_paths, self.annotation_files):
            cap = cv2.VideoCapture(video_path)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            for frame_num, txt_path in ann_files:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if not ret:
                    continue
                bboxes = []
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            x = float(parts[1]) * w
                            y = float(parts[2]) * h
                            ww = float(parts[3]) * w
                            hh = float(parts[4]) * h
                            conf = float(parts[5])
                            x1 = x - ww / 2
                            y1 = y - hh / 2
                            x2 = x + ww / 2
                            y2 = y + hh / 2
                            bboxes.append([x1, y1, x2, y2, class_id])
                if bboxes:
                    self.samples.append((frame, bboxes))
            cap.release()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, bboxes = self.samples[idx]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        return image, bboxes


def find_data(video_dir, runs_dir):
    video_paths = []
    ann_files_per_video = []
    for file in os.listdir(video_dir):
        if file.endswith((".mp4", ".avi", ".mkv")):
            video_name = os.path.splitext(file)[0]
            video_path = os.path.join(video_dir, file)
            video_paths.append(video_path)
            ann_files = []
            for ann_file in os.listdir(runs_dir):
                if ann_file.startswith(video_name + "_") and ann_file.endswith(".txt"):
                    parts = ann_file.split("_")
                    if len(parts) == 2:
                        try:
                            frame_num = int(parts[1].replace(".txt", ""))
                            ann_files.append(
                                (frame_num, os.path.join(runs_dir, ann_file))
                            )
                        except ValueError:
                            pass
            ann_files_per_video.append(ann_files)
    return video_paths, ann_files_per_video


if __name__ == "__main__":
    video_paths, ann_files_per_video = find_data(video_dir, runs_dir)
    if not video_paths:
        print("Видео не найдены")
        exit()

    transform = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )

    dataset = VideoDataset(video_paths, ann_files_per_video, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = RFDETRBase()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        for images, bboxes in dataloader:
            optimizer.zero_grad()
            # Предполагаем, что модель имеет метод forward для обучения
            # outputs = model(images)
            # loss = compute_loss(outputs, bboxes)  # Нужно определить loss
            # loss.backward()
            # optimizer.step()
            print(f"Epoch {epoch}, processing batch")
        print(f"Epoch {epoch} completed")

    torch.save(model.state_dict(), "rf-detr-finetuned.pth")
    print("Модель сохранена")
