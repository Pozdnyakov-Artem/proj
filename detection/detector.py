from rfdetr import RFDETRBase
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model_kwargs=dict()
        self.model_kwargs["pretrain_weights"] = r"rf-detr-finetuned.pth"
        self.model = RFDETRBase(**self.model_kwargs)
        self.model.optimize_for_inference()
        logger.info("Модель RFDETR загружена")

    def predict(self, frame_rgb, threshold: float = None):
        image_pil = Image.fromarray(frame_rgb)
        return self.model.predict(
            image_pil,
            threshold=threshold or self.threshold
        )

    def set_threshold(self, threshold: float):
        self.threshold = threshold