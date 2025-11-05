from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from matplotlib import cm

from . import config, data, modeling, utils
from .gradcam import GradCAM


@dataclass
class PredictionResult:
    label: str
    probabilities: Dict[str, float]
    heatmap: Optional[Image.Image]
    overlay: Optional[Image.Image]


class AutismDetector:
    """Wraps model loading, preprocessing, prediction, and Grad-CAM generation."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: Optional[torch.device] = None,
    ) -> None:
        checkpoint_path = (
            config.DEFAULT_CHECKPOINT_PATH if checkpoint_path is None else checkpoint_path
        )
        self.device = device or utils.get_device()
        self.model = modeling.create_resnet18()

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = utils.load_checkpoint(Path(checkpoint_path), map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. Train the model first."
            )

        self.model.eval()
        self.model.to(self.device)

        # Eğitimde kullanılan dönüşümlerin birebir aynısını tutarak dağılımı koruyoruz.
        self.transform = data.default_transforms(train=False)
        # En son konvolüsyon bloğundan Grad-CAM çıkarmak yüz bölgelerinin vurgulanmasını sağlıyor.
        self.grad_cam = GradCAM(self.model, self.model.layer4[-1])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def _heatmap_to_image(
        self,
        heatmap: np.ndarray,
        base_image: Image.Image,
        alpha: float = 0.5,
    ) -> Image.Image:
        base_image = base_image.convert("RGB")
        heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
            base_image.size, resample=Image.BILINEAR
        )
        heatmap_np = np.array(heatmap_img) / 255.0
        colormap = cm.get_cmap("jet")
        colored_heatmap = colormap(heatmap_np)[..., :3]

        base_np = np.array(base_image) / 255.0
        overlay = (1 - alpha) * base_np + alpha * colored_heatmap
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray(np.uint8(overlay * 255))
        return overlay_img

    def predict(
        self,
        image: Image.Image,
        generate_heatmap: bool = True,
    ) -> PredictionResult:
        tensor = self._preprocess(image)
        heatmap_img = None
        overlay_img = None

        if generate_heatmap:
            tensor.requires_grad_(True)
            cam_output = self.grad_cam.generate(tensor)
            logits = cam_output.logits
            probs = cam_output.probs
            heatmap = cam_output.heatmap
            heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
            overlay_img = self._heatmap_to_image(heatmap, image)
        else:
            with torch.inference_mode():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)

        probs = probs.squeeze().cpu().numpy()
        label_idx = int(np.argmax(probs))
        probabilities = {
            class_name: float(probs[idx]) for idx, class_name in enumerate(config.CLASS_NAMES)
        }
        predicted_label = config.CLASS_NAMES[label_idx]

        # Sonuç objesi arayüz ve raporlama katmanlarına tek bir yerden veri taşıyor.
        return PredictionResult(
            label=predicted_label,
            probabilities=probabilities,
            heatmap=heatmap_img,
            overlay=overlay_img,
        )
