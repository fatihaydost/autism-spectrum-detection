from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GradCAMOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    heatmap: np.ndarray


class GradCAM:
    """Minimal Grad-CAM implementation for convolutional backbones."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inputs, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # register_full_backward_hook to capture gradients w.r.t. activations.
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> GradCAMOutput:
        if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
            raise ValueError("GradCAM expects a single image tensor with shape (1, C, H, W)")

        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        self.model.zero_grad(set_to_none=True)

        target = logits[:, class_idx]
        target.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        gradients = self._gradients
        activations = self._activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return GradCAMOutput(logits=logits.detach(), probs=probs.detach(), heatmap=cam)
