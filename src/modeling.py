from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from . import config

# ResNet18 iskeletini ihtiyaç halinde farklı sınıf sayılarıyla yeniden kuruyoruz.


def create_resnet18(num_classes: int | None = None, pretrained: bool = True) -> nn.Module:
    weights = (
        models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model = models.resnet18(weights=weights)
    num_classes = num_classes or config.NUM_CLASSES
    in_features = model.fc.in_features
    dropout_p = getattr(config, "CLASSIFIER_DROPOUT", 0.0)
    if dropout_p and dropout_p > 0.0:
        model.fc = nn.Sequential(
            # Dropout küçük veri setlerinde aşırı öğrenmeyi frenlemeye yardımcı oluyor.
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module, train_bn: bool = True) -> nn.Module:
    # İnce ayar sırasında gövdeyi dondurup yalnızca son katmanları eğitebilmek için.
    for name, param in model.named_parameters():
        if name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    if train_bn:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = True
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
