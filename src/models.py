import torch
import torch.nn as nn
from torchvision import models

def build_model(arch: str, num_classes: int):
    """
    Returns (model, input_size). Supports resnet50, efficientnet_b3, inception_v3.
    """
    a = arch.lower()
    if a == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.BatchNorm1d(in_f),
            nn.Dropout(0.5),
            nn.Linear(in_f, num_classes)
        )
        return m, 224

    elif a == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        # find last linear in classifier
        in_f = None
        for mod in m.classifier:
            if isinstance(mod, nn.Linear):
                in_f = mod.in_features
        if in_f is None:
            in_f = 1536
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m, 300

    elif a == "inception_v3":
        # aux_logits must be False for standard training loop
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.BatchNorm1d(in_f),
            nn.Dropout(0.5),
            nn.Linear(in_f, num_classes)
        )
        return m, 299

    else:
        raise ValueError(f"Unsupported arch: {arch}")
