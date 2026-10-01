"""Model zoo. Every model here is a CNN (no ViTs / CLIP / DINO, per the assignment rules)."""
import torch.nn as nn
from torchvision import models as tvm


class TNet(nn.Module):
    """The deliberately weak starter baseline from the notebook (grayscale, 64x64)."""

    def __init__(self, num_classes=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(inplace=True), nn.MaxPool2d(4, 4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 15 * 15, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SceneCNN(nn.Module):
    """A VGG-style CNN with BatchNorm and global average pooling, trained from scratch.

    Four stages of two 3x3 convs, each followed by 2x2 max-pooling. GAP + dropout
    keeps the head tiny so the ~1.2M parameters are spent on convolutional features.
    """

    def __init__(self, num_classes=16, in_ch=3, width=32, dropout=0.3):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        layers, cin = [], in_ch
        for cout in w:
            layers += [conv_bn(cin, cout), conv_bn(cout, cout), nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


# torchvision CNNs. Pretrained weights are ImageNet-1k (see README for exact weight tags).
_TV = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def _replace_head(model, name, num_classes, dropout):
    if name.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name.startswith("mobilenet") or name.startswith("efficientnet"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    elif name.startswith("convnext"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(last.in_features, num_classes))
    return model


def head_parameters(model, name):
    if name.startswith("resnet"):
        return list(model.fc.parameters())
    return list(model.classifier[-1].parameters())


def build_model(cfg, num_classes):
    name = cfg["name"]
    if name == "tnet":
        return TNet(num_classes)
    if name == "scenecnn":
        return SceneCNN(num_classes, in_ch=cfg.get("in_ch", 3), width=cfg.get("width", 32),
                        dropout=cfg.get("dropout", 0.3))
    fn, weights = _TV[name]
    model = fn(weights=weights if cfg.get("pretrained", False) else None)
    model = _replace_head(model, name, num_classes, cfg.get("dropout", 0.0))
    if cfg.get("freeze_backbone", False):
        head = set(id(p) for p in head_parameters(model, name))
        for p in model.parameters():
            p.requires_grad = id(p) in head
    return model
