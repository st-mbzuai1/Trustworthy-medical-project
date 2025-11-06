
import torch, torch.nn as nn
import torchvision.models as tv
import timm

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.backbone = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, num_classes)
    def forward(self, x): return self.backbone(x)

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.backbone = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_feat, num_classes)
    def forward(self, x): return self.backbone(x)

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.backbone = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_feat, num_classes)
    def forward(self, x): return self.backbone(x)

class DinoViTB16Classifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224.dino", pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class UNetDenoiser(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.u3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.u2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.u1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, in_ch, 1)
    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.p1(c1))
        c3 = self.d3(self.p2(c2))
        c4 = self.d4(self.p3(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return torch.clamp(self.out(u1), 0, 1)

def build_classifier(arch, num_classes=7, pretrained=True):
    a = arch.lower()
    if a=='resnet50': return ResNet50Classifier(num_classes, pretrained)
    if a=='densenet121': return DenseNet121Classifier(num_classes, pretrained)
    if a=='efficientnet_b0': return EfficientNetB0Classifier(num_classes, pretrained)
    if a in ('dino','dino_vitb16','vitb16_dino'): return DinoViTB16Classifier(num_classes, pretrained)
    raise ValueError(f"Unknown arch: {arch}")
