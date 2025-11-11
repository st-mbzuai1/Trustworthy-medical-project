import torch, torch.nn as nn
import torchvision.models as tv
import timm

# ---------------------- Classifiers (unchanged API) ----------------------

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

# ---------------------- DAE building blocks (upgraded) --------------------

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)

class GNAct(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.gn(x))

class DoubleConvGN(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.c1 = conv3x3(in_ch, out_ch)
        self.n1 = GNAct(out_ch, groups)
        self.c2 = conv3x3(out_ch, out_ch)
        self.n2 = GNAct(out_ch, groups)
        # residual within block
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        identity = x
        x = self.n1(self.c1(x))
        x = self.n2(self.c2(x))
        if self.proj is not None:
            identity = self.proj(identity)
        return x + identity

class BottleneckSelfAttn(nn.Module):
    """Lightweight spatial self-attention (non-local-lite) at the bottleneck."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = (ch // heads) ** -0.5
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, self.heads, C//self.heads, H*W)      # B,h,c,HW
        k = self.k(x).view(B, self.heads, C//self.heads, H*W)
        v = self.v(x).view(B, self.heads, C//self.heads, H*W)
        attn = torch.softmax((q.transpose(-2,-1) @ k) * self.scale, dim=-1)  # B,h,HW,HW
        out = (attn @ v.transpose(-2,-1)).transpose(-2,-1)                  # B,h,c,HW
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)

# ---------------------- Upgraded UNet DAE (same class name) ----------------

class UNetDenoiser(nn.Module):
    """
    Input: normalized or unnormalized image depending on training script.
    Output: reconstructed image in pixel space [0,1] (sigmoid head).
    Architecture:
      - UNet with GroupNorm + residual blocks
      - Self-attention at bottleneck
    """
    def __init__(self, in_ch=3, base=32, groups=8):
        super().__init__()
        b = base
        self.enc1 = DoubleConvGN(in_ch,   b,   groups)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvGN(b,       b*2, groups)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvGN(b*2,     b*4, groups)
        self.pool3 = nn.MaxPool2d(2)

        self.bott = DoubleConvGN(b*4, b*8, groups)
        self.attn = BottleneckSelfAttn(b*8, heads=4)

        self.up3  = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = DoubleConvGN(b*8, b*4, groups)
        self.up2  = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = DoubleConvGN(b*4, b*2, groups)
        self.up1  = nn.ConvTranspose2d(b*2, b,   2, stride=2)
        self.dec1 = DoubleConvGN(b*2, b,   groups)

        self.head = nn.Conv2d(b, in_ch, 1)   # will be passed through sigmoid

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool1(c1))
        c3 = self.enc3(self.pool2(c2))
        b  = self.attn(self.bott(self.pool3(c3)))
        u3 = self.dec3(torch.cat([self.up3(b),  c3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), c1], dim=1))
        return torch.sigmoid(self.head(u1))   # pixel-space [0,1]


def build_classifier(arch, num_classes=7, pretrained=True):
    a = arch.lower()
    if a=='resnet50': return ResNet50Classifier(num_classes, pretrained)
    if a=='densenet121': return DenseNet121Classifier(num_classes, pretrained)
    if a=='efficientnet_b0': return EfficientNetB0Classifier(num_classes, pretrained)
    if a in ('dino','dino_vitb16','vitb16_dino'): return DinoViTB16Classifier(num_classes, pretrained)
    raise ValueError(f"Unknown arch: {arch}")
