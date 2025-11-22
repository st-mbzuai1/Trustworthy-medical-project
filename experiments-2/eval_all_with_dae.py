#!/usr/bin/env python3
# eval_all_with_dae.py
import argparse, os, json
from typing import Dict, Any
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as T

HAM_LABELS = ['akiec','bcc','bkl','df','mel','nv','vasc']
LABEL2IDX = {c:i for i,c in enumerate(HAM_LABELS)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1,3,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,3,1,1)
    return (x - mean) / (std + 1e-8)

class HAMDatasetRaw(Dataset):
    def __init__(self, csv_path: str, split: str, img_size: int):
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = Image.open(row['image_path']).convert('RGB'); x = self.tf(x)
        y = LABEL2IDX[row['label']]; return x,y,row['image_path']

def build_classifier(arch: str, num_classes: int = 7) -> nn.Module:
    import torchvision.models as tv; import timm
    a = arch.lower()
    if a == 'resnet50':
        m = tv.resnet50(); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if a == 'densenet121':
        m = tv.densenet121(); m.classifier = nn.Linear(m.classifier.in_features, num_classes); return m
    if a == 'efficientnet_b0':
        m = tv.efficientnet_b0(); in_feat = m.classifier[1].in_features; m.classifier[1] = nn.Linear(in_feat, num_classes); return m
    if a in ('dino','dino_vitb16','vitb16_dino'):
        m = timm.create_model("vit_base_patch16_224.dino", pretrained=False, num_classes=num_classes); return m
    raise ValueError(arch)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base=32, groups=8):
        super().__init__()
        def C(in_c, out_c): return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(num_groups=min(groups, out_c), num_channels=out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(num_groups=min(groups, out_c), num_channels=out_c), nn.ReLU(inplace=True),
        )
        self.down1 = C(in_ch, base); self.pool = nn.MaxPool2d(2)
        self.down2 = C(base, base*2); self.down3 = C(base*2, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2); self.conv2 = C(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2); self.conv1 = C(base*2, base)
        self.outc = nn.Conv2d(base, 3, 1)
    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool(d1); d2 = self.down2(p1); p2 = self.pool(d2); d3 = self.down3(p2)
        u2 = self.up2(d3); c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2); c1 = self.conv1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.outc(c1))

def build_dae(base=32, groups=8) -> nn.Module:
    try:
        from models import UNetDenoiser
        return UNetDenoiser(in_ch=3, base=base, groups=groups)
    except Exception:
        return UNetSmall(in_ch=3, base=base, groups=groups)

def clamp01(x): return x.clamp(0,1)
def add_gaussian(x, sigma):      return clamp01(x + torch.randn_like(x)*sigma)
def add_speckle(x, sigma):       return clamp01(x + x*torch.randn_like(x)*sigma)
def add_saltpepper(x, p):
    noise = torch.empty_like(x[:, :1]).uniform_(0,1)
    salt = (noise < (p/2)).float(); pepper = (noise > 1 - (p/2)).float()
    x_sp = x.clone(); x_sp = x_sp * (1 - salt) + salt; x_sp = x_sp * (1 - pepper)
    return x_sp
def add_poisson(x, scale):
    x_clamped = (x.clamp(0,1)*scale); noisy = torch.poisson(x_clamped) / (scale + 1e-8)
    return clamp01(noisy)

SEVERITIES = {
    "gaussian":   {"sigma":  [0.05, 0.08]},
    "speckle":    {"sigma":  [0.05, 0.08]},
    "saltpepper": {"p":      [0.02, 0.03]},
    "poisson":    {"scale":  [20.0, 30.0]},
}

@torch.no_grad()
def top1_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    logits = model(x); pred = logits.argmax(1); return (pred == y).float().sum().item(), y.numel()

def load_classifier(arch: str, ckpt_path: str, device: torch.device) -> nn.Module:
    m = build_classifier(arch).to(device).eval()
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and any(k in sd for k in ('state_dict','model','net')):
        for k in ('state_dict','model','net'):
            if k in sd and isinstance(sd[k], dict): sd = sd[k]; break
    clean = {}
    for k,v in sd.items():
        kk = k
        for p in ('module.','backbone.','model.'):
            if kk.startswith(p): kk = kk[len(p):]
        clean[kk] = v
    m.load_state_dict(clean, strict=False); return m

def eval_all(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_ds = HAMDatasetRaw(args.csv, 'val', args.img_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    out_root = "outputs_new_code"
    dae_dir = os.path.join(out_root, "dae_unet"); dae_ckpt = os.path.join(dae_dir, "best.pt")
    dae = None
    if os.path.exists(dae_ckpt):
        dae = build_dae(base=args.dae_base, groups=args.dae_groups).to(device).eval()
        dae.load_state_dict(torch.load(dae_ckpt, map_location='cpu'), strict=False); print("[DAE] loaded", dae_ckpt)
    arches = ['resnet50','densenet121','efficientnet_b0']
    results: Dict[str, Any] = {}
    for arch in arches:
        ckpt = os.path.join(out_root, f"clean_{arch}", "best.pt")
        if not os.path.exists(ckpt): print(f"[{arch}] missing {ckpt}"); continue
        clf = load_classifier(arch, ckpt, device)
        ok = tot = ok_d = 0
        with torch.no_grad():
            for x,y,_ in tqdm(val_loader, desc=f"{arch} clean"):
                x,y = x.to(device), y.to(device)
                x_norm = normalize_imagenet(x); n_ok, n = top1_acc(clf, x_norm, y); ok += n_ok; tot += n
                if dae is not None:
                    x_d = torch.clamp(dae(x), 0, 1); x_d_norm = normalize_imagenet(x_d)
                    n_ok_d, _ = top1_acc(clf, x_d_norm, y); ok_d += n_ok_d
        acc_clean = ok / max(1, tot); acc_dae_on_clean = (ok_d / max(1, tot)) if dae is not None else None
        corr = {}
        for kind, grid in SEVERITIES.items():
            corr[kind] = {}
            for pname, vals in grid.items():
                for v in vals:
                    okn = okden = totc = 0
                    with torch.no_grad():
                        for x,y,_ in tqdm(val_loader, desc=f"{arch} {kind} {pname}={v}"):
                            x,y = x.to(device), y.to(device)
                            if kind == "gaussian":   x_noisy = clamp01(x + torch.randn_like(x)*float(v))
                            if kind == "speckle":    x_noisy = clamp01(x + x*torch.randn_like(x)*float(v))
                            if kind == "saltpepper":
                                noise = torch.empty_like(x[:, :1]).uniform_(0,1)
                                salt = (noise < (float(v)/2)).float(); pepper = (noise > 1 - (float(v)/2)).float()
                                x_noisy = x * (1 - salt) + salt; x_noisy = x_noisy * (1 - pepper)
                            if kind == "poisson":
                                scale = float(v); x_noisy = torch.poisson((x.clamp(0,1)*scale)) / (scale + 1e-8); x_noisy = clamp01(x_noisy)
                            xn = normalize_imagenet(x_noisy); n_ok, n = top1_acc(clf, xn, y); okn += n_ok; totc += n
                            if dae is not None:
                                xd = dae(x_noisy).clamp(0,1); xd_n = normalize_imagenet(xd)
                                n_ok_d, _ = top1_acc(clf, xd_n, y); okden += n_ok_d
                    corr[kind][f"{pname}={v}"] = {"acc_noisy": okn/max(1,totc), "acc_denoised": (okden/max(1,totc)) if dae is not None else None}
        results[arch] = {"clean_acc": acc_clean, "dae_on_clean_acc": acc_dae_on_clean, "corruptions": corr}
        print(f"[{arch}] clean={acc_clean:.4f}  dae(clean)={acc_dae_on_clean if acc_dae_on_clean is not None else 'N/A'}")
    os.makedirs(out_root, exist_ok=True); outp = os.path.join(out_root, "eval_summary.json")
    with open(outp, "w") as f: json.dump(results, f, indent=2); print("[done] wrote", outp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32); ap.add_argument('--dae_base', type=int, default=32); ap.add_argument('--dae_groups', type=int, default=8)
    args = ap.parse_args(); eval_all(args)

if __name__ == "__main__": main()
