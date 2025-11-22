#!/usr/bin/env python3
# train_all_and_dae.py
import argparse, os, json, math, random
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
import torchvision.transforms as T

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

HAM_LABELS = ['akiec','bcc','bkl','df','mel','nv','vasc']
LABEL2IDX = {c:i for i,c in enumerate(HAM_LABELS)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1,3,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,3,1,1)
    return (x - mean) / (std + 1e-8)

class HAMDatasetRaw(Dataset):
    def __init__(self, csv_path: str, split: str, img_size: int, augment: bool):
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        if augment and split == 'train':
            self.tf = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2,0.2,0.2,0.02),
                T.ToTensor(),
            ])
        else:
            self.tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = Image.open(row['image_path']).convert('RGB')
        x = self.tf(x)
        y = LABEL2IDX[row['label']]
        return x, y, row['image_path']

def make_loaders(csv: str, img_size: int, batch: int) -> Tuple[DataLoader, DataLoader, Dict[int,float], np.ndarray]:
    train_ds = HAMDatasetRaw(csv, 'train', img_size, augment=True)
    val_ds   = HAMDatasetRaw(csv, 'val',   img_size, augment=False)
    counts = pd.Series([LABEL2IDX[l] for l in pd.read_csv(csv).query("split=='train'")['label']]).value_counts().to_dict()
    class_counts = np.array([counts.get(i, 1) for i in range(len(HAM_LABELS))], dtype=np.float32)
    inv_freq = class_counts.sum() / (class_counts + 1e-6)
    labels_train = [y for _,y,_ in train_ds]
    sample_w = [float(inv_freq[y]) for y in labels_train]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, {i: float(inv_freq[i]) for i in range(len(HAM_LABELS))}, class_counts

def build_classifier(arch: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    import torchvision.models as tv; import timm
    a = arch.lower()
    if a == 'resnet50':
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if a == 'densenet121':
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes); return m
    if a == 'efficientnet_b0':
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = m.classifier[1].in_features; m.classifier[1] = nn.Linear(in_feat, num_classes); return m
    if a in ('dino','dino_vitb16','vitb16_dino'):
        m = timm.create_model("vit_base_patch16_224.dino", pretrained=pretrained, num_classes=num_classes); return m
    raise ValueError(f"Unknown arch: {arch}")

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base=32, groups=8):
        super().__init__()
        def C(in_c, out_c): return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(num_groups=min(groups, out_c), num_channels=out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(num_groups=min(groups, out_c), num_channels=out_c), nn.ReLU(inplace=True),
        )
        self.down1 = C(in_ch, base); self.pool = nn.MaxPool2d(2)
        self.down2 = C(base, base*2)
        self.down3 = C(base*2, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = C(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = C(base*2, base)
        self.outc = nn.Conv2d(base, 3, 1)
    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool(d1)
        d2 = self.down2(p1); p2 = self.pool(d2)
        d3 = self.down3(p2)
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

def cosine_with_warmup_lambda(total_epochs: int, warmup_frac: float = 0.1):
    warmup_epochs = max(1, int(math.ceil(warmup_frac * total_epochs)))
    def lr_lambda(epoch):
        if epoch < warmup_epochs: return float(epoch + 1) / float(warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return lr_lambda

@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); ok = tot = 0
    for x,y,_ in loader:
        x,y = x.to(device), y.to(device)
        x_norm = normalize_imagenet(x)
        logits = model(x_norm); pred = logits.argmax(1)
        ok += (pred == y).sum().item(); tot += y.numel()
    return ok / max(1, tot)

def train_one_classifier(arch: str, train_loader: DataLoader, val_loader: DataLoader,
                         out_dir: str, epochs: int, device: torch.device, lr: float = 3e-4):
    model = build_classifier(arch, num_classes=len(HAM_LABELS), pretrained=True).to(device)
    ema   = AveragedModel(model).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup_lambda(epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    best_acc = -1.0; os.makedirs(out_dir, exist_ok=True)
    meta = {"arch": arch, "num_classes": len(HAM_LABELS), "normalize": "imagenet"}
    with open(os.path.join(out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
    for epoch in range(epochs):
        model.train(); pbar = tqdm(train_loader, desc=f"{arch} epoch {epoch+1}/{epochs}")
        for x,y,_ in pbar:
            x,y = x.to(device), y.to(device); x_norm = normalize_imagenet(x)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(x_norm); loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            scaler.scale(loss).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); ema.update_parameters(model)
            pbar.set_postfix(loss=float(loss.item()))
        sched.step()
        acc = eval_accuracy(ema, val_loader, device)
        print(f"[val] {arch}: epoch {epoch+1}/{epochs} acc={acc:.4f}")
        with open(os.path.join(out_dir, "val_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "acc": float(acc), "lr": sched.get_last_lr()[0]}))
        if acc > best_acc:
            best_acc = acc; torch.save(ema.state_dict(), os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "best.json"), "w") as f: json.dump({"epoch": epoch, "acc": float(best_acc)}, f, indent=2)
    return best_acc

def train_dae(train_loader: DataLoader, val_loader: DataLoader, out_dir: str,
              epochs: int, base: int, groups: int, device: torch.device, lr: float = 2e-4):
    dae = build_dae(base=base, groups=groups).to(device)
    opt = optim.Adam(dae.parameters(), lr=lr)
    sched = optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup_lambda(epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda')); os.makedirs(out_dir, exist_ok=True)
    def noisy_batch(x):
        bs = x.size(0); kinds = torch.randint(0, 4, (bs,), device=x.device); out = []
        for i in range(bs):
            xi = x[i:i+1]; k = int(kinds[i].item())
            if k == 0: yi = add_gaussian(xi, sigma=0.08)
            elif k == 1: yi = add_speckle(xi, sigma=0.08)
            elif k == 2: yi = add_saltpepper(xi, p=0.02)
            else: yi = add_poisson(xi, scale=30.0)
            out.append(yi)
        return torch.cat(out, dim=0)
    best_l1 = float('inf')
    for epoch in range(epochs):
        dae.train(); pbar = tqdm(train_loader, desc=f"DAE epoch {epoch+1}/{epochs}")
        for x,_,_ in pbar:
            x = x.to(device); x_noisy = noisy_batch(x)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                recon = dae(x_noisy); loss_l1 = F.l1_loss(recon, x)
            scaler.scale(loss_l1).backward(); torch.nn.utils.clip_grad_norm_(dae.parameters(), 1.0)
            scaler.step(opt); scaler.update(); pbar.set_postfix(l1=float(loss_l1.item()))
        sched.step()
        dae.eval(); l1_sum = n = 0
        with torch.no_grad():
            for x,_,_ in val_loader:
                x = x.to(device); x_noisy = noisy_batch(x)
                recon = dae(x_noisy); l1 = F.l1_loss(recon, x, reduction='sum')
                l1_sum += l1.item(); n += x.numel()
        l1_epoch = l1_sum / max(1, n)
        print(f"[val] DAE: epoch {epoch+1}/{epochs} L1={l1_epoch:.6f}")
        with open(os.path.join(out_dir, "val_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "l1": float(l1_epoch), "lr": sched.get_last_lr()[0]}))
        if l1_epoch < best_l1:
            best_l1 = l1_epoch; torch.save(dae.state_dict(), os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "best.json"), "w") as f: json.dump({"epoch": epoch, "l1": float(best_l1)}, f, indent=2)
    with open(os.path.join(out_dir, "stats.json"), "w") as f: json.dump({"mean": [0.0,0.0,0.0], "std": [1.0,1.0,1.0]}, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--dae_epochs', type=int, default=20)
    ap.add_argument('--dae_base', type=int, default=32); ap.add_argument('--dae_groups', type=int, default=8)
    args = ap.parse_args()
    set_seed(1337); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.backends.cudnn.benchmark = True
    train_loader, val_loader, _, _ = make_loaders(args.csv, args.img_size, args.batch_size)
    arches = ['resnet50','densenet121','efficientnet_b0']; out_root = "outputs_new_code"; results = {}
    for arch in arches:
        out_dir = os.path.join(out_root, f"clean_{arch}")
        print(f"\n=== Training {arch} ==="); acc = train_one_classifier(arch, train_loader, val_loader, out_dir, args.epochs, device); results[arch] = {"best_acc": float(acc)}
    dae_dir = os.path.join(out_root, "dae_unet"); print(f"\n=== Training DAE ==="); train_dae(train_loader, val_loader, dae_dir, args.dae_epochs, args.dae_base, args.dae_groups, device)
    with open(os.path.join(out_root, "train_summary.json"), "w") as f: json.dump(results, f, indent=2)
    print("\n[done] Training complete. Outputs in:", out_root)

if __name__ == "__main__": main()
