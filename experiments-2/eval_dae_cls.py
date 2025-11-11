#!/usr/bin/env python3
import argparse, os, io, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageFilter
import torchvision.transforms as T

# --- try to import your DAE from models.py ---
try:
    from models import UNetDenoiser  # your upgraded U-Net
except Exception as e:
    raise ImportError(f"Could not import UNetDenoiser from models.py: {e}")

# --- local fallback build_classifier (so we don't depend on models.build_classifier) ---
def build_classifier(arch: str, num_classes: int = 7, pretrained: bool = False):
    import torchvision.models as tv
    import timm
    a = arch.lower()
    if a == "resnet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        return m
    if a == "densenet121":
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = m.classifier.in_features
        m.classifier = nn.Linear(in_feat, num_classes)
        return m
    if a == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
        return m
    if a in ("dino","dino_vitb16","vitb16_dino"):
        m = timm.create_model("vit_base_patch16_224.dino", pretrained=pretrained, num_classes=0)
        head = nn.Linear(m.num_features, num_classes)
        return nn.Sequential(m, head)
    raise ValueError(f"Unknown arch: {arch}")

# -------------------- IO helpers --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_stats_from_dir(dae_ckpt_path, device):
    statspath = os.path.join(os.path.dirname(dae_ckpt_path), "stats.json")
    if not os.path.exists(statspath):
        mean = torch.zeros(3, device=device).tolist()
        std  = torch.ones(3, device=device).tolist()
        return mean, std, False
    d = json.load(open(statspath, "r"))
    return d["mean"], d["std"], True

def norm_tensor(x, mean, std):
    mean = x.new_tensor(mean).view(1,3,1,1)
    std  = x.new_tensor(std).view(1,3,1,1)
    return (x - mean) / (std + 1e-8)

# -------------------- Corruptions (same as eval_dae.py) --------------------
def clamp01(x): return x.clamp(0,1)

def add_gaussian(x, sigma):      return clamp01(x + torch.randn_like(x)*sigma)
def add_speckle(x, sigma):       return clamp01(x + x*torch.randn_like(x)*sigma)

def add_saltpepper(x, p):
    noise = torch.empty_like(x[:, :1]).uniform_(0,1)
    salt = (noise < (p/2)).float()
    pepper = (noise > 1 - (p/2)).float()
    x_sp = x.clone()
    x_sp = x_sp * (1 - salt) + salt   # salt->1
    x_sp = x_sp * (1 - pepper)        # pepper->0
    return x_sp

def add_poisson(x, scale):
    x_clamped = (x.clamp(0,1)*scale)
    noisy = torch.poisson(x_clamped) / (scale + 1e-8)
    return clamp01(noisy)

def add_jpeg(x, quality):
    import torchvision.utils as vutils  # for ToTensor
    B = x.size(0)
    out = []
    for i in range(B):
        xi = (x[i].detach().cpu().clamp(0,1)*255).byte().permute(1,2,0).numpy()
        img = Image.fromarray(xi, mode='RGB')
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=int(quality)); buf.seek(0)
        out_img = Image.open(buf).convert('RGB')
        out.append(T.ToTensor()(out_img))
    return torch.stack(out, dim=0).to(x.device)

def add_blur(x, sigma):
    B = x.size(0)
    out = []
    for i in range(B):
        xi = (x[i].detach().cpu().clamp(0,1)*255).byte().permute(1,2,0).numpy()
        img = Image.fromarray(xi, mode='RGB').filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        out.append(T.ToTensor()(img))
    return torch.stack(out, dim=0).to(x.device)

SEVERITIES = {
    "gaussian":   {"sigma":  [0.03, 0.05, 0.08, 0.10]},
    "speckle":    {"sigma":  [0.03, 0.05, 0.08, 0.10]},
    "saltpepper": {"p":      [0.01, 0.02, 0.03, 0.04]},
    "poisson":    {"scale":  [20.0, 30.0, 40.0]},
    "jpeg":       {"quality":[80, 60, 40]},
    "blur":       {"sigma":  [0.8, 1.2, 2.0]},
}

def apply_corruption(x, kind, param_name, param_value):
    if kind == "gaussian":   return add_gaussian(x, sigma=float(param_value))
    if kind == "speckle":    return add_speckle(x, sigma=float(param_value))
    if kind == "saltpepper": return add_saltpepper(x, p=float(param_value))
    if kind == "poisson":    return add_poisson(x, scale=float(param_value))
    if kind == "jpeg":       return add_jpeg(x, quality=int(param_value))
    if kind == "blur":       return add_blur(x, sigma=float(param_value))
    raise ValueError(kind)

# -------------------- Accuracy helpers --------------------
@torch.no_grad()
def top1_acc(model, x, y):
    logits = model(x)
    preds = logits.argmax(dim=1)
    return (preds == y).float().sum().item(), y.numel()

# -------------------- Main eval --------------------
def eval_cls(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    from data_utils import make_loader  # late import to avoid circulars
    val_loader = make_loader(args.csv, 'val', args.img_size, args.batch_size, shuffle=False)

    # classifier
    clf = build_classifier(args.arch, num_classes=7, pretrained=False).to(device).eval()
    clf_sd = torch.load(args.ckpt, map_location='cpu')
    if isinstance(clf_sd, dict) and any(k in clf_sd for k in ('state_dict','model','net')):
        for k in ('state_dict','model','net'):
            if k in clf_sd and isinstance(clf_sd[k], dict):
                clf_sd = clf_sd[k]; break
    clean_sd = {}
    for k,v in clf_sd.items():
        kk = k
        for p in ('module.','backbone.','model.'):
            if kk.startswith(p): kk = kk[len(p):]
        clean_sd[kk] = v
    clf.load_state_dict(clean_sd, strict=False)

    # DAE (optional)
    dae = None
    if args.dae_ckpt and args.dae_ckpt.lower() != 'none':
        dae = UNetDenoiser(in_ch=3, base=args.base, groups=args.groups).to(device).eval()
        dae_sd = torch.load(args.dae_ckpt, map_location='cpu')
        if isinstance(dae_sd, dict) and any(k in dae_sd for k in ('state_dict','model','net')):
            for k in ('state_dict','model','net'):
                if k in dae_sd and isinstance(dae_sd[k], dict):
                    dae_sd = dae_sd[k]; break
        dclean = {}
        for k,v in dae_sd.items():
            kk = k
            for p in ('module.','model.','net.'):
                if kk.startswith(p): kk = kk[len(p):]
            dclean[kk] = v
        dae.load_state_dict(dclean, strict=False)

        mean, std, _ = load_stats_from_dir(args.dae_ckpt, device)
        mean_t = torch.tensor(mean, device=device).view(1,3,1,1)
        std_t  = torch.tensor(std,  device=device).view(1,3,1,1)
        def forward_dae(z):
            z_n = (z - mean_t) / (std_t + 1e-8)
            return torch.clamp(dae(z_n), 0, 1)
    else:
        def forward_dae(z): return z  # identity if no dae

    results = {"clean": {}, "corruptions": {}}

    # -------- clean accuracy --------
    num_ok_clean = tot = 0
    num_ok_deno  = 0
    with torch.no_grad():
        for x,y,_ in tqdm(val_loader, desc="CLS clean"):
            x,y = x.to(device), y.to(device)
            ok, n = top1_acc(clf, x, y); num_ok_clean += ok; tot += n
            x_d = forward_dae(x)
            okd, _ = top1_acc(clf, x_d, y); num_ok_deno += okd
    results["clean"]["acc_clean"]    = float(num_ok_clean / max(1, tot))
    results["clean"]["acc_denoised"] = float(num_ok_deno  / max(1, tot))

    # -------- corruption sweeps --------
    for kind, grid in SEVERITIES.items():
        results["corruptions"][kind] = {}
        for param_name, values in grid.items():
            for val in values:
                num_ok_noisy = num_ok_deno = tot = 0
                with torch.no_grad():
                    for x,y,_ in tqdm(val_loader, desc=f"CLS {kind} {param_name}={val}"):
                        x,y = x.to(device), y.to(device)
                        x_noisy = apply_corruption(x, kind, param_name, val)
                        okn, n = top1_acc(clf, x_noisy, y); num_ok_noisy += okn; tot += n
                        x_d = forward_dae(x_noisy)
                        okd, _ = top1_acc(clf, x_d, y); num_ok_deno += okd
                results["corruptions"][kind][f"{param_name}={val}"] = {
                    "acc_noisy":    float(num_ok_noisy / max(1, tot)),
                    "acc_denoised": float(num_ok_deno  / max(1, tot)),
                }

    ensure_dir(args.out_dir)
    outp = os.path.join(args.out_dir, "dae_cls_eval.json")
    with open(outp, "w") as f: json.dump(results, f, indent=2)
    print("[done] Wrote:", outp)

# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32)

    ap.add_argument('--arch', required=True, choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--ckpt', required=True)

    ap.add_argument('--dae_ckpt', default='None')
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--groups', type=int, default=8)

    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    eval_cls(args)
