#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, csv
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T, torchvision.utils as vutils
import torchvision.models as tv

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

# -------------------- Minimal models: classifier + DAE --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.ReLU(inplace=True))
    def forward(self,x): return self.seq(x)

class UNetDenoiser(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.d1=DoubleConv(in_ch,base); self.p1=nn.MaxPool2d(2)
        self.d2=DoubleConv(base,base*2); self.p2=nn.MaxPool2d(2)
        self.d3=DoubleConv(base*2,base*4); self.p3=nn.MaxPool2d(2)
        self.d4=DoubleConv(base*4,base*8)
        self.up3=nn.ConvTranspose2d(base*8,base*4,2,2); self.u3=DoubleConv(base*8,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,2); self.u2=DoubleConv(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,2,2);   self.u1=DoubleConv(base*2,base)
        self.out=nn.Conv2d(base,in_ch,1)
    def forward(self,x):
        c1=self.d1(x); c2=self.d2(self.p1(c1)); c3=self.d3(self.p2(c2)); c4=self.d4(self.p3(c3))
        u3=self.u3(torch.cat([self.up3(c4),c3],1))
        u2=self.u2(torch.cat([self.up2(u3),c2],1))
        u1=self.u1(torch.cat([self.up1(u2),c1],1))
        return self.out(u1)

def build_classifier(arch, num_classes=7, pretrained=False):
    a = arch.lower()
    if a == 'resnet50':
        m = tv.resnet50(weights=None if not pretrained else tv.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if a == 'densenet121':
        m = tv.densenet121(weights=None if not pretrained else tv.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes); return m
    if a == 'efficientnet_b0':
        m = tv.efficientnet_b0(weights=None if not pretrained else tv.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes); return m
    if a in ('dino','dino_vitb16','vitb16_dino'):
        if not _HAS_TIMM:
            raise RuntimeError("timm is required for DINO model; pip install timm")
        return timm.create_model("vit_base_patch16_224.dino", pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")

def load_flex(model, ckpt_path, strict=False, device='cpu'):
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw
    if isinstance(raw, dict):
        for k in ('state_dict','model','net','ema_state_dict'):
            if k in raw and isinstance(raw[k], dict):
                sd = raw[k]; break
    out = {}
    for k,v in sd.items():
        kk = k
        for p in ('module.','model.','net.','backbone.'):
            if kk.startswith(p): kk = kk[len(p):]
        out[kk]=v
    model.load_state_dict(out, strict=strict)
    return model

# -------------------- I/O helpers (no normalization) ----------------------
def load_image(path, img_size, device):
    img = Image.open(path).convert('RGB')
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    return tf(img).unsqueeze(0).to(device)

def save_img(t, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    vutils.save_image(t.clamp(0,1), p, nrow=1)

# -------------------- FGSM attack (same math: eps/255) --------------------
def fgsm_like_eval(model, dae, x, y, eps):
    eps_f = eps/255.0
    x_req = x.clone().detach().requires_grad_(True)
    if dae is not None:
        logits = model(dae(x_req))
    else:
        logits = model(x_req)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_req)[0]
    x_adv = torch.clamp(x + eps_f*grad.sign(), 0, 1).detach()
    return x_adv

# -------------------- Labels mapping (supports str/int in CSV) -----------
CODES = ["akiec","bcc","bkl","df","mel","nv","vasc"]
CODE2ID = {c:i for i,c in enumerate(CODES)}

def label_to_id(lbl):
    """Supports 'nv' or '5' or 'Melanocytic nevi' (best effort)."""
    if isinstance(lbl, int):
        return lbl
    s = str(lbl).strip().lower()
    if s in CODE2ID:
        return CODE2ID[s]
    # Try integer-ish
    try:
        return int(s)
    except Exception:
        pass
    # Fuzzy names
    if "basal" in s: return CODE2ID["bcc"]
    if "melanoma" in s and "in situ" not in s: return CODE2ID["mel"]
    if "dermatofibroma" in s: return CODE2ID["df"]
    if "keratos" in s and ("actinic" in s or "bowen" in s): return CODE2ID["akiec"]
    if "keratos" in s or "lentigo" in s or "lplk" in s: return CODE2ID["bkl"]
    if "melanocytic" in s or "nev" in s or "naev" in s: return CODE2ID["nv"]
    if "vascular" in s or "angioma" in s or "angiokeratoma" in s or "granuloma" in s or "hemorrhage" in s or "haemorrhage" in s:
        return CODE2ID["vasc"]
    raise ValueError(f"Unrecognized label: {lbl}")

# -------------------- Main dataset loop -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help="labels.csv with columns: image_path,label,split")
    ap.add_argument('--image_root', required=True, help="Root folder so we can build absolute image path")
    ap.add_argument('--out_root', required=True, help="Output root to save fgsm and denoised images")
    ap.add_argument('--splits', default='test', help="Comma list: test,val,train,all")
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--ckpt', required=True, help='Classifier checkpoint')
    ap.add_argument('--dae_ckpt', default='None', help='DAE checkpoint (UNet); "None" to skip denoise branch')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--eps_list', type=int, nargs='+', default=[8], help='FGSM eps values (pixel scale)')
    ap.add_argument('--skip_existing', action='store_true', help='Skip samples if both adv and denoised already exist')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build & load classifier
    clf = build_classifier(args.arch, num_classes=7, pretrained=False).to(device).eval()
    clf = load_flex(clf, args.ckpt, strict=False, device='cpu').to(device).eval()

    # Optional DAE
    dae = None
    if args.dae_ckpt and args.dae_ckpt.lower() != 'none':
        dae = UNetDenoiser(in_ch=3, base=32).to(device).eval()
        dae = load_flex(dae, args.dae_ckpt, strict=False, device='cpu').to(device).eval()

    # Prepare split filter
    want_splits = set(s.strip().lower() for s in (args.splits.split(',') if args.splits else []))
    if 'all' in want_splits or not want_splits:
        want_splits = {'train','val','test'}

    # Load CSV
    rows = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            split = r.get('split','').strip().lower()
            if split in want_splits:
                rows.append(r)

    print(f"Total images to process (splits {sorted(want_splits)}): {len(rows)}")

    # Process each image
    processed = 0
    for i, r in enumerate(rows, 1):
        img_rel = r['image_path'].strip()
        label_raw = r['label'].strip()
        split = r.get('split','').strip().lower()

        # Make absolute path (if already absolute, join will just ignore root)
        img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(args.image_root, img_rel)
        if not os.path.isfile(img_abs):
            print(f"[WARN] Missing image: {img_abs}")
            continue

        # Compute label id
        try:
            yid = label_to_id(label_raw)
        except Exception as e:
            print(f"[WARN] {e}; skipping {img_rel}")
            continue
        y = torch.tensor([yid], dtype=torch.long, device=device)

        # Load image
        x = load_image(img_abs, args.img_size, device)

        # Relative path inside out_root (preserve directories, keep same filename but .jpg)
        rel_noext = os.path.splitext(img_rel)[0]
        rel_png = rel_noext + ".jpg"  # save as PNG

        for eps in args.eps_list:
            out_adv = os.path.join(args.out_root, f"fgsm_eps{eps}", rel_png)
            out_den = os.path.join(args.out_root, f"fgsm_eps{eps}_denoised", rel_png)

            if args.skip_existing:
                adv_done = os.path.isfile(out_adv)
                den_done = (dae is None) or os.path.isfile(out_den)
                if adv_done and den_done:
                    continue

            # FGSM
            x_adv = fgsm_like_eval(clf, dae, x, y, eps)
            save_img(x_adv.detach().cpu(), out_adv)

            # Denoised
            if dae is not None:
                with torch.no_grad():
                    x_den = dae(x_adv).clamp(0,1)
                save_img(x_den.detach().cpu(), out_den)

        processed += 1
        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] processed...")

    print(f"Done. Processed: {processed} images.")
    print("Outputs were written under:", args.out_root)
    print("Subfolders per eps: fgsm_eps{eps}/ and fgsm_eps{eps}_denoised/ preserving original relative paths.")

if __name__ == '__main__':
    main()
