
import argparse, os, json, numpy as np, torch
from tqdm import tqdm
from data_utils import make_loader
from models import build_classifier
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from metrics import multiclass_metrics

def apply_corruption(x, kind, severity):
    if kind=="gaussian_noise":
        std = 0.02 * severity
        return torch.clamp(x + torch.randn_like(x)*std, 0,1)
    if kind=="blur":
        k = 3 + 2*(severity-1)
        return T.GaussianBlur(kernel_size=k)(x)
    if kind=="jpeg":
        q = max(10, 100 - severity*15)
        x_img = T.ToPILImage()(x[0].cpu())
        buf = T.functional.pil_to_tensor(T.ToPILImage()(x[0].cpu())).float()/255.0
        # Per-sample JPEG (batchwise approximate): apply to each item
        out = []
        for i in range(x.size(0)):
            pil = T.ToPILImage()(x[i].cpu())
            pil.save("/tmp/tmp.jpg", quality=q)
            out.append(T.ToTensor()(T.Image.open("/tmp/tmp.jpg").convert("RGB")))
        return torch.stack(out,0).to(x.device)
    if kind=="contrast":
        fac = 1.0 + 0.2*severity
        return torch.clamp(TF.adjust_contrast(x, fac), 0,1)
    if kind=="brightness":
        fac = 1.0 + 0.2*severity
        return torch.clamp(TF.adjust_brightness(x, fac), 0,1)
    return x

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = make_loader(args.csv, 'val', args.img_size, args.batch_size, False)
    model = build_classifier(args.arch, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    kinds = ["gaussian_noise","blur","contrast","brightness"]
    severities = [1,2,3,4,5]
    results = {}
    for kind in kinds:
        results[kind] = {}
        for s in severities:
            all_logits, all_y = [], []
            for x,y,_ in tqdm(val_loader, desc=f"{kind} s={s}"):
                x,y = x.to(device), y.to(device)
                x_c = apply_corruption(x, kind, s)
                with torch.no_grad():
                    all_logits.append(model(x_c).cpu().numpy()); all_y.append(y.cpu().numpy())
            import numpy as np
            m = multiclass_metrics(np.concatenate(all_y), np.concatenate(all_logits), 7)
            results[kind][str(s)] = m
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"corruptions_metrics.json"),"w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True); args = ap.parse_args(); main(args)
