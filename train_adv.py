
import argparse, os, json, numpy as np, torch, torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from data_utils import make_loader
from models import build_classifier
from metrics import multiclass_metrics
from attacks import pgd

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = make_loader(args.csv, 'train', args.img_size, args.batch_size, True)
    val_loader   = make_loader(args.csv, 'val',   args.img_size, args.batch_size, False)
    model = build_classifier(args.arch, num_classes=7, pretrained=True).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = {'metric': -1, 'epoch': -1}
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"AT-{args.arch} Epoch {epoch+1}/{args.epochs}")
        for x,y,_ in pbar:
            x,y = x.to(device), y.to(device)
            x_adv = pgd(model, x, y, eps=args.eps, alpha=max(1, args.eps//4), steps=args.steps, rand_start=True)
            loss = F.cross_entropy(model(x_adv), y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss.item()))
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x,y,_ in val_loader:
                x,y = x.to(device), y.to(device)
                all_logits.append(model(x).cpu().numpy()); all_y.append(y.cpu().numpy())
        import numpy as np
        m = multiclass_metrics(np.concatenate(all_y), np.concatenate(all_logits), 7)
        with open(os.path.join(args.out_dir, "val_log.jsonl"), "a") as f: f.write(json.dumps({"epoch": epoch, **m})+"\n")
        if m['macro_auroc'] > best['metric']:
            best = {'metric': m['macro_auroc'], 'epoch': epoch}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
    with open(os.path.join(args.out_dir, "best.json"), "w") as f: json.dump(best, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--eps', type=int, default=4); ap.add_argument('--steps', type=int, default=7)
    args = ap.parse_args(); train(args)
