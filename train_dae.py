
import argparse, os, json, torch, numpy as np, torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from data_utils import make_loader
from models import UNetDenoiser

def add_corruptions(x):
    noise = torch.randn_like(x) * 0.05
    return torch.clamp(x + noise, 0, 1)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = make_loader(args.csv, 'train', args.img_size, args.batch_size, True)
    val_loader   = make_loader(args.csv, 'val',   args.img_size, args.batch_size, False)
    model = UNetDenoiser(in_ch=3, base=32).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = {'metric': 1e9, 'epoch': -1}
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"DAE Epoch {epoch+1}/{args.epochs}")
        for x,_,_ in pbar:
            x = x.to(device)
            x_noisy = add_corruptions(x)
            loss = F.mse_loss(model(x_noisy), x)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(mse=float(loss.item()))
        # val
        model.eval()
        mses = []
        with torch.no_grad():
            for x,_,_ in val_loader:
                x = x.to(device)
                mses.append(F.mse_loss(model(add_corruptions(x)), x).item())
        import numpy as np
        val_mse = float(np.mean(mses))
        with open(os.path.join(args.out_dir, "val_log.jsonl"), "a") as f: f.write(json.dumps({"epoch": epoch, "val_mse": val_mse})+"\n")
        if val_mse < best['metric']:
            best = {'metric': val_mse, 'epoch': epoch}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
    with open(os.path.join(args.out_dir, "best.json"), "w") as f: json.dump(best, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=16); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out_dir', required=True)
    args = ap.parse_args(); train(args)
