import argparse, os, json, math, io
import torch, numpy as np, torch.nn.functional as F
from torch import optim
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from PIL import Image, ImageFilter
import torchvision.transforms as T

from data_utils import make_loader   # expected to return pixel-space [0,1] tensors (no Normalize)
from models import UNetDenoiser

# -------------------- Corruptions (matching your eval flavor) --------------------

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
    # per-image PIL roundtrip
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

def mixed_corruptions(x, t_frac):
    """
    Draw a random corruption per batch with a curriculum on severity via t_frac in [0,1].
    """
    # curriculum severity
    sigma = 0.03 + 0.07 * t_frac       # 0.03 -> 0.10
    p_sp  = 0.01 + 0.03 * t_frac       # 0.01 -> 0.04
    scale = 20.0 + 20.0 * t_frac       # 20   -> 40
    q_jpg = int(80 - 40 * t_frac)      # 80   -> 40
    s_blr = 0.8 + 1.2 * t_frac         # 0.8  -> 2.0

    ops = [
        lambda z: add_gaussian(z, sigma),
        lambda z: add_speckle(z, sigma),
        lambda z: add_saltpepper(z, p_sp),
        lambda z: add_poisson(z, scale),
        lambda z: add_jpeg(z, q_jpg),
        lambda z: add_blur(z, s_blr),
    ]
    op = np.random.choice(ops)
    return op(x)

# -------------------- SSIM (lightweight) -----------------------------------

def _gaussian_window(ch: int, k: int = 11, sigma: float = 1.5, device: torch.device = "cpu"):
    # Build a proper 2D Gaussian kernel and expand to (ch, 1, k, k)
    ax = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')  # (k, k)
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = kernel.view(1, 1, k, k).repeat(ch, 1, 1, 1)  # (ch, 1, k, k)
    return window

def ssim(x, y, k: int = 11, sigma: float = 1.5, C1: float = 0.01**2, C2: float = 0.03**2):
    ch = x.size(1)
    device = x.device
    window = _gaussian_window(ch, k=k, sigma=sigma, device=device)

    mu_x = torch.conv2d(x, window, padding=k//2, groups=ch)
    mu_y = torch.conv2d(y, window, padding=k//2, groups=ch)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sigma_x  = torch.conv2d(x * x, window, padding=k//2, groups=ch) - mu_x_sq
    sigma_y  = torch.conv2d(y * y, window, padding=k//2, groups=ch) - mu_y_sq
    sigma_xy = torch.conv2d(x * y, window, padding=k//2, groups=ch) - mu_xy

    num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (denom + 1e-12)
    return ssim_map.mean()


# -------------------- Utilities --------------------------------------------

def compute_stats(csv, img_size, batch, device):
    loader = make_loader(csv, 'train', img_size, batch, shuffle=False)
    m = torch.zeros(3, device=device); v = torch.zeros(3, device=device); n = 0
    with torch.no_grad():
        for x,_,_ in tqdm(loader, desc="Computing dataset mean/std"):
            x = x.to(device)
            b = x.size(0); n += b
            m += x.mean(dim=(0,2,3)) * b
            v += x.var(dim=(0,2,3), unbiased=False) * b
    mean = (m / n).tolist()
    var  = (v / n).tolist()
    std  = torch.sqrt(v / n).tolist()
    return mean, std

def denorm_tensor(x, mean, std):
    mean = x.new_tensor(mean).view(1,3,1,1)
    std  = x.new_tensor(std).view(1,3,1,1)
    return (x * std + mean).clamp(0,1)

def norm_tensor(x, mean, std):
    mean = x.new_tensor(mean).view(1,3,1,1)
    std  = x.new_tensor(std).view(1,3,1,1)
    return (x - mean) / (std + 1e-8)

# -------------------- Training --------------------------------------------

def train(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    train_loader = make_loader(args.csv, 'train', args.img_size, args.batch_size, True)
    val_loader   = make_loader(args.csv, 'val',   args.img_size, args.batch_size, False)

    # stats
    mean, std = compute_stats(args.csv, args.img_size, args.batch_size, device)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    # model/opt
    model = UNetDenoiser(in_ch=3, base=args.base, groups=args.groups).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = AveragedModel(model, avg_fn=lambda avg, cur, n: avg*0.99 + cur*0.01)

    best = {'metric': 1e9, 'epoch': -1}

    for epoch in range(args.epochs):
        # ---------------- train ----------------
        model.train()
        pbar = tqdm(train_loader, desc=f"DAE Epoch {epoch+1}/{args.epochs}")
        t_frac = epoch / max(1, args.epochs - 1)
        for x,_,_ in pbar:
            x = x.to(device)

            # corruption curriculum
            x_noisy = mixed_corruptions(x, t_frac)

            # normalize inputs for the model; model outputs pixel-space [0,1]
            x_noisy_n = norm_tensor(x_noisy, mean, std)
            pred = model(x_noisy_n)

            # losses: MSE + SSIM (perceptual)
            mse = F.mse_loss(pred, x)
            ssim_term = 1.0 - ssim(pred, x)
            loss = mse + args.lambda_ssim * ssim_term

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ema.update_parameters(model)

            pbar.set_postfix(mse=float(mse.item()), ssim=float(1.0-ssim_term.item()))

        sched.step()

        # ---------------- val ----------------
        model.eval()
        ema_model = ema
        ema_model.eval()
        mses = []; ssims = []

        with torch.no_grad():
            for x,_,_ in val_loader:
                x = x.to(device)
                x_noisy = mixed_corruptions(x, t_frac)  # evaluate at current severity
                x_noisy_n = norm_tensor(x_noisy, mean, std)
                pred = ema_model(x_noisy_n)  # EMA for stabler val
                mses.append(F.mse_loss(pred, x).item())
                ssims.append(ssim(pred, x).item())

        val_mse  = float(np.mean(mses))
        val_ssim = float(np.mean(ssims))

        with open(os.path.join(args.out_dir, "val_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "val_mse": val_mse, "val_ssim": val_ssim,
                                "lr": sched.get_last_lr()[0]})+"\n")

        # save best on MSE (you can also use (MSE - SSIM) combo)
        if val_mse < best['metric']:
            best = {'metric': val_mse, 'epoch': epoch, 'val_ssim': val_ssim}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.out_dir, "best_ema.pt"))

    with open(os.path.join(args.out_dir, "best.json"), "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--groups', type=int, default=8)
    ap.add_argument('--lambda_ssim', type=float, default=0.15)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    train(args)
