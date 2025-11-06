
import argparse, os, torch, torchvision.utils as vutils, torch.nn.functional as F
from data_utils import make_loader
from models import build_classifier, UNetDenoiser
from attacks import fgsm, pgd, deepfool, adversarial_patch

def save_grid(tensor, path, nrow=6): vutils.save_image(tensor, path, nrow=nrow)

def visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = make_loader(args.csv, 'val', args.img_size, args.batch_size, False)
    model = build_classifier(args.arch, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()
    dae = None
    if args.dae_ckpt and args.dae_ckpt.lower()!='none':
        dae = UNetDenoiser(in_ch=3, base=32).to(device)
        dae.load_state_dict(torch.load(args.dae_ckpt, map_location=device)); dae.eval()

    xs, ys = [], []
    for x,y,_ in loader:
        xs.append(x); ys.append(y)
        if len(torch.cat(xs,0)) >= args.num_samples: break
    x = torch.cat(xs,0)[:args.num_samples].to(device)
    y = torch.cat(ys,0)[:args.num_samples].to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    save_grid(x.cpu(), os.path.join(args.out_dir, "orig_grid.png"))

    if dae:
        x_req = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(dae(x_req)), y)
        grad = torch.autograd.grad(loss, x_req)[0]
        x_fgsm = torch.clamp(x + (args.eps/255.0)*grad.sign(), 0, 1).detach()
    else:
        x_fgsm = fgsm(model, x, y, eps=args.eps)
    save_grid(x_fgsm.cpu(), os.path.join(args.out_dir, f"fgsm_eps{args.eps}.png"))

    if dae:
        x_nat = x.clone().detach(); eps_f = args.eps/255.0; alpha = max(1,args.eps//4)/255.0
        x_pgd = torch.clamp(x_nat + torch.empty_like(x_nat).uniform_(-eps_f, eps_f), 0, 1)
        for _ in range(args.pgd_steps):
            x_pgd.requires_grad_(True)
            loss = F.cross_entropy(model(dae(x_pgd)), y)
            grad = torch.autograd.grad(loss, x_pgd)[0]
            x_pgd = x_pgd + alpha * grad.sign()
            x_pgd = torch.max(torch.min(x_pgd, x_nat + eps_f), x_nat - eps_f)
            x_pgd = torch.clamp(x_pgd, 0, 1).detach()
    else:
        x_pgd = pgd(model, x, y, eps=args.eps, alpha=max(1,args.eps//4), steps=args.pgd_steps, rand_start=True)
    save_grid(x_pgd.cpu(), os.path.join(args.out_dir, f"pgd_eps{args.eps}.png"))

    x_df = deepfool((lambda z: model(dae(z))) if dae else model, x)
    save_grid(x_df.cpu(), os.path.join(args.out_dir, "deepfool.png"))

    x_patch = adversarial_patch(lambda z: (model(dae(z)) if dae else model(z)), x, y=y, patch_frac=args.patch_frac, steps=200, targeted=False)
    save_grid(x_patch.cpu(), os.path.join(args.out_dir, "patch.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dae_ckpt', default='None'); ap.add_argument('--num_samples', type=int, default=12)
    ap.add_argument('--eps', type=int, default=4); ap.add_argument('--pgd_steps', type=int, default=10)
    ap.add_argument('--patch_frac', type=float, default=0.1); ap.add_argument('--out_dir', required=True)
    args = ap.parse_args(); visualize(args)
