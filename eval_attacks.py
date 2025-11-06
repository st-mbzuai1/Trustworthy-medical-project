
import argparse, os, json, numpy as np, torch, torch.nn.functional as F
from tqdm import tqdm
from data_utils import make_loader
from models import build_classifier, UNetDenoiser
from metrics import multiclass_metrics
from attacks import fgsm, pgd, deepfool, adversarial_patch

def run_inference(model, dae, x):
    if dae is not None: x = dae(x)
    return model(x)

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = make_loader(args.csv, 'val', args.img_size, args.batch_size, False)
    model = build_classifier(args.arch, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()
    dae = None
    if args.dae_ckpt and args.dae_ckpt.lower()!='none':
        dae = UNetDenoiser(in_ch=3, base=32).to(device)
        dae.load_state_dict(torch.load(args.dae_ckpt, map_location=device)); dae.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    # Clean
    all_logits, all_y = [], []
    with torch.no_grad():
        for x,y,_ in tqdm(val_loader, desc="Clean Eval"):
            x,y = x.to(device), y.to(device)
            all_logits.append(run_inference(model, dae, x).cpu().numpy()); all_y.append(y.cpu().numpy())
    logits = np.concatenate(all_logits); y_true = np.concatenate(all_y)
    with open(os.path.join(args.out_dir, "clean_metrics.json"), "w") as f: json.dump(multiclass_metrics(y_true, logits, 7), f, indent=2)

    results = {"fgsm":{}, "pgd":{}, "deepfool":{}, "patch":{}}
    # FGSM / PGD sweeps
    for eps in args.eps_list:
        # FGSM through pipeline if needed
        all_logits, all_y = [], []
        for x,y,_ in tqdm(val_loader, desc=f"FGSM eps={eps}/255"):
            x,y = x.to(device), y.to(device)
            if dae:
                x_req = x.clone().detach().requires_grad_(True)
                loss = F.cross_entropy(model(dae(x_req)), y)
                grad = torch.autograd.grad(loss, x_req)[0]
                x_adv = torch.clamp(x + (eps/255.0)*grad.sign(), 0, 1).detach()
            else:
                x_adv = fgsm(model, x, y, eps)
            with torch.no_grad():
                logits = run_inference(model, dae, x_adv)
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(y.cpu().numpy())
        results["fgsm"][str(eps)] = multiclass_metrics(np.concatenate(all_y), np.concatenate(all_logits), 7)

        # PGD (adaptive if dae)
        all_logits, all_y = [], []
        for x,y,_ in tqdm(val_loader, desc=f"PGD eps={eps}/255 steps={args.pgd_steps}"):
            x,y = x.to(device), y.to(device)
            if dae:
                x_nat = x.clone().detach(); eps_f = eps/255.0; alpha = max(1, eps//4)/255.0
                x_adv = torch.clamp(x_nat + torch.empty_like(x_nat).uniform_(-eps_f, eps_f), 0, 1)
                for _ in range(args.pgd_steps):
                    x_adv.requires_grad_(True)
                    loss = F.cross_entropy(model(dae(x_adv)), y)
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    x_adv = x_adv + alpha * grad.sign()
                    x_adv = torch.max(torch.min(x_adv, x_nat + eps_f), x_nat - eps_f)
                    x_adv = torch.clamp(x_adv, 0, 1).detach()
            else:
                x_adv = pgd(model, x, y, eps=eps, alpha=max(1, eps//4), steps=args.pgd_steps, rand_start=True)
            with torch.no_grad():
                logits_eval = run_inference(model, dae, x_adv)
            all_logits.append(logits_eval.detach().cpu().numpy()); all_y.append(y.cpu().numpy())
        results["pgd"][str(eps)] = multiclass_metrics(np.concatenate(all_y), np.concatenate(all_logits), 7)

    # DeepFool
    all_logits, all_y = [], []
    for x,y,_ in tqdm(val_loader, desc="DeepFool"):
        x,y = x.to(device), y.to(device)
        pipe = model if dae is None else torch.nn.Sequential(dae, model)
        pipe.eval()
        x_adv = deepfool(pipe, x)
        # when logging metrics, also use the same pipeline
        with torch.no_grad():
            logits = pipe(x_adv)
        all_logits.append(logits.detach().cpu().numpy()); all_y.append(y.cpu().numpy())
    results["deepfool"]["na"] = multiclass_metrics(np.concatenate(all_y), np.concatenate(all_logits), 7)

    # # Patch (untargeted)
    # all_logits, all_y = [], []
    # for x,y,_ in tqdm(val_loader, desc="Patch (untargeted)"):
    #     x,y = x.to(device), y.to(device)
    #     x_adv = adversarial_patch(lambda z: run_inference(model, dae, z), x, y=y, patch_frac=args.patch_frac, steps=200, targeted=False)
    #     with torch.no_grad():
    #         logits_eval = run_inference(model, dae, x_adv)
    #     all_logits.append(logits_eval.detach().cpu().numpy()); all_y.append(y.cpu().numpy())
    # results["patch"]["untargeted"] = multiclass_metrics(np.concatenate(all_y), np.concatenate(all_logits), 7)

    with open(os.path.join(args.out_dir, "robust_metrics.json"), "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dae_ckpt', default='None'); ap.add_argument('--eps_list', type=int, nargs='+', default=[1,2,4,8])
    ap.add_argument('--pgd_steps', type=int, default=10); ap.add_argument('--patch_frac', type=float, default=0.1)
    ap.add_argument('--out_dir', required=True); args = ap.parse_args(); evaluate(args)
