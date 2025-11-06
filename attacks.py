
import torch, torch.nn.functional as F
import numpy as np

def fgsm(model, x, y, eps):
    eps = eps/255.0
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + eps * grad.sign()
    return torch.clamp(x_adv, 0, 1).detach()

def pgd(model, x, y, eps, alpha, steps, rand_start=True):
    eps_f = eps/255.0; alpha_f = alpha/255.0
    x_nat = x.clone().detach()
    x_adv = x.clone().detach()
    if rand_start:
        x_adv = torch.clamp(x_adv + torch.empty_like(x_adv).uniform_(-eps_f, eps_f), 0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha_f * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_nat + eps_f), x_nat - eps_f)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

def deepfool(model, x, num_classes=7, max_iter=50, overshoot=0.02):
    model.eval()
    x_adv = x.clone().detach()
    for i in range(x.size(0)):
        xi = x_adv[i:i+1].clone().detach().requires_grad_(True)
        with torch.no_grad():
            pred = model(xi).argmax(dim=1).item()
        for _ in range(max_iter):
            logits = model(xi)
            if logits.argmax(dim=1).item() != pred: break
            grad_orig = torch.autograd.grad(logits[0, pred], xi, retain_graph=True)[0]
            min_val, w = 1e10, None
            for j in range(num_classes):
                if j == pred: continue
                grad_j = torch.autograd.grad(logits[0, j], xi, retain_graph=True)[0]
                wj = grad_j - grad_orig
                fj = (logits[0, j] - logits[0, pred]).detach()
                val = torch.abs(fj) / (wj.view(-1).norm() + 1e-12)
                if val < min_val: min_val, w = val, wj
            r = (min_val + 1e-4) * w / (w.view(-1).norm() + 1e-12)
            xi = (xi + (1+overshoot) * r).detach().requires_grad_(True)
        x_adv[i:i+1] = xi.detach()
    return torch.clamp(x_adv, 0, 1)

def adversarial_patch(model, x, y=None, patch_frac=0.1, steps=300, targeted=False, target_class=0, lr=5e-2):
    B, C, H, W = x.shape
    size = int((patch_frac**0.5) * min(H, W))
    patch = torch.rand(B, C, size, size, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x_clone = x.clone()
        for i in range(B):
            top = np.random.randint(0, H - size + 1)
            left = np.random.randint(0, W - size + 1)
            x_clone[i, :, top:top+size, left:left+size] = torch.sigmoid(patch[i])
        logits = model(x_clone)
        if targeted:
            tgt = torch.full((B,), target_class, dtype=torch.long, device=x.device)
            loss = -F.cross_entropy(logits, tgt)
        else:
            assert y is not None, "y required for untargeted patch"
            loss = F.cross_entropy(logits, y)
        loss.backward(); opt.step()
    return torch.clamp(x_clone.detach(), 0, 1)
