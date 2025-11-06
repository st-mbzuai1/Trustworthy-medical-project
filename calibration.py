
import argparse, os, json, numpy as np, torch
from data_utils import make_loader
from models import build_classifier
from metrics import softmax_np, ece
import matplotlib.pyplot as plt

def nll(logits, y):
    # logits [N,C], y [N]
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = make_loader(args.csv, 'val', args.img_size, args.batch_size, False)
    model = build_classifier(args.arch, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    logits_list, y_list = [], []
    with torch.no_grad():
        for x,y,_ in val_loader:
            x = x.to(device)
            logits_list.append(model(x).cpu().numpy()); y_list.append(y.numpy())
    logits = np.concatenate(logits_list); y = np.concatenate(y_list)
    probs = softmax_np(logits)
    before = {"nll": nll(logits,y), "ece": ece(probs, y)}

    # temperature search on a grid
    temps = np.linspace(0.5, 5.0, 46)
    best_T, best_nll = 1.0, 1e9
    for T in temps:
        n = nll(logits / T, y)
        if n < best_nll: best_nll, best_T = n, T
    probs_T = softmax_np(logits / best_T)
    after = {"nll": nll(logits / best_T, y), "ece": ece(probs_T, y), "T": float(best_T)}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"calibration_metrics.json"),"w") as f:
        json.dump({"before": before, "after": after}, f, indent=2)

    # reliability diagram
    def reliability(probs, y, bins=15):
        conf = probs.max(axis=1); pred = probs.argmax(axis=1); acc = (pred==y).astype(float)
        edges = np.linspace(0,1,bins+1); mids = (edges[:-1]+edges[1:])/2
        accs, confs = [], []
        for i in range(bins):
            m = (conf>edges[i]) & (conf<=edges[i+1])
            if m.any():
                accs.append(acc[m].mean()); confs.append(conf[m].mean())
            else:
                accs.append(0.0); confs.append((edges[i]+edges[i+1])/2)
        return mids, accs, confs
    mids, a1, c1 = reliability(probs, y); mids2, a2, c2 = reliability(probs_T, y)
    plt.figure(); plt.plot(mids, a1); plt.plot(mids2, a2)
    plt.plot([0,1],[0,1]); plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability (before vs after T)")
    plt.legend(["Before","After","Ideal"]); plt.savefig(os.path.join(args.out_dir,"reliability.png")); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True); args = ap.parse_args(); main(args)
