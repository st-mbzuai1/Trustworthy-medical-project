
import argparse, os, json, numpy as np, torch
from data_utils import make_loader
from models import build_classifier
from metrics import softmax_np
import matplotlib.pyplot as plt

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = make_loader(args.csv, 'val', args.img_size, args.batch_size, False)
    model = build_classifier(args.arch, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    logits, labels = [], []
    with torch.no_grad():
        for x,y,_ in val_loader:
            x = x.to(device)
            logits.append(model(x).cpu().numpy()); labels.append(y.numpy())
    logits = np.concatenate(logits); labels = np.concatenate(labels)
    probs = softmax_np(logits)
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); correct = (pred==labels).astype(float)
    order = np.argsort(-conf)  # high to low
    cov = []; acc = []
    for k in range(1, len(order)+1):
        idx = order[:k]; cov.append(k/len(order)); acc.append(correct[idx].mean())
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"risk_coverage.json"),"w") as f:
        json.dump({"coverage": cov, "accuracy": acc}, f)
    plt.figure(); plt.plot(cov, acc); plt.xlabel("Coverage"); plt.ylabel("Accuracy"); plt.title("Risk–Coverage")
    plt.savefig(os.path.join(args.out_dir,"risk_coverage.png")); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True); args = ap.parse_args(); main(args)
