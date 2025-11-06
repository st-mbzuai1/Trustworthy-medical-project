
import argparse, os, json, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from data_utils import make_loader
from models import build_classifier
from sklearn.metrics import roc_auc_score

class OODFolder(Dataset):
    def __init__(self, root, img_size):
        paths = []
        for dirpath,_,filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    paths.append(os.path.join(dirpath,f))
        self.paths = paths
        self.tf = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = Image.open(self.paths[i]).convert('RGB')
        return self.tf(x), 0, self.paths[i]

def energy_from_logits(logits):  # -T*logsumexp(logits/T); T=1
    m = logits.max(axis=1, keepdims=True)
    return - (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))).squeeze(1)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    id_loader = make_loader(args.csv, 'val', args.img_size, args.batch_size, False)
    ood_loader = DataLoader(OODFolder(args.ood_dir, args.img_size), batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = build_classifier(args.arch, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    id_scores = []
    with torch.no_grad():
        for x,_,_ in id_loader:
            x = x.to(device); id_scores.append(energy_from_logits(model(x).cpu().numpy()))
    ood_scores = []
    with torch.no_grad():
        for x,_,_ in ood_loader:
            x = x.to(device); ood_scores.append(energy_from_logits(model(x).cpu().numpy()))
    id_scores = np.concatenate(id_scores); ood_scores = np.concatenate(ood_scores)
    # Higher energy => more OOD (since negative logsumexp), so labels: ID=0, OOD=1
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    auroc = float(roc_auc_score(labels, scores))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"ood_energy_auroc.json"),"w") as f: json.dump({"auroc": auroc}, f, indent=2)
    print("OOD AUROC (energy):", auroc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50', choices=['resnet50','densenet121','efficientnet_b0','dino'])
    ap.add_argument('--csv', required=True); ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ood_dir', required=True); ap.add_argument('--out_dir', required=True)
    args = ap.parse_args(); main(args)
