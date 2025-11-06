
import argparse, os, pandas as pd, glob

def main(args):
    meta = pd.read_csv(os.path.join(args.data_root, 'HAM10000_metadata.csv'))
    paths = {}
    for folder in ['HAM10000_images_part_1','HAM10000_images_part_2']:
        full = os.path.join(args.data_root, folder)
        for p in glob.glob(os.path.join(full, '*.jpg')):
            img_id = os.path.splitext(os.path.basename(p))[0]
            paths[img_id] = p
    rows = []
    for _,r in meta.iterrows():
        img_id = r['image_id']; label = r['dx']
        p = paths.get(img_id, None)
        if p is not None: rows.append((p, label))
    df = pd.DataFrame(rows, columns=['image_path','label'])
    df['split'] = 'train'
    for lab,grp in df.groupby('label'):
        idx = grp.sample(frac=args.val_frac, random_state=args.seed).index
        df.loc[idx,'split'] = 'val'
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False); print("Wrote", args.out_csv, "n=", len(df))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True); ap.add_argument('--out_csv', required=True)
    ap.add_argument('--val_frac', type=float, default=0.15); ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(); main(args)
