
import argparse, os, pandas as pd
def main(args):
    df = pd.read_csv(args.summary); os.makedirs(args.out_dir, exist_ok=True)
    clean = df[df.attack=="clean"][["job","acc","macro_auroc","macro_auprc","sens_at_sp95","ece"]]
    clean.to_markdown(os.path.join(args.out_dir,'clean_table.md'), index=False)
    with open(os.path.join(args.out_dir,'clean_table.tex'),'w') as f:
        f.write(clean.to_latex(index=False, float_format="%.4f", escape=False, caption="Clean performance across jobs", label="tab:clean"))
    pgd = df[df.attack=="pgd"].pivot_table(index="job", columns="eps", values="acc")
    pgd.to_markdown(os.path.join(args.out_dir,'pgd_acc_table.md'))
    with open(os.path.join(args.out_dir,'pgd_acc_table.tex'),'w') as f:
        f.write(pgd.to_latex(float_format="%.4f", escape=False, caption="Accuracy under PGD at different $\\epsilon$", label="tab:pgd"))
    print("Tables written to", args.out_dir)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--summary', required=True); ap.add_argument('--out_dir', required=True)
    args = ap.parse_args(); main(args)
