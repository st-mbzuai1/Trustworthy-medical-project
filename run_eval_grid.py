
import argparse, json, os, subprocess, shlex
def main(args):
    with open(args.config,'r') as f: cfg = json.load(f)
    csv = cfg['csv']; eps_list = cfg.get('eps_list', [1,2,4,8]); pgd_steps = cfg.get('pgd_steps', 10)
    out_root = cfg.get('out_root', 'outputs/eval_grid'); jobs = cfg['jobs']
    os.makedirs(out_root, exist_ok=True)
    for job in jobs:
        name, arch, ckpt = job['name'], job['arch'], job['ckpt']
        dae  = job.get('dae', 'None'); img_size = job.get('img_size', 256); batch = job.get('batch_size', 32)
        out_dir = os.path.join(out_root, name); os.makedirs(out_dir, exist_ok=True)
        cmd = f"python scripts/eval_attacks.py --arch {arch} --csv {csv} --img_size {img_size} --batch_size {batch} --ckpt {ckpt} --dae_ckpt {dae} --eps_list {' '.join(map(str, eps_list))} --pgd_steps {pgd_steps} --out_dir {out_dir}"
        print('\n[RUN]', cmd); subprocess.run(shlex.split(cmd), check=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args(); main(args)
