python scripts/train_clean.py --arch resnet50        --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_resnet50
python scripts/train_clean.py --arch densenet121     --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_densenet121
python scripts/train_clean.py --arch efficientnet_b0 --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_efficientnet_b0
python scripts/train_clean.py --arch dino            --csv data/ham10000/labels.csv --img_size 224 --batch_size 32 --epochs 10 --out_dir outputs/clean_dino

python scripts/train_adv.py --arch resnet50        --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_resnet50        --eps 4 --steps 7
python scripts/train_adv.py --arch densenet121     --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_densenet121     --eps 4 --steps 7
python scripts/train_adv.py --arch efficientnet_b0 --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_efficientnet_b0 --eps 4 --steps 7
python scripts/train_adv.py --arch dino            --csv data/ham10000/labels.csv --img_size 224 --batch_size 32 --epochs 10 --out_dir outputs/adv_dino            --eps 4 --steps 7

python scripts/train_dae.py --csv data/ham10000/labels.csv --img_size 256 --epochs 10 --out_dir outputs/dae_unet

python scripts/eval_attacks.py --arch resnet50 --csv data/ham10000/labels.csv --img_size 256   --ckpt outputs/clean_resnet50/best.pt --dae_ckpt None   --eps_list 1 2 4 8 --pgd_steps 10 --out_dir outputs/eval_clean_resnet50
