
# HAM10000 Trustworthy Suite (Training + Defenses + Attacks + Comprehensive Eval)

**What you get**
- Models: **ResNet50**, **DenseNet121**, **EfficientNet-B0**, **DINO ViT-B/16** (`--arch` flag)
- Defenses: **PGD Adversarial Training**, **Denoising Autoencoder (U-Net)**
- Attacks: **FGSM**, **PGD**, **DeepFool**,

### Outputs can be found here : https://drive.google.com/file/d/1arK0F118Ol5IS5e9_iosP8pJ0exBYKM4/view?usp=sharing


## 0) Environment
```bash
conda create -n hamrob python=3.10 -y
conda activate hamrob
pip install -r requirements.txt
```

## 1) Data
```
data/ham10000/HAM10000_images_part_1/*.jpg
data/ham10000/HAM10000_images_part_2/*.jpg
data/ham10000/HAM10000_metadata.csv
```
Build split CSV:
```bash
python download_dataset.py
```

Build split CSV:
```bash
python build_ham10000_csv.py --data_root data/ham10000   --out_csv data/ham10000/labels.csv --val_frac 0.15 --seed 42
```

## 2) Train
**Clean**
```bash
python train_clean.py --arch resnet50        --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_resnet50
python train_clean.py --arch densenet121     --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_densenet121
python train_clean.py --arch efficientnet_b0 --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_efficientnet_b0
python train_clean.py --arch dino            --csv data/ham10000/labels.csv --img_size 224 --batch_size 32 --epochs 10 --out_dir outputs/clean_dino
```
**Adversarial training (PGD-AT)**
```bash
python train_adv.py --arch resnet50        --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_resnet50        --eps 4 --steps 7
python train_adv.py --arch densenet121     --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_densenet121     --eps 4 --steps 7
python train_adv.py --arch efficientnet_b0 --csv data/ham10000/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_efficientnet_b0 --eps 4 --steps 7
python train_adv.py --arch dino            --csv data/ham10000/labels.csv --img_size 224 --batch_size 32 --epochs 10 --out_dir outputs/adv_dino            --eps 4 --steps 7
```
**Denoising autoencoder**
```bash
python train_dae.py --csv data/ham10000/labels.csv --img_size 256 --epochs 10 --out_dir outputs/dae_unet
```

## 3) Evaluate Attacks
Example:
```bash
python eval_attacks.py --arch resnet50 --csv data/ham10000/labels.csv --img_size 256   --ckpt outputs/clean_resnet50/best.pt --dae_ckpt None   --eps_list 1 2 4 8 --pgd_steps 10 --out_dir outputs/eval_clean_resnet50
```

## 4) Visualize Attacks
```bash
python visualize_attacks.py --arch resnet50 --csv data/ham10000/labels.csv   --img_size 256 --ckpt outputs/clean_resnet50/best.pt --dae_ckpt None   --num_samples 12 --eps 4 --pgd_steps 10 --out_dir outputs/viz_resnet50
```

## 5) Comprehensive Grid (all models + defenses + DAE)
Edit `grid.json` (already populated) and run:
```bash
python run_eval_grid.py --config grid.json
python aggregate_summary.py --eval_root outputs/eval_grid --out_csv outputs/eval_grid/summary.csv
python make_tables.py --summary outputs/eval_grid/summary.csv --out_dir outputs/eval_grid/tables
python more_plots.py --summary outputs/eval_grid/summary.csv --out_dir outputs/eval_grid/plots
```

## 6) other attack types experiments(gaussian noise, salt pepper)
train:
```bash
python train_dae_2.py --csv data /ham10000/labels.csv --img_size 256 --epochs 10 --out_dir outputs/dae_unet_2
```

 eval:
```bash
python eval_dae_cls.py \

   --csv data/ham10000/labels.csv \

   --img_size 256 \

   --batch_size 32 \

   --arch resnet50 \

   --ckpt outputs/clean_resnet50/best.pt \

   --dae_ckpt outputs/dae_unet_2/best.pt \

   --base 32 --groups 8 \

   --out_dir outputs/dae_unet_changed/eval_cls_resnet50
```



