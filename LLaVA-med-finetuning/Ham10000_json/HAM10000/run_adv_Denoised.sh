# Example paths (adjust to your setup)
CSV=/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/HAM10000/labels.csv
IMG_ROOT=/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/HAM10000/images_train
OUT_ROOT=/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/HAM10000/HAM10000_adv

# resnet50 classifier, FGSM eps=8 only, test split, with DAE
python generate_ham10000_fgsm.py \
  --csv "$CSV" \
  --image_root "$IMG_ROOT" \
  --out_root "$OUT_ROOT" \
  --splits val \
  --arch resnet50 \
  --ckpt /home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/attacked/outputs/clean_resnet50/best.pt \
  --dae_ckpt /home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/attacked/outputs/dae_unet/best.pt \
  --img_size 256 \
  --eps_list 16 \
  --skip_existing
