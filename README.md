
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
Download data:
```bash
python download_dataset.py
mkdir data
mv ham10000_data/ data
```

Build split CSV:
```bash
python build_ham10000_csv.py --data_root data/ham10000_data   --out_csv data/ham10000_data/labels.csv --val_frac 0.15 --seed 42
```

## 2) Train
**Clean**
```bash
python train_clean.py --arch resnet50        --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_resnet50
python train_clean.py --arch densenet121     --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_densenet121
python train_clean.py --arch efficientnet_b0 --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_efficientnet_b0
python train_clean.py --arch dino            --csv data/ham10000_data/labels.csv --img_size 224 --batch_size 32 --epochs 10 --out_dir outputs/clean_dino
```
**Adversarial training (PGD-AT)**
```bash
python train_adv.py --arch resnet50        --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_resnet50        --eps 4 --steps 7
python train_adv.py --arch densenet121     --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_densenet121     --eps 4 --steps 7
python train_adv.py --arch efficientnet_b0 --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/adv_efficientnet_b0 --eps 4 --steps 7
python train_adv.py --arch dino            --csv data/ham10000_data/labels.csv --img_size 224 --batch_size 32 --epochs 10 --out_dir outputs/adv_dino            --eps 4 --steps 7
```
**Denoising autoencoder**
```bash
python train_dae.py --csv data/ham10000_data/labels.csv --img_size 256 --epochs 10 --out_dir outputs/dae_unet
```

## 3) Evaluate Attacks
Example:
```bash
python eval_attacks.py --arch resnet50 --csv data/ham10000_data/labels.csv --img_size 256   --ckpt outputs/clean_resnet50/best.pt --dae_ckpt None   --eps_list 1 2 4 8 --pgd_steps 10 --out_dir outputs/eval_clean_resnet50
```

## 4) Visualize Attacks
```bash
python visualize_attacks.py --arch resnet50 --csv data/ham10000_data/labels.csv   --img_size 256 --ckpt outputs/clean_resnet50/best.pt --dae_ckpt None   --num_samples 12 --eps 4 --pgd_steps 10 --out_dir outputs/viz_resnet50
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

```bash
cd experiments-2
```

train clean models(example) (options for arch: 'resnet50','densenet121','efficientnet_b0','dino'). Set img size to 224 for dino:
```bash
python train_clean.py --arch efficientnet_b0 --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --epochs 10 --out_dir outputs/clean_efficientnet_b0
```

train dae. Set img size to 224 for dino based experiments:
```bash
python train_dae_2.py --csv data/ham10000_data/labels.csv --img_size 256 --epochs 10 --out_dir outputs/dae_unet_2
```

 eval. Set img size to 224 for dino based experiments:
```bash
python eval_dae_cls.py --csv data/ham10000_data/labels.csv --img_size 256 --batch_size 32 --arch efficientnet_b0 --ckpt outputs/clean_efficientnet_b0/best.pt --dae_ckpt outputs/dae_unet_2/best.pt --base 32 --groups 8 --out_dir outputs/dae_unet_changed/eval_cls_efficientnet_b0
```

below two run well 

```bash
python train_all_and_dae.py   --csv data/ham10000_data/labels.csv   --img_size 256   --batch_size 32   --epochs 30   --dae_epochs 10
```

```bash
python eval_all_with_dae.py   --csv data/ham10000_data/labels.csv   --img_size 256   --batch_size 32
```


## Reformulate the classification as closed-ended VQA and evaluate the performance of the pre-trained LLaVA-Med.

We will work in the LLaVA-Med official repo for this(inference). 

The json format for vqa question answering for Ham10000 dataset with LLava-Med: LLaVA-med-finetuning/Ham10000_json

The predictions will be saved to answers_file.jsonl given below (using the fixed test file)

1. This is for Zero-shot LLaVA-Med VQA on HAM10000 for clean: 
git clone https://github.com/microsoft/LLaVA-Med.git
cd LLaVA-Med
python llava/eval/model_vqa.py --conv-mode mistral_instruct --model-path microsoft/llava-med-v1.5-mistral-7b --question-file Ham10000_json/HAM10000/ham10000_vqa_val.jsonl --image-folder path_to_clean_Ham10000 --answers-file /path_to_answers_file.jsonl --temperature 0.0
```
2.  This is for Zero-shot LLaVA-Med VQA on HAM10000 under attacked: 

we need generated the attacked sample from clean Ham10000 here: Ham10000_json/HAM10000/generate_ham10000_fgsm.py

git clone https://github.com/microsoft/LLaVA-Med.git
cd LLaVA-Med
python llava/eval/model_vqa.py --conv-mode mistral_instruct --model-path microsoft/llava-med-v1.5-mistral-7b --question-file Ham10000_json/HAM10000/ham10000_vqa_val.jsonl --image-folder path_to_attacked_sample_Ham10000 --answers-file /path_to_answers_file_attacked.jsonl --temperature 0.0
```
We get score metrics using the below script(in the repo root)
```
python eval_vqa_ham10000_accuracy.py --gt Ham10000_json/HAM10000/ham10000_vqa_val.jsonl --pr /path_to_answers_file.jsonl --out ./eval_ham1000_vqa_Acc.jsonl

python eval_vqa_ham10000_accuracy.py --gt Ham10000_json/HAM10000/ham10000_vqa_val.jsonl --pr /path_to_answers_file_attacked.jsonl --out ./eval_ham1000_vqa_Acc_attacked.jsonl

```

## Step - 4 - Fine-tune VLM: Fine-tune LLaVA-Med for improved performance(using LORA) (it is done on the whole training split)
Note that for this step it is best to build a new environment using conda or pip from the LLaVA-med-finetuning folder. 
This is the modified LLaVA(fixed) repo to do the finetuning (refer to [here for relevant issue](https://github.com/microsoft/LLaVA-Med/issues/87)) and this [click](https://github.com/haotian-liu/LLaVA/issues/1423#issuecomment-2068356012)

⚠️ **Important:** In train.sh modify the --data_path, --image_folder and --output_dir . The datapath is the questions file which was already made during the preprocessing step. The relevant checkpoints will be saved based on what --output_dir is set. 

⚠️ **Important:** Also make sure that the model-path is set to microsoft/llava-med-v1.5-mistral-7b.

⚠️ **Important:** Also make sure the output dir has the word "llava" and "lora" in it. Check out this github issue - [click](https://github.com/haotian-liu/LLaVA/issues/1567). For safety you can have the name as /path/llava-lora-XXX

⚠️ **Important:** We need generated the LLava-vqa trainning format for Ham10000. Please find code here: Ham10000_json/HAM10000/convert_to_llava_train.py


```
cd LLaVA-med-finetuning
```
```
bash train.sh 
```

⚠️ **Very Important:** After training, we need to merge the checkpoints of lora with the base model. [check this](https://github.com/microsoft/LLaVA-Med/issues/87#issuecomment-2293292356)

Note that --model-path should be the path to one of the saved checkpoint which was obtained during the previous step (eg. llava-lora_ckpts_finetune/checkpoint-2400).


--save-model-path is where the merged weights will be saved.
```
python merge_lora_weights.py --model-path /path_to_saved_checkpoint_based_on_output_dir --model-base microsoft/llava-med-v1.5-mistral-7b --save-model-path /path_to_merged_checkpoint
```


We do inference using the merged checkpoints using the following script(Note - this has to be done from the llava-med repo). The predictions will be saved to answers_file.jsonl given below
```
python llava/eval/model_vqa.py --conv-mode mistral_instruct --model-path microsoft/llava-med-v1.5-mistral-7b --question-file Ham10000_json/HAM10000/ham10000_vqa_val.jsonl --image-folder path_to_clean_Ham10000 --answers-file /path_to_answers_file.jsonl --temperature 0.0
```

We get score metrics using the below script(from the repo root)
```
python eval_vqa_ham10000_accuracy.py --gt Ham10000_json/HAM10000/ham10000_vqa_val.jsonl --pr /path_to_answers_file.jsonl --out ./eval_ham1000_vqa_Acc.jsonl
```



