#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILES = {
    "Clean": "/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/fintune/CV-8501-Assignment-2/LLaVA-med-finetuning/outputs/ham10000_llavamed_answers_val_finetune_acc_resport_clean.jsonl",
    "FGSM ε=8": "/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/fintune/CV-8501-Assignment-2/LLaVA-med-finetuning/outputs/ham10000_llavamed_answers_val_finetune_acc_resport_fdsm_8.jsonl",
    "FGSM ε=16": "/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/fintune/CV-8501-Assignment-2/LLaVA-med-finetuning/outputs/ham10000_llavamed_answers_val_finetune_acc_resport_fdsm_16.jsonl",
    "FGSM ε=8 (denoised)": "/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/fintune/CV-8501-Assignment-2/LLaVA-med-finetuning/outputs/ham10000_llavamed_answers_val_finetune_acc_resport_fdsm_8_denoised.jsonl",
    "FGSM ε=16 (denoised)": "/home/tuan.vo/CV8501/MCM/CV8502_FA_Tuan-Van-Vo_LLaVA-Med_code/fintune/CV-8501-Assignment-2/LLaVA-med-finetuning/outputs/ham10000_llavamed_answers_val_finetune_acc_resport_fdsm_16_denoised.jsonl",
}

LETTER2DESC = {
    "A": "akiec — Actinic keratoses / IEC (Bowen’s)",
    "B": "bcc — Basal cell carcinoma",
    "C": "bkl — Benign keratosis-like (SK, lentigo, LPLK)",
    "D": "df — Dermatofibroma",
    "E": "mel — Melanoma",
    "F": "nv — Melanocytic nevi",
    "G": "vasc — Vascular lesions",
}

OUTDIR = Path("ham10000_llavamed_report")
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_single_json(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, list) and len(obj) == 1:
            obj = obj[0]
        return obj
    except Exception:
        # JSONL fallback
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line:
                    continue
                return json.loads(line)
    raise ValueError(f"Could not parse JSON from {jsonl_path}")

# ----- Load -----
results = {tag: load_single_json(p) for tag, p in FILES.items()}

# ----- Overall summary -----
rows = []
for tag, obj in results.items():
    rows.append({
        "Condition": tag,
        "Total": obj.get("total_evaluated"),
        "Correct": obj.get("correct"),
        "Accuracy": obj.get("accuracy"),
        "Missing": obj.get("missing_predictions"),
    })
df_overall = pd.DataFrame(rows).set_index("Condition")
clean_acc = df_overall.loc["Clean", "Accuracy"]
df_overall["Δ Accuracy vs. Clean"] = df_overall["Accuracy"] - clean_acc
df_overall.to_csv(OUTDIR / "overall_summary.csv")

# ----- Per-class summary from confusion -----
pc_rows = []
for tag, obj in results.items():
    conf = obj.get("confusion", {})
    for letter in ["A","B","C","D","E","F","G"]:
        pred_counts = conf.get(letter, {})
        support = sum(int(v) for v in pred_counts.values())
        correct = int(pred_counts.get(letter, 0))
        acc = (correct / support) if support > 0 else np.nan
        pc_rows.append({
            "Condition": tag,
            "Class": letter,
            "Description": LETTER2DESC[letter],
            "Support": support,
            "Correct": correct,
            "Accuracy": acc
        })
df_pc = pd.DataFrame(pc_rows)
df_pc.to_csv(OUTDIR / "per_class_summary.csv", index=False)

# ----- LaTeX overall table -----
latex = "\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
latex += "Condition & Total & Correct & Accuracy & $\\Delta$ vs.\\ Clean \\\\\n\\midrule\n"
for tag, row in df_overall.iterrows():
    latex += f"{tag} & {int(row['Total'])} & {int(row['Correct'])} & {row['Accuracy']:.3f} & {row['Δ Accuracy vs. Clean']:+.3f} \\\\\n"
latex += "\\bottomrule\n\\end{tabular}\n"
latex += "\\caption{\\textbf{Overall accuracy across conditions} for LLaVA-Med (LoRA fine-tuned) on HAM10000. $\\Delta$ is the accuracy change vs. Clean.}\n"
latex += "\\label{tab:ham_overall}\n\\end{table}\n"
(OUTDIR / "overall_table.tex").write_text(latex, encoding="utf-8")

# ----- Figures (matplotlib only, one chart per figure, no custom colors) -----
# 1) Overall accuracy bar chart
plt.figure(figsize=(8,4))
plt.bar(df_overall.index.tolist(), df_overall["Accuracy"].values)
plt.ylim(0,1.0)
for i, v in enumerate(df_overall["Accuracy"].values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
plt.ylabel("Accuracy")
plt.title("Overall Accuracy by Condition (HAM10000)")
plt.tight_layout()
plt.savefig(OUTDIR / "overall_accuracy.png", dpi=200)
plt.close()

# 2) Per-class grouped bars
conditions = list(results.keys())
classes = ["A","B","C","D","E","F","G"]
mat = np.zeros((len(classes), len(conditions)))
for ci, cls in enumerate(classes):
    for ti, tag in enumerate(conditions):
        val = df_pc[(df_pc.Condition==tag) & (df_pc.Class==cls)]["Accuracy"].values[0]
        mat[ci, ti] = 0 if np.isnan(val) else val

plt.figure(figsize=(max(8, 1.4*len(conditions)), 5))
x = np.arange(len(classes))
width = 0.8 / max(1, len(conditions))
for ti, tag in enumerate(conditions):
    plt.bar(x + (ti - (len(conditions)-1)/2)*width, mat[:, ti], width=width, label=tag)
plt.xticks(x, classes)
plt.ylim(0,1.0)
plt.ylabel("Per-class Accuracy")
plt.title("Per-class Accuracy by Condition (HAM10000)")
plt.legend(fontsize=8, ncols=min(3, len(conditions)))
plt.tight_layout()
plt.savefig(OUTDIR / "per_class_accuracy.png", dpi=200)
plt.close()

print("Wrote to:", OUTDIR)
print(" - overall_summary.csv")
print(" - per_class_summary.csv")
print(" - overall_table.tex")
print(" - overall_accuracy.png")
print(" - per_class_accuracy.png")
