import argparse, json, os, sys
import pandas as pd

# Fixed 7-class mapping and option order
CLASS_ORDER = [
    ("A", "akiec", "Actinic keratoses and intraepithelial carcinoma (Bowen’s disease)"),
    ("B", "bcc",   "Basal cell carcinoma"),
    ("C", "bkl",   "Benign keratosis-like lesions (incl. seborrheic keratosis, solar lentigo, LPLK)"),
    ("D", "df",    "Dermatofibroma"),
    ("E", "mel",   "Melanoma"),
    ("F", "nv",    "Melanocytic nevi"),
    ("G", "vasc",  "Vascular lesions (e.g., angioma, angiokeratoma, pyogenic granuloma, hemorrhage)"),
]

OPTIONS_STRING = ", ".join([f"{k}: {code} – {name}" for k, code, name in CLASS_ORDER])
CODE2LETTER = {code: k for k, code, _ in CLASS_ORDER}
CODE2LONG   = {code: name for _, code, name in CLASS_ORDER}

QUESTION_STEM = (
    "This is a medical Question with several Options, and there is only one correct answer among these options. "
    "Please select the correct answer for the question. Remember, you can only select one option. "
    "The Question is: Which single best diagnosis matches the dermoscopic lesion shown in the image? "
    f"The candidate Options are: [{OPTIONS_STRING}]"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: image_path,label,split")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--split", default=None, help="Optional split filter (e.g., test/val/train)")
    ap.add_argument("--images_root", default="", help="Optional root to prefix image_path (for relative paths)")
    ap.add_argument("--model_id", default="llava-med-v1.5-mistral-7b")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"image_path", "label", "split"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {required}, got {list(df.columns)}")

    if args.split:
        before = len(df)
        df = df[df["split"].astype(str).str.lower() == args.split.lower()]
        print(f"[INFO] Filtered split='{args.split}': {before} -> {len(df)} rows", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_total = 0
    n_skipped_unknown_label = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            n_total += 1
            img_rel = str(row["image_path"]).strip()
            label_code = str(row["label"]).strip().lower()

            if label_code not in CODE2LETTER:
                n_skipped_unknown_label += 1
                continue

            letter = CODE2LETTER[label_code]
            long_name = CODE2LONG[label_code]

            # question_id: use filename stem for readability
            base = os.path.basename(img_rel)
            qid = os.path.splitext(base)[0] or base

            # image path to place in JSON (prefix root if provided)
            img_for_json = os.path.join(args.images_root, img_rel) if args.images_root else img_rel

            record = {
                "question_id": qid,
                "prompt": QUESTION_STEM,
                "text": f"The correct answer for the question is: {long_name} ({label_code}).",
                "options": OPTIONS_STRING,
                "fig_caption": f"{letter}:{label_code} — {long_name}",
                "answer": letter,          # convenient for evaluation
                "image": img_for_json,
                "model_id": args.model_id,
                "metadata": {}
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote {args.out}", file=sys.stderr)
    if n_skipped_unknown_label:
        print(f"[NOTE] Skipped rows with unknown label codes: {n_skipped_unknown_label}", file=sys.stderr)
    print(f"[STATS] Processed rows: {n_total}", file=sys.stderr)

if __name__ == "__main__":
    main()
