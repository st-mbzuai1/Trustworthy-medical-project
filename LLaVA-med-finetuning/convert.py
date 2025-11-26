#!/usr/bin/env python3
import json, argparse, re
from pathlib import Path

CHOICES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

SYS_PROMPT = (
    "You are a dermatology assistant. Look at the dermoscopic image and answer the diagnosis "
    "as a single token. Choices: akiec, bcc, bkl, df, mel, nv, vasc. Output exactly one of these, "
    "lowercase, no punctuation or explanation."
)

def normalize_label(s: str) -> str:
    if not s: return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for tok in s.split():
        if tok in CHOICES: return tok
    for c in CHOICES:
        if c in s: return c
    return s.split()[0] if s else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", delsfault="training_json/train.jsonl", help="Input JSONL (one example per line)")
    ap.add_argument("--dst", default="training_json/llava_train.json", help="Output JSON (array)")
    ap.add_argument("--image_root", default=".", help="Root dir relative to which the `image` paths are valid")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    out = []
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            img_rel = ex.get("image")  # e.g., "preprocessed/images/ISIC_....jpg"
            q_text = ex.get("question") or "What is the diagnosis of this skin lesion?"
            gt = normalize_label(ex.get("answer", ""))

            # LLaVA conversation format:
            # - The user ("human") message must include the image token and the instruction.
            # - The assistant ("gpt") responds with the single-token label.
            # If your local LLaVA build uses "<image>" instead of special token triplet, you can switch accordingly.
            human_value = "<image>\n" + SYS_PROMPT + "\nQuestion: " + q_text
            assistant_value = gt

            item = {
                "id": ex.get("id", f"train-{i}"),
                "image": img_rel,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt",   "value": assistant_value}
                ]
            }
            out.append(item)

    with dst.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # quick sanity check: can json.load it back?
    with dst.open("r", encoding="utf-8") as f:
        _ = json.load(f)

    print(f"Wrote {len(out)} examples to {dst}")

if __name__ == "__main__":
    main()
