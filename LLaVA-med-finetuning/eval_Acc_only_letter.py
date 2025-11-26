#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute letter-based (A–G) accuracy for HAM10000 VQA.

Input format (JSONL, one object per line):
  {
    "question_id": "ISIC_0034761",
    "answer": "C"   # single letter A–G
  }

You can pass files in either JSONL (recommended) or a single JSON list.
The script aligns by question_id, compares letters case-insensitively,
and outputs a JSON summary with accuracy and confusion matrix.

Usage:
  python eval_letters.py --gt gt.jsonl --pred preds.jsonl --out results.json
"""

import argparse, json, sys, re
from collections import Counter, defaultdict

VALID = set(list("ABCDEFG"))

def load_any(path):
    """Load JSONL (one obj/line) or JSON (list of objs)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return data
    # Try JSONL first
    if "\n" in raw:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data
    # Else single JSON
    obj = json.loads(raw)
    if isinstance(obj, list):
        return obj
    return [obj]

def norm_letter(x):
    """Return uppercase A–G or None."""
    if not isinstance(x, str):
        return None
    x = x.strip().upper()
    return x if x in VALID else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Ground-truth JSONL/JSON")
    ap.add_argument("--pred", required=True, help="Predictions JSONL/JSON")
    ap.add_argument("--out", required=True, help="Where to save JSON summary")
    args = ap.parse_args()

    gt_list = load_any(args.gt)
    pred_list = load_any(args.pred)

    gt_by_qid = {d.get("question_id"): d for d in gt_list if d.get("question_id")}
    pred_by_qid = {d.get("question_id"): d for d in pred_list if d.get("question_id")}

    total = 0
    correct = 0
    missing = 0
    bad_gt = 0
    bad_pred = 0

    # confusion[gt_letter][pred_letter] = count
    confusion = {L: Counter() for L in sorted(VALID)}

    for qid, g in gt_by_qid.items():
        gt_letter = norm_letter(g.get("answer"))
        print("gt_letter",gt_letter)
        if gt_letter is None:
            bad_gt += 1
            continue

        total += 1

        p = pred_by_qid.get(qid)
        if p is None:
            missing += 1
            confusion[gt_letter]["<missing>"] += 1
            continue

        pred_letter = norm_letter(p.get("text"))
        print('pred_letter:', pred_letter)
        if pred_letter is None:
            bad_pred += 1
            confusion[gt_letter]["<invalid>"] += 1
            continue

        confusion[gt_letter][pred_letter] += 1
        if pred_letter == gt_letter:
            correct += 1

    acc = (correct / total) if total > 0 else 0.0

    # Build JSON-friendly confusion
    confusion_json = {gt: dict(sorted(cnt.items())) for gt, cnt in confusion.items()}

    summary = {
        "total_evaluated": total,
        "correct": correct,
        "accuracy": acc,
        "missing_predictions": missing,
        "invalid_gt_answers": bad_gt,
        "invalid_pred_answers": bad_pred,
        "confusion": confusion_json,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also print a short summary
    print(json.dumps(
        {"total_evaluated": total, "correct": correct, "accuracy": round(acc, 4)},
        indent=2
    ))

if __name__ == "__main__":
    main()
