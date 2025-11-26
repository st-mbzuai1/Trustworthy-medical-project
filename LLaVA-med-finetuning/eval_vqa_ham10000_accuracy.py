#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate LLaVA(-Med) VQA predictions on HAM10000 7-class diagnosis.

- Reads ground-truth JSONL and predictions JSONL (one JSON object per line).
- Robustly parses predictions that may mention:
   * a letter (e.g., "Option B", "Answer: (B)")
   * a code (e.g., "bcc")
   * a spelled-out diagnosis (e.g., "Basal cell carcinoma")
   * mixed forms (e.g., "diagnosis of BCC, which stands for Basal Cell Carcinoma")
- Maps everything to one of: akiec, bcc, bkl, df, mel, nv, vasc
- Handles British spellings (naevus/naevi, haemorrhage), punctuation, and unicode quirks.
- Resolves conflicts between a parsed letter and a parsed code via --conflict_policy.
- Outputs metrics as JSON (accuracy, counts, confusion matrix).

Usage:
  python eval_ham_vqa.py \
      --gt gt.jsonl \
      --pred preds.jsonl \
      --out results.json \
      --conflict_policy any

conflict_policy:
  any     -> if a code is found, use it; else use letter->code mapping (default)
  code    -> always use the parsed code (ignore any letter)
  letter  -> always use the letter mapping (ignore any code)
"""

import argparse
import json
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

CODES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}

# ---- Synonyms / phrases for code extraction (lowercased regex) ----
SYN2CODE = {
    # akiec
    r"\bactinic\s+keratos(?:is|es)\b": "akiec",
    r"\bbowen'?s\s+disease\b": "akiec",
    r"\bintraepithelial\s+carcinoma\b": "akiec",
    r"\bsquamous\s+cell\s+carcinoma\s+in\s+situ\b": "akiec",
    r"\bscc\s+in\s+situ\b": "akiec",

    # bcc
    r"\bbasal\s+cell\s+carcinoma\b": "bcc",
    r"\bbcc\b": "bcc",

    # bkl
    r"\bbenign\s+keratosis-?like\s+lesion[s]?\b": "bkl",
    r"\bkeratosis-?like\b": "bkl",
    r"\bseborrhoe?ic\s+keratos(?:is|es)\b": "bkl",
    r"\bsolar\s+lentigo\b": "bkl",
    r"\blichen[-\s]?planus[-\s]?like\s+keratosis\b": "bkl",
    r"\blplk\b": "bkl",

    # df
    r"\bdermatofibroma\b": "df",
    r"\bfibroma\b": "df",

    # mel
    r"\bmelanoma\b": "mel",

    # nv (American & British spellings; variants)
    r"\bmelanocytic\s+nevi\b": "nv",
    r"\bmelanocytic\s+nev(?:us|i)\b": "nv",
    r"\bnaevus\b": "nv",
    r"\bnaevi\b": "nv",
    r"\bnevocellular\s+nev(?:us|i)\b": "nv",
    r"\bcommon\s+nev(?:us|i)\b": "nv",

    # vasc
    r"\bvascular\s+lesion[s]?\b": "vasc",
    r"\bangioma\b": "vasc",
    r"\bangiokeratoma\b": "vasc",
    r"\bpyogenic\s+granuloma\b": "vasc",
    r"\bhemorrhage\b": "vasc",
    r"\bhaemorrhage\b": "vasc",
}

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def normalize_whitespace(s: str) -> str:
    s = _nfkc(s)
    # strip zero-width chars
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def clean_prediction_text(text: str) -> str:
    """Clean but preserve parentheses so '(nv)' stays parsable."""
    if not text:
        return ""
    t = _nfkc(text)

    # Common echo patterns; remove everything after they appear
    t = re.sub(r"The\s+candidate\s+Options\s+are:\s*\[[\s\S]*$", "", t, flags=re.I)
    t = re.sub(r"Options\s+are:\s*\[[\s\S]*$", "", t, flags=re.I)

    # Drop bracketed lists (keep parentheses)
    t = re.sub(r"\[[^\]]+\]", "", t)

    # Normalize dashes and whitespace
    t = t.replace("–", "-").replace("—", "-")
    t = normalize_whitespace(t)
    return t

def extract_letter(s: str) -> Optional[str]:
    """Extract A–G in many formats (Option B, Answer: (B), 'B: bcc', 'B) bcc')."""
    if not s:
        return None
    low = s.lower()

    patterns = [
        r"(?:answer\s*(?:is)?|choose|select|prediction|predicted|option|choice)\s*[:\-]?\s*\(?\s*([a-g])\s*\)?\b",
        r"\boption\s*\(?\s*([a-g])\s*\)?\b",
        r"\bchoice\s*\(?\s*([a-g])\s*\)?\b",
        r"(?:^|[\s\(])([a-g])\s*[\)\.:\-]\s",   # "B) bcc", "B: bcc"
        r"^\s*([a-g])\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, low, flags=re.I)
        if m:
            return m.group(1).upper()
    return None

def extract_code_from_text(s: str) -> Optional[str]:
    """Directly find a code/synonym inside free-form text."""
    if not s:
        return None
    low = s.lower()

    # direct code token
    m = re.search(r"\b(akiec|bcc|bkl|df|mel|nv|vasc)\b", low, flags=re.I)
    if m:
        return m.group(1).lower()

    # synonyms / phrases
    for pat, code in SYN2CODE.items():
        if re.search(pat, low, flags=re.I):
            return code

    # code in parentheses, e.g., "(nv)"
    m = re.search(r"\(([a-z]{2,5})\)", low, flags=re.I)
    if m and m.group(1).lower() in CODES:
        return m.group(1).lower()

    return None

def _norm_alnum(s: str) -> str:
    """Lowercase + NFKC, keep only letters/digits as word stream for tolerant contains()."""
    s = _nfkc(s).lower()
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def _option_code_and_phrase_chunks(options_str: str) -> Dict[str, List[str]]:
    """
    From the full options string, build code -> list[phrases to match].
    E.g., "B: bcc – Basal cell carcinoma" -> {"bcc": ["bcc", "basal cell carcinoma"]}
    """
    out = defaultdict(list)
    if not options_str:
        return out

    s = options_str.replace("–", "-").replace("—", "-")
    parts = re.split(r"\s*,\s*(?=[A-G]\s*:)", s)
    for part in parts:
        # Try "B: bcc - Basal cell carcinoma ..."
        m = re.search(r"\b([A-G])\s*:\s*([a-z]{2,5})\b\s*[-:]\s*(.*)$", part, flags=re.I)
        if m:
            code = m.group(2).lower()
            if code in CODES:
                out[code].append(code)  # include the code itself
                long_phrase = m.group(3).strip()
                # remove parentheses content and split into chunks
                long_phrase = re.sub(r"\(.*?\)", "", long_phrase)
                for chunk in re.split(r"[,/;]", long_phrase):
                    chunk = normalize_whitespace(chunk)
                    if len(chunk) >= 3:
                        out[code].append(chunk)
        else:
            # Simpler "B: bcc"
            m2 = re.search(r"\b([A-G])\s*:\s*([a-z]{2,5})\b", part, flags=re.I)
            if m2:
                code = m2.group(2).lower()
                if code in CODES:
                    out[code].append(code)
    return out

def extract_code_via_options_text(pred_text: str, options_str: str) -> Optional[str]:
    """
    Match against actual phrases present in the options string.
    Handles outputs that spell out the diagnosis (e.g., 'Basal Cell Carcinoma').
    """
    if not pred_text or not options_str:
        return None
    norm_pred = _norm_alnum(pred_text)
    code2phr = _option_code_and_phrase_chunks(options_str)
    for code, phrases in code2phr.items():
        for ph in phrases:
            if not ph:
                continue
            if _norm_alnum(ph) in norm_pred:
                return code
    return None

def extract_pred_letter_and_code(pred_text: str, options_str: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extract both letter and code from prediction text, with a reason string for debugging.
    Order:
      1) clean + extract letter
      2) extract code (direct or synonyms)
      3) fallback: match against options phrases
    """
    text = clean_prediction_text(pred_text)
    reason = ""
    letter = extract_letter(text)
    code = extract_code_from_text(text)
    if code is None:
        code = extract_code_via_options_text(text, options_str)
        if code:
            reason = "options_phrase_match"
    else:
        reason = "code_or_synonym"
    if letter and not reason:
        reason = "letter_only"
    if not letter and not code:
        reason = "unparsed"
    return letter, code, reason

def build_letter2code_from_options(options_str: str) -> Dict[str, str]:
    """
    Parse "A: akiec – Actinic keratoses..., B: bcc – Basal cell carcinoma, ..." into {'A':'akiec', ...}
    """
    out = {}
    if not options_str:
        return out
    s = options_str.replace("–", "-").replace("—", "-")
    for part in re.split(r"\s*,\s*(?=[A-G]\s*:)", s):
        m = re.search(r"\b([A-G])\s*:\s*([a-z]{2,5})\b", part, flags=re.I)
        if m:
            out[m.group(1).upper()] = m.group(2).lower()
    return out

def resolve_prediction(letter: Optional[str], code: Optional[str], letter2code: Dict[str, str], policy: str) -> Optional[str]:
    """
    Choose the final predicted code from parsed letter/code under the conflict policy.
    - 'any'   : prefer code if present; else map letter
    - 'code'  : use code only
    - 'letter': use letter->code only
    """
    letter_code = letter2code.get(letter) if letter else None

    if policy == "code":
        return code
    if policy == "letter":
        return letter_code

    # 'any' (default): prefer code if we have it; otherwise fallback to letter mapping
    return code or letter_code

def parse_gt_code(gt_obj: dict) -> Optional[str]:
    """
    Extract the ground-truth code from:
      1) fig_caption like "F:nv — Melanocytic nevi"
      2) text like "... (nv)."
      3) answer-like fields if present (rare)
    """
    # 1) fig_caption "F:nv — ..."
    fc = gt_obj.get("fig_caption") or ""
    m = re.search(r"[A-G]\s*:\s*([a-z]{2,5})\b", fc, flags=re.I)
    if m and m.group(1).lower() in CODES:
        return m.group(1).lower()
    m2 = re.search(r"\b([A-G])\s*:\s*([a-z]{2,5})", fc, flags=re.I)
    if m2 and m2.group(2).lower() in CODES:
        return m2.group(2).lower()

    # Sometimes "F:nv — ..." is without space
    m3 = re.search(r"^[A-G]\s*:\s*([a-z]{2,5})", fc, flags=re.I)
    if m3 and m3.group(1).lower() in CODES:
        return m3.group(1).lower()

    # Rare: code present in text
    txt = gt_obj.get("text") or ""
    code = extract_code_from_text(txt)
    if code in CODES:
        return code

    # Fallback: try letter via options
    letter2code = build_letter2code_from_options(gt_obj.get("options", ""))
    m4 = re.search(r"\b([A-G])\b", fc, flags=re.I)
    if m4:
        return letter2code.get(m4.group(1).upper())

    return None

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Ground-truth JSONL")
    ap.add_argument("--pred", required=True, help="Predictions JSONL")
    ap.add_argument("--out", required=True, help="Where to save JSON summary")
    ap.add_argument("--conflict_policy", default="any", choices=["any", "code", "letter"], help="Resolve letter/code conflicts")
    args = ap.parse_args()

    gt_list = load_jsonl(args.gt)    # each has question_id, prompt, options, text, fig_caption (typical)
    pred_list = load_jsonl(args.pred)  # each has question_id, text (model output), maybe answer_id, model_id

    gt_by_qid = {d["question_id"]: d for d in gt_list}
    pred_by_qid = {d["question_id"]: d for d in pred_list}

    totals = 0
    correct = 0
    missing_predictions = 0
    unparsed_predictions = 0
    echoed_options = 0

    # per-class support/correct
    support = Counter()
    per_class_correct = Counter()

    # confusion: gt -> pred -> count
    conf = {c: Counter() for c in sorted(CODES)}

    # debug reasons
    parse_reasons = Counter()

    for qid, gt in gt_by_qid.items():
        gt_code = parse_gt_code(gt)
        if gt_code not in CODES:
            # skip if GT couldn't be parsed
            continue

        support[gt_code] += 1
        totals += 1

        if qid not in pred_by_qid:
            missing_predictions += 1
            conf[gt_code]["<missing>"] += 1
            continue

        ptxt = pred_by_qid[qid].get("text", "") or ""
        if re.search(r"The\s+candidate\s+Options\s+are:", ptxt, flags=re.I):
            echoed_options += 1

        letter, code, reason = extract_pred_letter_and_code(ptxt, gt.get("options", ""))
        parse_reasons[reason] += 1

        letter2code = build_letter2code_from_options(gt.get("options", ""))
        pred_code = resolve_prediction(letter, code, letter2code, args.conflict_policy)

        if pred_code not in CODES:
            unparsed_predictions += 1
            conf[gt_code]["<unparsed>"] += 1
            continue

        conf[gt_code][pred_code] += 1

        if pred_code == gt_code:
            correct += 1
            per_class_correct[gt_code] += 1

    acc = (correct / totals) if totals > 0 else 0.0

    # Build JSON-friendly confusion
    confusion_json = {gt: dict(sorted(cnt.items())) for gt, cnt in conf.items()}

    # Per-class accuracy
    per_class = {}
    for c in sorted(CODES):
        s = support[c]
        pc = per_class_correct[c]
        per_class[c] = {
            "support": int(s),
            "correct": int(pc),
            "accuracy": (pc / s) if s > 0 else None,
        }

    summary = {
        "total_evaluated": int(totals),
        "correct": int(correct),
        "accuracy": acc,
        "missing_predictions": int(missing_predictions),
        "unparsed_predictions": int(unparsed_predictions),
        "echoed_options_in_prediction_text": int(echoed_options),
        "conflict_policy": args.conflict_policy,
        "parse_reasons": dict(parse_reasons),  # e.g., {'code_or_synonym': 840, 'letter_only': 200, 'options_phrase_match': 400, 'unparsed': 12}
        "per_class": per_class,
        "confusion": confusion_json,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also print a concise summary to stdout
    print(json.dumps({
        "total_evaluated": totals,
        "correct": correct,
        "accuracy": round(acc, 4),
        "missing_predictions": missing_predictions,
        "unparsed_predictions": unparsed_predictions,
        "echoed_options": echoed_options,
        "parse_reasons": parse_reasons,
    }, indent=2, default=int))

if __name__ == "__main__":
    main()
