#!/usr/bin/env python3
import json, argparse, os, sys
from typing import Any, Dict, Iterable, List

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                print(f"[warn] Skipping line {ln}: cannot parse JSON ({e})", file=sys.stderr)

def resolve_image(rec: Dict[str, Any], image_root: str) -> str:
    img = rec.get("image") or rec.get("image_path")
    if not img:
        raise ValueError("record missing 'image' or 'image_path'")
    img = img.strip()
    if image_root and not os.path.isabs(img):
        img = os.path.join(image_root, img)
    return img

def to_llava_item(rec: Dict[str, Any], image_root: str) -> Dict[str, Any]:
    qid = rec.get("question_id") or rec.get("id") or rec.get("qid") or ""
    prompt = rec.get("prompt") or rec.get("question") or rec.get("text_prompt") or rec.get("text") or ""
    answer = rec.get("answer") or rec.get("gpt_answer") or rec.get("text") or ""

    if not prompt:
        raise ValueError("missing 'prompt' (question text)")
    if not answer:
        raise ValueError("missing 'answer'/'text' (answer text)")

    image_path = resolve_image(rec, image_root)

    return {
        "id": str(qid),
        "image": image_path,
        "conversations": [
            {"from": "human", "value": "<image>\n" + prompt.strip()},
            {"from": "gpt",   "value": answer.strip()}
        ]
    }

def main():
    ap = argparse.ArgumentParser(description="Convert HAM10000 VQA JSONL -> LLaVA train JSON")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--image_root", default="", help="Optional prefix folder for image paths")
    args = ap.parse_args()

    out: List[Dict[str, Any]] = []
    total = keep = skipped = 0

    for rec in read_jsonl(args.input_jsonl):
        total += 1
        try:
            out.append(to_llava_item(rec, args.image_root))
            keep += 1
        except Exception as e:
            skipped += 1
            print(f"[warn] skipping record {total}: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Done. Read={total} | Kept={keep} | Skipped={skipped}")
    print(f"Wrote: {args.output_json}")

if __name__ == "__main__":
    main()
