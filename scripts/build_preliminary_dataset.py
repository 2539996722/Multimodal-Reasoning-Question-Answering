"""Build preliminary train JSON for LLaMA-Factory fine-tuning.

Modes:
  (default)        Render every page of every PDF, export ALL questions.
  --only-pages     Render only specified pages per doc; export only matching
                   questions. JSON mapping doc_stem -> [page_no].
  --only-images    Do NOT render. Export only questions whose referenced
                   image path is in the given JSON list.
  --groups         Space-separated list of group ids to keep, e.g. "2 3".
                   Default: keep all groups.
  --skip-render    Reuse existing PNGs in imgs/ without re-rendering.

Output: single-turn messages in mllm_demo.json style, with forward-slash
relative paths under imgs/.
"""
import argparse
import json
import re
from pathlib import Path

import fitz

ROOT = Path(r"E:\第七期-Vibe Coding\Code部分\Agent 自主交付实战\多模态文档处理")
DOC_DIR = ROOT / "preliminary_dataset" / "train" / "documents"
QUESTIONS = ROOT / "preliminary_dataset" / "train" / "questions.jsonl"
OUT_IMG_DIR = ROOT / "imgs"
DEFAULT_OUT = ROOT / "preliminary_train.json"

PAGE_PATTERN = re.compile(r"第\s*(\d+)\s*页")
RENDER_DPI = 180


def render_pdf_pages(pdf_path, out_dir, only_pages):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    page_map = {}
    zoom = RENDER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    targets = set(range(1, doc.page_count + 1)) if not only_pages else set(only_pages)
    for i in range(doc.page_count):
        page_no = i + 1
        if page_no not in targets:
            continue
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_file = out_dir / f"page_{page_no}.png"
        pix.save(str(out_file))
        page_map[page_no] = out_file
    doc.close()
    return page_map


def build_answer_text(answer_letter, options):
    idx = ord(answer_letter.upper()) - ord("A")
    if 0 <= idx < len(options):
        return f"答案：{answer_letter}。{options[idx]}"
    return f"答案：{answer_letter}。"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only-pages", default=None,
                   help="JSON mapping doc_stem -> [page_no, ...].")
    p.add_argument("--only-images", default=None,
                   help="JSON list of relative image paths.")
    p.add_argument("--groups", nargs="+", type=int, default=None,
                   help="Space-separated list of group ids to keep.")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--skip-render", action="store_true",
                   help="Reuse existing PNGs without re-rendering.")
    return p.parse_args()


def main():
    args = parse_args()
    if not DOC_DIR.exists():
        raise SystemExit(f"missing docs dir: {DOC_DIR}")
    if not QUESTIONS.exists():
        raise SystemExit(f"missing questions: {QUESTIONS}")

    only_pages = {}
    if args.only_pages:
        only_pages = {k: set(v) for k, v in json.loads(args.only_pages).items()}
    only_images = set(json.loads(args.only_images)) if args.only_images else set()
    keep_groups = set(args.groups) if args.groups else None
    out_json = Path(args.out) if args.out else DEFAULT_OUT

    # 1. Read questions.
    records = []
    with QUESTIONS.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            obj = json.loads(raw_line)
            if keep_groups is not None and obj.get("group") not in keep_groups:
                continue
            doc = obj["document"]
            stem = Path(doc).stem
            m = PAGE_PATTERN.search(obj["question"])
            page_no = int(m.group(1)) if m else None
            records.append({"obj": obj, "stem": stem, "page": page_no})

    print(f"kept {len(records)} questions after group filter", flush=True)

    # 2. Decide which docs/pages need rendering.
    doc_pages = {}
    if not args.skip_render and not only_images:
        if only_pages:
            render_stems = list(only_pages.keys())
        else:
            render_stems = sorted(set(r["stem"] for r in records))
        print(f"rendering {len(render_stems)} docs...", flush=True)
        for stem in render_stems:
            pdf = DOC_DIR / f"{stem}.pdf"
            if not pdf.exists():
                print(f"  [warn] missing PDF for {stem}")
                continue
            out_dir = OUT_IMG_DIR / stem
            page_map = render_pdf_pages(pdf, out_dir, only_pages.get(stem))
            doc_pages[stem] = page_map
            print(f"  rendered {stem}: {sorted(page_map)}", flush=True)

    # 3. Assemble samples.
    samples = []
    skipped = []
    for r in records:
        obj = r["obj"]
        stem = r["stem"]
        page_no = r["page"] or 1
        rel_img = Path("imgs") / stem / f"page_{page_no}.png"
        rel_img_str = rel_img.as_posix()

        if only_pages:
            if stem not in only_pages or page_no not in only_pages[stem]:
                continue
        if only_images and rel_img_str not in only_images:
            continue
        if not (ROOT / rel_img_str).exists():
            skipped.append((obj.get("id"), f"image missing: {rel_img_str}"))
            continue

        options = obj.get("options", [])
        options_block = "\n".join(options)
        user_text = f"<image>{obj['question']}\n{options_block}"
        assistant_text = build_answer_text(obj["answer"], options)
        samples.append({
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "images": [rel_img_str],
        })

    out_json.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(samples)} samples -> {out_json}")
    if skipped:
        print(f"skipped {len(skipped)} entries")
        for sid, reason in skipped[:20]:
            print(f"  id={sid}: {reason}")


if __name__ == "__main__":
    main()