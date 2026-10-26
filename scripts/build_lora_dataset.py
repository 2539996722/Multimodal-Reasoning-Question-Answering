# -*- coding: utf-8 -*-
"""构建 LLaMA-Factory 多模态(LoRA)训练数据集：复赛附图页 -> ShareGPT 样本。

约定（与平台推理输入对齐）：
  - 样本内容 = 目标附图页图像 + 题干原文（逐字保留，含「第N页」表述）
  - 监督目标 = 参考答案原文（短答案，中位数 7 字，贴合 EM/ROUGE-1/BERTScore 口径）
  - 只取能从题干解析出目标页的题；无页码提示的题证据在整篇文本中，单独构造会诱导幻觉，故剔除并记录
  - 训练/验证按 PDF 分组切分，保证同一份文档不跨集（无泄漏）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None

ROOT = Path(__file__).resolve().parent.parent
PAGE_PATTERN = re.compile(r"第\s*(\d+)\s*页")


def load_rows(stage: str, split: str) -> list[dict]:
    base = ROOT / ("preliminary_dataset" if stage == "preliminary" else "semi_final_dataset") / split
    name = "questions.jsonl" if stage == "preliminary" else ("train.jsonl" if split == "train" else "test.jsonl")
    path = base / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def docs_dir(stage: str, split: str) -> Path:
    base = ROOT / ("preliminary_dataset" if stage == "preliminary" else "semi_final_dataset") / split
    return base / "documents"


def render_page(pdf_path: Path, page: int, max_edge: int) -> bytes | None:
    """把指定页渲染为 PNG，最长边不超过 max_edge（控制 image token 数）。"""
    if pymupdf is None:
        raise RuntimeError("缺少 pymupdf，请先安装 PyMuPDF")
    with pymupdf.open(pdf_path) as doc:
        if page < 1 or page > doc.page_count:
            return None
        target = doc[page - 1]
        rect = target.rect
        scale = max_edge / max(rect.width, rect.height)
        matrix = pymupdf.Matrix(scale, scale)
        pixmap = target.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")


def split_documents(documents: list[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    """按 PDF 分组切分：同一文档的题只会出现在一个集合里。"""
    unique = sorted(set(documents))
    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    val = set(shuffled[:val_count])
    train = set(shuffled[val_count:])
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 LLaMA-Factory 多模态 LoRA 训练数据")
    parser.add_argument("--stage", default="semi_final", choices=("semi_final", "preliminary"))
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--out", default="runtime/lora-data", help="输出目录（相对项目根）")
    parser.add_argument("--max-edge", type=int, default=768, help="渲染最长边像素（决定 image token 数）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集 PDF 占比")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--dataset-name", default="patent_fig")
    parser.add_argument("--patch-preprocessor", default="", help="模型目录：把 size.longest_edge 限制为图像上限")
    args = parser.parse_args()

    out_dir = (ROOT / args.out).resolve()
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.stage, args.split)
    pdf_dir = docs_dir(args.stage, args.split)

    usable: list[dict] = []
    skipped_no_page = 0
    skipped_no_doc = 0
    for row in rows:
        match = PAGE_PATTERN.search(str(row.get("question", "")))
        if not match:
            skipped_no_page += 1
            continue
        document = str(row.get("document") or "")
        if not (pdf_dir / document).is_file():
            skipped_no_doc += 1
            continue
        answer = str(row.get("answer") or "").strip()
        if not answer:
            skipped_no_page += 1
            continue
        usable.append({"row": row, "page": int(match.group(1)), "document": document, "answer": answer})

    train_docs, val_docs = split_documents([item["document"] for item in usable], args.val_ratio, args.seed)

    rendered: dict[tuple[str, int], str] = {}
    stats = Counter()
    samples: dict[str, list[dict]] = {"train": [], "val": []}
    for item in usable:
        document = item["document"]
        split = "val" if document in val_docs else "train"
        key = (document, item["page"])
        rel_path = rendered.get(key)
        if rel_path is None:
            png = render_page(pdf_dir / document, item["page"], args.max_edge)
            if png is None:
                stats["page_out_of_range"] += 1
                continue
            stem = Path(document).stem
            target_dir = images_dir / stem
            target_dir.mkdir(parents=True, exist_ok=True)
            name = f"p{item['page']:04d}.png"
            (target_dir / name).write_bytes(png)
            rel_path = f"images/{stem}/{name}"
            rendered[key] = rel_path
            stats["pages_rendered"] += 1
        question = str(item["row"]["question"]).strip()
        samples[split].append(
            {
                "messages": [
                    {"role": "user", "content": f"<image>{question}"},
                    {"role": "assistant", "content": item["answer"]},
                ],
                "images": [rel_path],
                "id": str(item["row"].get("id", "")),
                "document": document,
                "page": item["page"],
            }
        )
        stats[f"{split}_samples"] += 1

    for split in ("train", "val"):
        path = out_dir / f"{args.dataset_name}_{split}.json"
        path.write_text(
            json.dumps(samples[split], ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

    entry = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    dataset_info = {
        args.dataset_name: {"file_name": f"{args.dataset_name}_train.json", **entry},
        f"{args.dataset_name}_val": {"file_name": f"{args.dataset_name}_val.json", **entry},
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (out_dir / "train_qwen3_5_lora.yaml").write_text(
        f"""### InduDoc AI · 复赛附图 LoRA 训练配置（Qwen3.5-4B · 只训视觉侧）
### 用法：llamafactory-cli train train_qwen3_5_lora.yaml
### 依据 LLaMA-Factory 对 qwen3_5 的注册：
###   vision_model_keys   = visual.pos_embed / visual.patch_embed / visual.blocks
###   projector_keys     = model.visual.merger
###   language_model_keys= language_model / lm_head
###   lora_conflict_keys = patch_embed（永不参与 LoRA）
### 因此 freeze_language_model=true 时，LoRA 只落在 ViT 的线性层上，语言模型与接口层完全冻结。

### model
model_name_or_path: /root/models/Qwen3.5-4B
trust_remote_code: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
freeze_vision_tower: false
freeze_multi_modal_projector: true
freeze_language_model: true

### dataset
dataset: {args.dataset_name}
eval_dataset: {args.dataset_name}_val
template: qwen3_5_nothink
cutoff_len: 1024
max_samples: 20000
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 4
packing: false

### output
output_dir: /root/outputs/qwen3.5-4b-patent-lora
logging_steps: 10
save_steps: 100
eval_strategy: steps
eval_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
### 显存实测依据（CPU 预检）：单图 543x768 -> grid 48x34 -> 408 image tokens
###   样本 input_ids 全长 437~466（cutoff_len=1024 不截断），监督 token 9
###   权重 9.3GB + LoRA(仅 ViT) ~0.1GB + 激活/logits ~2.7GB + ctx ~1GB = ~13GB
###   => 24GB 卡用 bs=4/accum=4 有余量；OOM 时退到 bs=2/accum=8
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true
ddp_timeout: 1800
resume_from_checkpoint: null
""",
        encoding="utf-8",
    )

    token_estimate = (args.max_edge ** 2) / 1024
    answer_lengths = sorted(len(item["answer"]) for item in usable)
    report = {
        "stage": args.stage,
        "split": args.split,
        "source_questions": len(rows),
        "kept_samples": len(samples["train"]) + len(samples["val"]),
        "train_samples": len(samples["train"]),
        "val_samples": len(samples["val"]),
        "skipped_without_page_hint": skipped_no_page,
        "skipped_missing_pdf": skipped_no_doc,
        "page_out_of_range": stats["page_out_of_range"],
        "pages_rendered": stats["pages_rendered"],
        "documents_total": len(train_docs) + len(val_docs),
        "documents_train": len(train_docs),
        "documents_val": len(val_docs),
        "image_max_edge": args.max_edge,
        "image_tokens_estimate": round(token_estimate),
        "answer_length_median": answer_lengths[len(answer_lengths) // 2] if answer_lengths else 0,
        "answer_length_max": answer_lengths[-1] if answer_lengths else 0,
        "dataset_dir": str(out_dir),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.patch_preprocessor:
        model_dir = Path(args.patch_preprocessor).resolve()
        config_path = model_dir / "preprocessor_config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload.setdefault("size", {})["longest_edge"] = args.max_edge * args.max_edge
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["preprocessor_patched"] = str(config_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
