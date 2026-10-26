# 复赛附图 LoRA 训练包（Qwen3.5-4B · 只训视觉侧）

由 `scripts/build_lora_dataset.py` 生成，可重复构建：
`python scripts/build_lora_dataset.py --stage semi_final --split train --max-edge 768 --val-ratio 0.2`

## 内容

| 文件 | 说明 |
| --- | --- |
| `patent_fig_train.json` | 训练样本（ShareGPT：`<image>` + 题干 → 短答案） |
| `patent_fig_val.json` | 验证样本（按 PDF 分组切分，与训练集文档不重叠） |
| `dataset_info.json` | LLaMA-Factory 数据集注册（`patent_fig` / `patent_fig_val`） |
| `images/**.png` | 目标附图页渲染图（最长边 768px，约 576 image tokens） |
| `train_qwen3_5_lora.yaml` | 训练配置（LoRA 只落在 ViT 线性层，LLM 与接口层冻结） |
| `build_report.json` | 构建统计 |

## 统计

- 源题 600 → 保留 442（剔除 158 道无「第N页」提示的题：其证据在整篇正文里，只给附图会诱导模型幻觉）
- 训练 350 / 验证 92，来自 126 个 PDF（训练 101 / 验证 25，**同一 PDF 不跨集**）
- 渲染 127 张唯一页面图，答案长度中位数 7 字
- 单图 token 估算 ≈ 576


## CPU 预检结论（已用本机 llamafactory 0.9.6.dev0 真实加载，未消耗 GPU）

| 指标 | 实测值 |
| --- | --- |
| template / mm_plugin | `qwen3_5_nothink` / `Qwen3VLPlugin` |
| train / eval 加载 | 40 / 40（max_samples 截断；实际 350 / 92） |
| 单样本 input_ids | 454（40 条样本 min/median/max = 437 / 449 / 466） |
| 监督 token | 9（答案部分，prompt 全为 -100） |
| 单图 image token | **408**（543×768 → grid 48×34，patch16 + 2×2 merge） |
| cutoff_len 截断 | 无（1024 未触发） |

据此把训练批大小定为 `bs=4 / accum=4`（等效 bs16），显存预估 ~13GB，24GB 卡有余量；OOM 时退到 `bs=2 / accum=8`。

## 上传到实例（Windows PowerShell，本地执行）

```powershell
scp -P <实例SSH端口> -r "E:\my_Program\多模态\runtime\lora-data" root@<实例IP>:/root/data
scp -P <实例SSH端口> -r "E:\my_Program\多模态项目\Qwen3.5-4B" root@<实例IP>:/root/models
scp -P <实例SSH端口> -r "E:\my_Program\LlamaFactory" root@<实例IP>:/root/
```

> 路径说明：当前工作区是 `E:\my_Program\多模态`，本训练包就在它的 `runtime\lora-data`；基座权重与训练框架**不在本仓库内**——`Qwen3.5-4B` 在 `E:\my_Program\多模态项目\Qwen3.5-4B`，`LlamaFactory` 在 `E:\my_Program\LlamaFactory`（`多模态项目\LlamaFactory` 是另一份副本）。

`ssh -p <端口> root@<实例IP>` 进实例后：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
cd /root/LlamaFactory && pip install -e .
pip install -r requirements/metrics.txt
pip install flash-attn --no-build-isolation        # 可选，装不上自动退 sdpa
mkdir -p /root/outputs
cd /root/LlamaFactory && llamafactory-cli train /root/data/train_qwen3_5_lora.yaml
```

> `Qwen3.5-4B/preprocessor_config.json` 的 `size.longest_edge` 已被改成 589824（768×768），**必须带上这个改动**，否则附图按原生 1440×900 变 1265 tokens/图，24GB 必 OOM。

## 训练后如何验证（回到本地工作台对比）

1. 在同一实例用 vLLM 起服务并挂 LoRA：
   ```bash
   vllm serve /root/models/Qwen3.5-4B --enable-lora \
     --lora-modules patent-lora=/root/outputs/qwen3.5-4b-patent-lora \
     --max-lora-rank 16 --api-key local-dev --port 8000
   ```
   `--max-lora-rank` 必须 ≥ 训练时的 `lora_rank`（本配置为 16，vLLM 默认也正好是 16）；`--api-key` 是因为工作台要求 api_key 非空才认为端点「已配置」，vLLM 不校验时随便给一个非空串即可。
2. 把工作台的 VLM 端点指向它（模型设置页 / `PUT /api/settings/models`），其余三个模型不动：
   - Base URL：`http://<实例IP>:8000/v1`
   - Model：`patent-lora`（即 `--lora-modules` 冒号左边的服务名，不是权重路径）
   - API Key：与 `--api-key` 一致
   - 保存后用「连通性测试」确认，再单跑一题看返回 `model` 字段是否回显 `patent-lora`。
3. 跑同一套回归集对比 LoRA 前后（按 PDF 分组、无泄漏）：
   ```powershell
   python scripts\evaluate_regression.py --size 60 --seed 20261026 --arms baseline,rag
   ```
   重点看官方综合分（0.5×BERTScore + 0.25×ROUGE-1 + 0.25×EM）与 Recall@K/MRR 的变化，而不是只看训练 loss。

> ⚠️ 本配置 `freeze_language_model: true`，LoRA 只落在 ViT 线性层。**上线前先验证这个 ViT LoRA 在 vLLM 里是否真的生效**：同一张附图、同一 prompt，分别用 `Qwen3.5-4B` 和 `patent-lora` 各跑一次，确认输出确实不同（vLLM 的 LoRA 主要作用于语言层，视觉塔是否吃到 adapter 取决于版本对该模型的支持）。若两者输出一致，说明 adapter 没生效，此时对比结果不能算 LoRA 收益。
