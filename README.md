# Seg-Zero 自有数据集测试

本仓库封装 Seg-Zero 单目标推理，面向自有数据的可重复测试。输入一张图片和一个文本问题，输出一个二值目标掩码。不包含模型权重或私有数据。

## 1. 下载这个仓库

```bash
git clone https://github.com/yiyipupu/seg-zero-benchmark.git
cd seg-zero-benchmark
```

本仓库已直接包含 `upstream/` 下的官方完整 Seg-Zero 源码（含训练、评估、推理、verl 和示例图片），无需再克隆官方仓库或初始化 submodule。根目录的 `infer.py` 与 `scripts/` 是为自有数据测试提供的适配入口。模型权重通过下面的命令另行下载。

目录结构：

```text
README.md                # 从安装到评估的使用指南
infer.py                 # 4090 单卡推理入口
scripts/                 # 下载、转换、批量推理、评估
examples/test.jsonl       # 输入格式示例
upstream/                # 固定版本的官方完整源码与 LICENSE
```

## 来源与原始数据集

- 论文：[Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/abs/2503.06520)。
- [官方代码](https://github.com/JIA-Lab-research/Seg-Zero)，单目标版本提交 `77f9ea5887ec7e6abf398ed3cb483c65631c82b7`。当前官方 main 已转向多目标 VisionReasoner，请勿混用。
- 原论文主要使用作者整理的 **RefCOCOg 9K 子集**进行 GRPO 强化学习训练；官方另提供 2K 子集。目标标注用于训练奖励，不需要人工推理链标注。
- 推理分割评估使用 **ReasonSeg val/test**。官方说明不同 benchmark 最优结果可能来自不同 checkpoint，不能将默认发布模型视为所有论文最优结果的同一 checkpoint。
- 数据：[RefCOCOg-9K](https://huggingface.co/datasets/Ricky06662/refCOCOg_9k_840)、[2K](https://huggingface.co/datasets/Ricky06662/refCOCOg_2k_840)、[ReasonSeg test](https://huggingface.co/datasets/Ricky06662/ReasonSeg_test)、[val](https://huggingface.co/datasets/Ricky06662/ReasonSeg_val)。测试自有数据无需下载这些原始数据集。
- 推理模型：`Ricky06662/Seg-Zero-7B` + `facebook/sam2-hiera-large`，流程为 Qwen2.5-VL 生成框和两个提示点，再由 SAM2 输出掩码。

## 2. 配环境（Linux / RTX 4090 24GB）

```bash
export SEG_ROOT=/data/yukun/Seg
mkdir -p "$SEG_ROOT"/{env,cache,checkpoints,datasets,outputs,weights}
export CONDA_PKGS_DIRS=$SEG_ROOT/cache/conda_pkgs
export PIP_CACHE_DIR=$SEG_ROOT/cache/pip
export HF_HOME=$SEG_ROOT/cache/huggingface
export TORCH_HOME=$SEG_ROOT/cache/torch
export TMPDIR=$SEG_ROOT/cache/tmp
mkdir -p "$TMPDIR"
conda create -p "$SEG_ROOT/env/segzero" python=3.10 pip -y
conda activate "$SEG_ROOT/env/segzero"
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python scripts/download_weights.py
```

每次开启新终端，在仓库目录执行：

```bash
source scripts/env.sh
conda activate "$SEG_ROOT/env/segzero"
```

如 `conda` 尚未安装，先安装 Miniconda；如果提示需要初始化，执行 `source ~/miniconda3/etc/profile.d/conda.sh`（按实际安装位置调整）。权重下载中断后重新运行 `python scripts/download_weights.py` 即可；仅在官方站不可达时尝试 `export HF_ENDPOINT=https://hf-mirror.com`。

## 3. 先跑仓库自带图片

```bash
source scripts/env.sh
conda activate "$SEG_ROOT/env/segzero"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer.py \
  --reasoning_model_path "$SEG_ROOT/weights/Seg-Zero-7B" \
  --segmentation_model_path "$SEG_ROOT/weights/sam2-hiera-large/sam2_hiera_large.pt" \
  --image_path upstream/assets/test_image.png \
  --text "the unusual object in the image" \
  --output_path "$SEG_ROOT/outputs/demo.png"
```

成功标志：命令正常退出，并生成 `outputs/demo.png` 和 `outputs/demo.png.mask.png`。示例问题对应西瓜轮子的自行车。选择其他物理 GPU 时修改 `CUDA_VISIBLE_DEVICES`，代码中的 `cuda:0` 表示所选的第一张可见卡。

CUDA 12.1 wheel 需要兼容 NVIDIA 驱动；无需额外安装系统 CUDA Toolkit。使用 RTX 4090 支持的 BF16，模型直接加载到单张 GPU，不再自动分卡或设置每卡显存上限。注意力保留 PyTorch SDPA，便于快速安装，无需编译 FlashAttention，也无需 vLLM/Ray。单张 4090 配置尚未实机验证；若显存不足，请先确认该卡没有其他进程占用。

## 4. 整理自有数据

这是本封装定义的接口，不是要求伪造 RefCOCOg 或下载其数据。将原始标注转换为以下 JSONL，每行一个“图片＋问题＋目标掩码”样本：

```text
/data/yukun/Seg/datasets/custom/
  test.jsonl
  images/001.jpg
  masks/001.png
```

```json
{"id":"sample_001","image":"images/001.jpg","text":"Which object can protect people from rain?","mask":"masks/001.png"}
```

- `id`：唯一，只含英文字母、数字、下划线、短横线。
- `image` / `mask`：相对数据根目录路径。
- `text`：目标描述或推理问题，建议英文；比较模型时保持原始问题一致。
- `mask`：与原图同宽高的灰度 PNG，背景 0、目标 255；评估将所有非零像素视为目标。若原始 mask 是实例/类别 ID 图，必须先提取当前目标并转为二值图，不能直接把全部非零类别作为目标。
- 一张图有多个问题时写多行，使用不同 id 和对应 mask。单目标版本不输出多实例列表。
- 纯推理允许省略 mask；评估必须有 mask。暂不支持 ignore 像素，须在转换阶段明确处理。
- 当前尚未收到团队原始标注样例，因此未实现 COCO polygon/RLE 等格式的专属转换器。

### 可选：从 JSON 列表转换

如果当前已有图片与单通道 mask，可整理一个 JSON 数组 `annotations.json`：

```json
[
  {"id":"sample_001","image":"images/001.jpg","text":"Which object can protect people from rain?","mask":"labels/001.png","target_id":3}
]
```

`target_id: 3` 表示从类别/实例 ID mask 中提取像素值为 3 的目标；已有二值 mask 时省略此字段。该工具不读取 COCO polygon/RLE，后者需先栅格化成与原图同尺寸的单通道 mask。

```bash
python scripts/convert_manifest.py \
  --input /data/yukun/Seg/datasets/custom/annotations.json \
  --root /data/yukun/Seg/datasets/custom \
  --output /data/yukun/Seg/datasets/custom/test.jsonl
```

转换工具生成二值 PNG 和 JSONL；JSONL 内使用绝对路径，迁移机器时重新转换。图片尺寸与 mask 不一致会报错，请修正标注后重试。

## 5. 批量推理与评估

```bash
export SEG_ROOT=/data/yukun/Seg
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
python scripts/batch_infer.py \
  --manifest "$SEG_ROOT/datasets/custom/test.jsonl" \
  --root "$SEG_ROOT/datasets/custom" \
  --output "$SEG_ROOT/outputs/custom"
python scripts/evaluate.py \
  --manifest "$SEG_ROOT/datasets/custom/test.jsonl" \
  --root "$SEG_ROOT/datasets/custom" \
  --predictions "$SEG_ROOT/outputs/custom"
```

输出：每例 `<id>.png` 可视化、`<id>.png.mask.png` 二值掩码、`<id>.log` 推理日志，及追加写入的 `status.jsonl`。每次实验使用新输出目录，以免旧预测混入评估。当前批量入口逐样本重载模型，适合正确性基线，速度不代表模型最佳吞吐。

指标为逐样本 IoU 的均值 mean_IoU 与累计交并比 global_IoU（0–1）。缺失预测按空掩码计分并统计数量，空真值与空预测的 IoU 定义为 1。不等同于完整复刻官方评估协议。正式比较时固定数据划分、问题文本、空掩码规则、分辨率及模型 checkpoint，并报告失败样本数。

## 修改与验证状态

infer.py 改编自上述官方提交：BF16、SDPA、单卡加载、本地 SAM2 加载、无界面保存与额外二值掩码。保留官方 840×840 输入缩放与坐标映射。先前 FP16 多卡版本已在 RTX 2080 Ti 上跑通官方示例；本仓库当前 BF16 单卡版本、4090、自有数据集与批量评估尚未实测。权重与数据均在外部目录，不提交 Git。

源代码保留 Apache-2.0 LICENSE；模型和数据使用条件请分别查看其发布页。
