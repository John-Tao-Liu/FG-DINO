
### 3. 细粒度伪标签生成

```bash
# 生成单区域语义概念包标注
python tools/generate_region_contains.py \
  --input_jsonl data/raw_annotations.jsonl \
  --output_jsonl data/fine_grained_annotations.jsonl \
  --model Qwen2-VL-7B-Instruct \
  --batch_size 8
```

## 快速开始

### 1. 模型下载

```bash
# 创建模型存储目录
mkdir -p huggingface

# 下载BERT基础模型
huggingface-cli download google-bert/bert-base-uncased --local-dir ./huggingface/bert-base-uncased

# 下载SigLIP视觉模型
modelscope download fireicewolf/siglip-so400m-patch14-384 --local-dir ./huggingface/siglip-so400m-patch14-384

# 下载FG-DINO预训练权重
modelscope download your-username/fg-dino-pretrain --local-dir ./huggingface/fg_dino_pretrain
```

### 2. 训练

```bash
# 单GPU训练（推荐）
bash dist_train.sh configs/grounding_dino_fine_grained.py 1 --amp --launcher none --work-dir ./work_dirs/fg_dino_fine_grained

# 多GPU分布式训练
bash dist_train.sh configs/grounding_dino_fine_grained.py 4 --amp --work-dir ./work_dirs/fg_dino_fine_grained

# 断点续训
bash dist_train.sh configs/grounding_dino_fine_grained.py 1 --amp --launcher none --resume True
```

### 3. 评估

```bash
# LVIS数据集评估
bash dist_test.sh configs/val/grounding_dino_swin-t_lvis.py work_dirs/fg_dino_fine_grained/latest.pth 1

# COCO-O数据集评估
bash dist_test.sh configs/val/grounding_dino_swin-t_coco-o.py work_dirs/fg_dino_fine_grained/latest.pth 1

# RefCOCO数据集评估
bash dist_test.sh configs/val/grounding_dino_swin-t_refexp.py work_dirs/fg_dino_fine_grained/latest.pth 1
```

## 实验结果

### 1. LVIS v1 验证集结果

| 方法 | AP | AP50 | AP75 | APr | APc | APf |
|------|----|------|------|-----|-----|-----|
| Grounding DINO | 0.412 | 0.523 | 0.435 | 0.321 | 0.365 | 0.478 |
| **FG-DINO (Ours)** | **0.452** | **0.565** | **0.479** | **0.386** | **0.398** | **0.512** |

### 2. COCO-O OOD泛化结果

| 方法 | 平均AP | 卡通 | 手绘 | 绘画 | 草图 | 纹身 | 天气 |
|------|--------|------|------|------|------|------|------|
| Grounding DINO | 0.358 | 0.378 | 0.299 | 0.424 | 0.327 | 0.274 | 0.445 |
| **FG-DINO (Ours)** | **0.368** | **0.385** | **0.312** | **0.436** | **0.335** | **0.283** | **0.457** |

### 3. RefCOCO指代表达结果

| 数据集 | 分割 | FG-DINO (P@1) |
|--------|------|--------------|
| RefCOCO | val | 70.8 |
| | testA | 75.7 |
| | testB | 68.0 |
| RefCOCO+ | val | 56.5 |
| | testA | 63.3 |
| | testB | 51.3 |
| RefCOCOg | val | 72.8 |
| | test | 73.7 |

## 项目结构

