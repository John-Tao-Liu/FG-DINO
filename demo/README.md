
# FG-DINO: Open-Vocabulary Object Detection with Vision-Text Fine-Grained Alignment

![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1+-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)
![Benchmark](https://img.shields.io/badge/Benchmark-LVIS%20|%20COCO--O%20|%20RefCOCO-red.svg)


---

## 📢 项目简介

FG-DINO 是一款**面向细粒度开放场景的目标检测框架**，以 Grounding DINO 为基线，围绕**视觉-文本细粒度跨模态对齐**构建全链路优化方案，解决传统开放词汇检测的四大痛点：

1. 训练数据标注粗粒度、异构标签割裂
2. 图文语义错位与伪标签噪声严重
3. 视觉底层特征与文本高层语义空间错位
4. 固定 Prompt 导致训练-推论语义断层

框架创新覆盖**数据构建、模型架构、训练推理**全流程，在 LVIS、OV-COCO、COCO-O、RefCOCO 等基准数据集上达到领先性能。

---

## 🧱 实验环境

### 环境配置（一键部署）

```bash
# 1. 创建虚拟环境
conda deactivate
conda create -n mmdet_env python=3.10 -y
conda activate mmdet_env

# 2. 安装 PyTorch
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装核心依赖
pip install numpy==1.22.2 transformers==4.37.2 scipy
pip install timm deepspeed pycocotools lvis jsonlines fairscale nltk peft wandb
pip install terminaltables shapely

# 4. 安装 MM 系列库
pip install -U openmim
mim install mmcv==2.2.0
mim install mmengine==0.10.5
```

### 环境切换

```bash
# 切回 base
conda deactivate
# 切回项目环境
conda activate mmdet_env
```

---

## 📂 数据准备

### 数据集目录结构

```
autodl-fs/
└── demo/
    ├── FG-DINO/                # 项目代码
    ├── huggingface/            # 预训练模型
    │   ├── bert-base-uncased
    │   ├── siglip-so400m-patch14-384
    │   ├── qwen2.5-vl
    │   └── mm_grounding_dino/
    └── grounding_data/         # 数据集根目录
        ├── coco/               # COCO2017 + LVIS 标注
        ├── v3det/              # V3Det 细粒度数据集
        ├── flickr30k_entities/ # GoldG 数据源
        ├── gqa/
        ├── llava_cap/          # LCS-558k 数据集
        └── ood_coco/           # COCO-O 域外数据集
```

---

## 🚀 快速开始

### 1. 模型训练
#### 基础训练（基线模型）
```bash
bash dist_train.sh configs/grounding_dino_swin_t.py 1 --amp
```

#### 微调训练（指定工作目录）
```bash
bash dist_train.sh configs/grounding_dino_test.py 1 --amp --launcher none --work-dir ./work_dirs/grounding_dino_t_finetune
```

#### 断点续训（自动加载最新权重）
```bash
bash dist_train.sh configs/grounding_dino_swin_t.py 1 --amp --launcher none --resume True
```


### 2. 模型测试
#### LVIS 数据集测试（核心基准）
```bash
bash dist_test.sh configs/val/grounding_dino_swin-t_lvis_val.py t.pth 1
```

#### COCO-O 域外泛化测试
```bash
bash dist_test.sh configs/val/grounding_dino_swin-t_coco-o.py tiny.pth 1
```

#### RefCOCO 指代理解测试
```bash
bash dist_test.sh configs/val/grounding_dino_swin-t_refexp.py tiny.pth 1
```

### 3. 独立评估脚本
```bash
# LVIS 分数校准与独立评估
python eval_lvis_independent.py

# 快速打印 LVIS 指标
python eval_lvis.py
```



---

## 🎯 Demo 演示
### 依赖安装
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
```
> 说明：短语定位与指代性表达式理解需提前下载 NLTK 工具包；若推理时无需加载 LLM，可修改配置文件设置 `llm=None`

### 1. 开放词汇目标检测
```bash
python image_demo.py images/demo.jpeg \
  configs/grounding_dino_swin_t.py --weight tiny.pth \
  --text 'apple .' -c --pred-score-thr 0.4
```

### 2. 短语定位
```bash
python image_demo.py images/demo.jpeg \
  configs/grounding_dino_swin_t.py --weight tiny.pth \
  --text 'There are many apples here.' --pred-score-thr 0.35
```

### 3. 指代性表达式理解
```bash
python image_demo.py images/demo.jpeg \
  configs/grounding_dino_swin_t.py --weight tiny.pth \
  --text 'red apple.' --tokens-positive -1 --pred-score-thr 0.4
```

---

## 📊 实验结果
### 1. LVIS minival 数据集

| 模型 | AP | APr(稀有) | APc(常见) | APf(频繁) |
|------|----|-----------|-----------|------------|
| Grounding DINO | 41.0 | 33.4 | 36.7 | 48.8 |
| FG-DINO-Swin-T | **45.8** | **39.9** | **40.6** | **51.4** |


---
