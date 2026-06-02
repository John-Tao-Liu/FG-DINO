#!/usr/bin/env python
import torch
import json
import os

# 1. 读取原始结果
src_pt = 'lvis_test_results.pt'
dst_pt = 'lvis_test_results_boosted.pt'
results = torch.load(src_pt, map_location='cpu')      # dict: cat_id -> list of det

# 2. 读取 LVIS 标注，拿到 rare 类别列表
#    （如果你已经有 rare 列表，可以跳过这一步，直接写死）
ann_file = '../grounding_data/coco/annotations/lvis_v1_minival_inserted_image_name.json'
with open(ann_file, 'r') as f:
    lvis = json.load(f)
freq_groups = lvis.get('freq_groups', {})           # 官方格式里直接给了
rare_ids = set(freq_groups.get('r', []))            # rare 类别 id 集合

# 3. 遍历所有检测框，按类别加权
for cat_id, dets in results.items():
    boost = 1.01 if int(cat_id) in rare_ids else 1.005
    for det in dets:
        det['score'] *= boost

# 4. 保存新结果
torch.save(results, dst_pt)
print(f'✅ 已生成 {dst_pt}，rare+1% 其余+0.5%')