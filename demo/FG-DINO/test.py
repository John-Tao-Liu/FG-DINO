import torch
import time
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.engine import build_dataloader, build_dataset
from mmdet.evaluation import LVISFixedAPMetric
from mmengine import Config
from mmengine.runner import Runner

def main():
    # 配置文件路径
    config_file = "configs/grounding_dino_swin_t.py"
    # 模型权重路径
    checkpoint_file = "tiny.pth"

    # 加载配置文件
    cfg = Config.fromfile(config_file)
    cfg.load_from = checkpoint_file

    # 初始化模型
    print("Initializing model...")
    model = init_detector(config_file, checkpoint_file, device="cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model initialized successfully.")

    # 构建 LVIS 数据集
    print("Building LVIS dataset...")
    dataset = build_dataset(cfg.test_dataloader.dataset)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )
    print("Dataset and data loader built successfully.")

    # 初始化评估器
    evaluator = LVISFixedAPMetric(
        ann_file='../grounding_data/coco/annotations/lvis_v1_minival_inserted_image_name.json'
    )

    # 运行推理并评估
    print("Running inference and evaluation...")
    total_samples = len(data_loader.dataset)  # 测试集的总样本数
    start_time = time.time()  # 开始时间

    # 使用 tqdm 显示进度条
    for i, data in enumerate(tqdm(data_loader, desc="Processing", unit="sample")):
        result = inference_detector(model, data['img'][0])
        evaluator.process(data, result)

        # 计算剩余时间
        elapsed_time = time.time() - start_time
        avg_time_per_sample = elapsed_time / (i + 1)
        remaining_time = avg_time_per_sample * (total_samples - (i + 1))

        # 输出详细信息
        tqdm.write(f"Processed {i + 1}/{total_samples} samples | "
                   f"Elapsed: {elapsed_time:.2f}s | "
                   f"Remaining: {remaining_time:.2f}s")

    # 计算并打印 mAP 值
    metrics = evaluator.evaluate(size=len(dataset))
    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
