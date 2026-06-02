import cv2
import torch
from mmdet.apis import init_detector, inference_detector
import numpy as np

def visualize_results(img, result, class_names, score_thr=0.3, output_path="output.jpg"):
    """
    可视化检测结果并保存图像。
    
    Args:
        img (np.ndarray): 输入图像。
        result (DetDataSample): 模型的输出结果。
        class_names (list): 类别名称列表。
        score_thr (float): 分数阈值，低于此阈值的检测框将被过滤。
        output_path (str): 输出图像的保存路径。
    """
    # 从 DetDataSample 中提取检测框、类别标签和置信度分数
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()  # 检测框
    labels = pred_instances.labels.cpu().numpy()  # 类别标签
    scores = pred_instances.scores.cpu().numpy()  # 置信度分数

    for bbox, label, score in zip(bboxes, labels, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        class_name = class_names[label]
        label_text = f"{class_name} {score:.2f}"

        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制类别和分数
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")

def main():
    # 配置文件路径
    config_file = "configs/grounding_dino_swin_t.py"
    # 模型权重路径
    checkpoint_file = "tiny.pth"
    # 输入图像路径
    img_path = "90.jpg"
    # 类别名称列表
    class_names = ["cat", "dog", "person"]  # 替换为你的类别名称
    # 文本提示
    text_prompt = ["cat", "dog", "person"]  # 替换为你的文本提示，必须是一个列表或元组
    # 输出图像路径
    output_path = "output.jpg"

    try:
        # 初始化模型
        print("Initializing model...")
        model = init_detector(config_file, checkpoint_file, device="cpu")
        print("Model initialized successfully.")

        # 加载输入图像
        print("Loading input image...")
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image from {img_path}")
        print("Image loaded successfully.")

        # 进行推理
        print("Running inference...")
        result = inference_detector(model, img, text_prompt=text_prompt)
        print("Inference completed successfully.")

        # 打印推理结果
        print("Inference result:", result)

        # 可视化检测结果并保存图像
        print("Visualizing and saving results...")
        visualize_results(img, result, class_names, output_path=output_path)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
