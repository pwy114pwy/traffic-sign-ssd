import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

from models.ssd import build_ssd
from models.utils import AnchorBoxes, detect
from datasets.dataset import get_transform

def load_model(model_path, num_classes=43, device='cuda'):
    """加载训练好的SSD模型"""
    model = build_ssd(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform=None):
    """预处理输入图像"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    if transform is not None:
        image = transform(image)
    
    return image, original_size

def visualize_detections(image, detections, original_size, class_names=None, threshold=0.5):
    """可视化检测结果"""
    # 如果没有提供类别名称，使用默认名称
    if class_names is None:
        class_names = [f"class_{i}" for i in range(43)]
    
    # 将图像转换回PIL格式以便绘制
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # 调整图像大小到原始尺寸
    image = image.resize(original_size)
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # 绘制检测结果
    for detection in detections:
        if detection["score"] < threshold:
            continue
        
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]
        
        # 转换归一化坐标到图像坐标
        x1 = int(box[0] * original_size[0])
        y1 = int(box[1] * original_size[1])
        x2 = int(box[2] * original_size[0])
        y2 = int(box[3] * original_size[1])
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        
        # 绘制标签和分数
        class_name = class_names[label - 1]  # 减1是因为背景类是0
        text = f"{class_name}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 15), text, font=font)
        draw.rectangle(text_bbox, fill=(0, 255, 0))
        draw.text((x1, y1 - 15), text, fill=(0, 0, 0), font=font)
    
    return image

def main(args):
    """主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, num_classes=args.num_classes, device=device)
    
    # 创建锚框生成器
    anchor_boxes = AnchorBoxes()
    anchors = anchor_boxes(None)
    anchors = anchors.to(device)
    
    # 获取图像变换
    transform = get_transform(train=False)
    
    # 处理输入图像
    print(f"Processing image {args.image_path}...")
    image, original_size = preprocess_image(args.image_path, transform)
    
    # 转换图像到设备
    image = image.unsqueeze(0).to(device)
    
    # 执行推理
    with torch.no_grad():
        predicted_scores, predicted_locations = model(image)
        detections = detect(predicted_locations, predicted_scores, anchors, 
                           min_score=args.conf_threshold, 
                           max_overlap=args.nms_threshold, 
                           top_k=args.top_k)
    
    # 可视化检测结果
    print(f"Found {len([d for d in detections[0] if d['score'] >= args.conf_threshold])} detections")
    result_image = visualize_detections(image[0], detections[0], original_size, threshold=args.conf_threshold)
    
    # 保存结果
    if args.output_path:
        result_image.save(args.output_path)
        print(f"Result saved to {args.output_path}")
    
    # 显示结果
    if args.show:
        result_image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSD model inference")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--num_classes", type=int, default=43, help="Number of classes")
    
    # 推理参数
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--top_k", type=int, default=200, help="Top K detections")
    
    # 输出参数
    parser.add_argument("--output_path", type=str, help="Path to save output image")
    parser.add_argument("--show", action="store_true", help="Show output image")
    
    # 设备参数
    parser.add_argument("--use_cuda", action="store_true", default=True, help="Use CUDA if available")
    
    args = parser.parse_args()
    
    main(args)