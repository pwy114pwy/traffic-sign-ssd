"""
可视化数据集中的边界框
用于验证边界框转换是否正确
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset import GTSRBDataset, get_transform

def visualize_samples(num_samples=5, save_dir="visualizations"):
    """可视化数据集样本"""
    print("=" * 60)
    print("可视化数据集样本")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集 (不使用transform,直接查看原始图像)
    dataset = GTSRBDataset(
        root_dir="datasets/GTSRB",
        set_name="train",
        transform=None  # 先不使用transform
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"将可视化前 {num_samples} 个样本\n")
    
    for i in range(min(num_samples, len(dataset))):
        # 获取样本
        image, target = dataset[i]
        boxes = target["boxes"]
        labels = target["labels"]
        
        print(f"样本 {i+1}:")
        print(f"  图像尺寸: {image.size}")
        print(f"  边界框数量: {len(boxes)}")
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图: 原始图像 + 归一化边界框
        ax1.imshow(image)
        ax1.set_title(f"样本 {i+1}: 归一化边界框 [0, 1]")
        ax1.axis('off')
        
        width, height = image.size
        
        for j, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.tolist()
            
            print(f"  边界框 {j+1}: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}], 类别: {label.item()}")
            
            # 验证坐标范围
            if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                print(f"    ⚠ 警告: 坐标超出[0,1]范围!")
            
            if not (x2 > x1 and y2 > y1):
                print(f"    ⚠ 警告: 边界框无效!")
            
            # 转换为像素坐标用于绘制
            px1, py1 = x1 * width, y1 * height
            px2, py2 = x2 * width, y2 * height
            
            # 绘制边界框
            rect = patches.Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(rect)
            
            # 添加标签
            ax1.text(px1, py1 - 5, f"Class {label.item()}", 
                    color='red', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 右图: resize后的图像 (300x300)
        transform = get_transform(train=False)
        transformed_image = transform(image)
        
        # 反归一化用于显示
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = transformed_image.permute(1, 2, 0).numpy()
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        
        ax2.imshow(img_display)
        ax2.set_title(f"Resize到300x300后 (边界框坐标不变)")
        ax2.axis('off')
        
        # 在resize后的图像上绘制边界框
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            
            # 归一化坐标可以直接用于任何尺寸的图像
            px1, py1 = x1 * 300, y1 * 300
            px2, py2 = x2 * 300, y2 * 300
            
            rect = patches.Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax2.add_patch(rect)
            
            ax2.text(px1, py1 - 5, f"Class {label.item()}", 
                    color='green', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 保存图像
        save_path = os.path.join(save_dir, f"sample_{i+1}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 已保存到: {save_path}\n")
    
    print("=" * 60)
    print(f"可视化完成! 图像已保存到 {save_dir}/ 目录")
    print("=" * 60)
    print("\n请检查可视化结果:")
    print("1. 边界框是否正确框住交通标志")
    print("2. 边界框坐标是否在[0, 1]范围内")
    print("3. resize后边界框位置是否仍然正确")

if __name__ == "__main__":
    visualize_samples(num_samples=5)
