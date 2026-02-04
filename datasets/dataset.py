import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class GTSRBDataset(Dataset):
    """GTSRB数据集加载器"""
    def __init__(self, root_dir, set_name="train", transform=None):
        """
        Args:
            root_dir: 数据集根目录
            set_name: 数据集名称 (train/val)
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        
        # 加载COCO格式的标注文件
        annotation_path = os.path.join(root_dir, "annotations", f"{set_name}.json")
        with open(annotation_path, "r") as f:
            self.coco_data = json.load(f)
        
        # 构建图像ID到标注的映射
        self.image_id_to_annotations = {}
        for annotation in self.coco_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(annotation)
        
        # 构建图像ID到图像信息的映射
        self.image_id_to_info = {}
        for image in self.coco_data["images"]:
            self.image_id_to_info[image["id"]] = image
        
        # 图像列表
        self.images = self.coco_data["images"]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        # 获取图像信息
        image_info = self.images[idx]
        image_id = image_info["id"]
        image_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]
        
        # 加载图像
        image_path = os.path.join(self.root_dir, "images", self.set_name, image_name)
        image = Image.open(image_path).convert("RGB")
        
        # 获取标注
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        # 处理标注
        boxes = []
        labels = []
        areas = []
        is_crowds = []
        
        for annotation in annotations:
            # COCO格式: [x, y, width, height] (绝对像素坐标)
            # 需要转换为: [x1, y1, x2, y2] (归一化坐标 [0, 1])
            x, y, w, h = annotation["bbox"]
            
            # 转换为 [x1, y1, x2, y2] 并归一化
            x1 = x / width
            y1 = y / height
            x2 = (x + w) / width
            y2 = (y + h) / height
            
            # 确保坐标在有效范围内 [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            boxes.append([x1, y1, x2, y2])
            labels.append(annotation["category_id"])
            areas.append(annotation["area"])
            is_crowds.append(annotation["iscrowd"])
        
        # 转换为numpy数组
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        areas = np.array(areas, dtype=np.float32) if areas else np.zeros((0,), dtype=np.float32)
        is_crowds = np.array(is_crowds, dtype=np.int64) if is_crowds else np.zeros((0,), dtype=np.int64)
        
        # 转换为PyTorch张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])
        area = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(is_crowds, dtype=torch.int64)
        
        # 创建目标字典
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        
        # 应用图像变换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target

def get_transform(train=True, size=(300, 300)):
    """获取图像变换
    
    注意: 暂时移除了RandomHorizontalFlip和ColorJitter等数据增强,
    因为这些变换需要同步调整边界框坐标。
    修复边界框问题并验证后,可以重新添加支持边界框变换的数据增强。
    """
    transforms_list = [
        transforms.Resize(size),  # 调整所有图像到相同尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # TODO: 后续可以添加支持边界框变换的数据增强
    # if train:
    #     transforms_list.insert(1, transforms.ColorJitter(brightness=0.2, contrast=0.2))
    
    return transforms.Compose(transforms_list)