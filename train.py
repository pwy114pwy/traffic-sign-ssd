import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import time

from models.ssd import build_ssd
from models.utils import AnchorBoxes
from datasets.dataset import GTSRBDataset, get_transform

class SSDLoss(nn.Module):
    """SSD损失函数"""
    def __init__(self, num_classes=43, alpha=1.0, neg_pos_ratio=3):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predicted_locations, predicted_scores, anchors, targets):
        """
        计算SSD损失
        Args:
            predicted_locations: 预测的位置偏移 (batch_size, num_anchors, 4)
            predicted_scores: 预测的分类分数 (batch_size, num_anchors, num_classes)
            anchors: 锚框 (num_anchors, 4)
            targets: 真实标注 (list of dicts)
        """
        batch_size = predicted_locations.size(0)
        num_anchors = anchors.size(0)
        num_classes = predicted_scores.size(2)
        
        # 初始化损失
        loc_loss = 0.0
        conf_loss = 0.0
        
        for i in range(batch_size):
            # 获取当前样本的预测
            predicted_loc = predicted_locations[i]  # (num_anchors, 4)
            predicted_score = predicted_scores[i]  # (num_anchors, num_classes)
            
            # 获取当前样本的真实标注
            target_boxes = targets[i]["boxes"]  # (num_targets, 4)
            target_labels = targets[i]["labels"]  # (num_targets,)
            
            if target_boxes.numel() == 0:
                # 如果没有标注，只计算背景类的损失
                conf_loss += self.cross_entropy_loss(predicted_score, torch.zeros(num_anchors, dtype=torch.long)).sum()
                continue
            
            # 匹配锚框和真实标注
            overlaps = self._compute_iou(anchors, target_boxes)  # (num_anchors, num_targets)
            best_target_per_anchor, best_target_index = overlaps.max(dim=1)  # (num_anchors,)
            best_anchor_per_target, best_anchor_index = overlaps.max(dim=0)  # (num_targets,)
            
            # 确保每个真实标注至少匹配一个锚框
            for t in range(best_anchor_index.size(0)):
                best_target_index[best_anchor_index[t]] = t
                best_target_per_anchor[best_anchor_index[t]] = 1.0
            
            # 匹配阈值
            positive_mask = best_target_per_anchor > 0.5
            
            # 获取正样本和对应的真实标注
            positive_indices = torch.where(positive_mask)[0]
            negative_indices = torch.where(~positive_mask)[0]
            
            # 位置损失
            if positive_indices.numel() > 0:
                target_locations = self._encode_boxes(anchors[positive_indices], target_boxes[best_target_index[positive_indices]])
                loc_loss += self.smooth_l1_loss(predicted_loc[positive_indices], target_locations)
            
            # 分类损失
            target_labels_for_anchors = torch.zeros(num_anchors, dtype=torch.long, device=predicted_score.device)
            target_labels_for_anchors[positive_indices] = target_labels[best_target_index[positive_indices]] + 1  # +1 因为背景类是0
            
            # 硬负样本挖掘
            conf_scores = self.cross_entropy_loss(predicted_score, target_labels_for_anchors)
            
            # 正样本损失
            pos_conf_loss = conf_scores[positive_indices].sum()
            
            # 负样本损失 (最多是正样本的neg_pos_ratio倍)
            num_pos = positive_indices.numel()
            num_neg = min(num_pos * self.neg_pos_ratio, negative_indices.numel())
            
            if num_neg > 0:
                neg_conf_loss = conf_scores[negative_indices].topk(num_neg)[0].sum()
            else:
                neg_conf_loss = 0.0
            
            conf_loss += pos_conf_loss + neg_conf_loss
        
        # 平均损失
        loc_loss /= batch_size
        conf_loss /= batch_size
        
        return loc_loss + self.alpha * conf_loss
    
    def _compute_iou(self, anchors, boxes):
        """计算锚框和真实框的IoU"""
        anchors = anchors.unsqueeze(1)
        boxes = boxes.unsqueeze(0)
        
        # 计算交集
        x1 = torch.max(anchors[:, :, 0], boxes[:, :, 0])
        y1 = torch.max(anchors[:, :, 1], boxes[:, :, 1])
        x2 = torch.min(anchors[:, :, 2], boxes[:, :, 2])
        y2 = torch.min(anchors[:, :, 3], boxes[:, :, 3])
        
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        area_inter = w * h
        
        # 计算并集
        area_anchor = (anchors[:, :, 2] - anchors[:, :, 0]) * (anchors[:, :, 3] - anchors[:, :, 1])
        area_box = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
        area_union = area_anchor + area_box - area_inter
        
        # IoU
        iou = area_inter / area_union
        
        return iou
    
    def _encode_boxes(self, anchors, boxes):
        """编码真实框为位置偏移"""
        anchors_width = anchors[:, 2] - anchors[:, 0]
        anchors_height = anchors[:, 3] - anchors[:, 1]
        anchors_ctr_x = anchors[:, 0] + 0.5 * anchors_width
        anchors_ctr_y = anchors[:, 1] + 0.5 * anchors_height
        
        boxes_width = boxes[:, 2] - boxes[:, 0]
        boxes_height = boxes[:, 3] - boxes[:, 1]
        boxes_ctr_x = boxes[:, 0] + 0.5 * boxes_width
        boxes_ctr_y = boxes[:, 1] + 0.5 * boxes_height
        
        # 编码
        dx = (boxes_ctr_x - anchors_ctr_x) / anchors_width
        dy = (boxes_ctr_y - anchors_ctr_y) / anchors_height
        dw = torch.log(boxes_width / anchors_width)
        dh = torch.log(boxes_height / anchors_height)
        
        return torch.stack([dx, dy, dw, dh], dim=1)

def train(args):
    """训练SSD模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型（包括背景类，所以总类别数是43+1=44）
    model = build_ssd(num_classes=args.num_classes + 1)
    model.to(device)
    
    # 创建锚框生成器
    anchor_boxes = AnchorBoxes()
    anchors = anchor_boxes(None)
    anchors = anchors.to(device)
    
    # 创建损失函数（包括背景类）
    criterion = SSDLoss(num_classes=args.num_classes + 1)
    
    # 创建优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # 梯度裁剪参数
    max_grad_norm = 2.0
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # 创建数据集
    train_dataset = GTSRBDataset(
        root_dir=args.dataset_dir,
        set_name="train",
        transform=get_transform(train=True)
    )
    
    val_dataset = GTSRBDataset(
        root_dir=args.dataset_dir,
        set_name="val",
        transform=get_transform(train=False)
    )
    
    # 定义collate函数
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # 开始训练
    best_val_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # 转换数据到设备
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            predicted_scores, predicted_locations = model(images)
            
            # 计算损失
            loss = criterion(predicted_locations, predicted_scores, anchors, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印训练进度
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                # 转换数据到设备
                images = torch.stack(images).to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # 前向传播
                predicted_scores, predicted_locations = model(images)
                
                # 计算损失
                loss = criterion(predicted_locations, predicted_scores, anchors, targets)
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_ssd_model.pth"))
            print(f"Best model saved with val loss: {best_val_loss:.4f}")
        
        # 保存每个epoch的模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"ssd_model_epoch_{epoch+1}.pth"))
        
        # 打印训练结果
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {end_time - start_time:.2f}s")
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSD model on GTSRB dataset")
    
    # 模型参数
    parser.add_argument("--num_classes", type=int, default=43, help="Number of classes (including background)")
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--step_size", type=int, default=30, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler")
    
    # 数据参数
    parser.add_argument("--dataset_dir", type=str, default="datasets/GTSRB", help="Dataset directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # 保存和日志
    parser.add_argument("--save_dir", type=str, default="models/saved", help="Directory to save models")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 开始训练
    train(args)