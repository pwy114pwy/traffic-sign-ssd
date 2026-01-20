import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_anchors(size, aspect_ratios):
    """生成锚框"""
    anchors = []
    for aspect_ratio in aspect_ratios:
        h = size * np.sqrt(aspect_ratio)
        w = size / np.sqrt(aspect_ratio)
        anchors.append([-w/2, -h/2, w/2, h/2])
    return torch.tensor(anchors, dtype=torch.float32)

class AnchorBoxes(nn.Module):
    """SSD锚框生成器"""
    def __init__(self, fig_size=300, feature_maps=[38, 19, 10, 5, 3, 1],
                 steps=[8, 16, 32, 64, 100, 300],
                 min_sizes=[30, 60, 111, 162, 213, 264],
                 max_sizes=[60, 111, 162, 213, 264, 315],
                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]):
        super(AnchorBoxes, self).__init__()
        
        self.fig_size = fig_size
        self.feature_maps = feature_maps
        self.steps = steps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
    
    def forward(self, x):
        """生成所有锚框"""
        anchors = []
        
        for k, f in enumerate(self.feature_maps):
            min_size = self.min_sizes[k]
            max_size = self.max_sizes[k]
            step = self.steps[k]
            aspect_ratios = self.aspect_ratios[k]
            
            # 生成基础锚框
            base_anchors = []
            # 1:1 比例的小锚框
            base_anchors.extend(generate_anchors(min_size, [1.0]))
            # 1:1 比例的大锚框
            base_anchors.extend(generate_anchors(np.sqrt(min_size * max_size), [1.0]))
            # 其他比例的锚框
            for ar in aspect_ratios:
                base_anchors.extend(generate_anchors(min_size, [ar, 1/ar]))
            
            base_anchors = torch.stack(base_anchors)
            
            # 生成网格中心
            shifts_x = torch.arange(0, f) * step
            shifts_y = torch.arange(0, f) * step
            shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            
            # 生成所有锚框
            anchors_layer = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors_layer = anchors_layer.reshape(-1, 4)
            
            # 归一化到[0, 1]
            anchors_layer[:, 0::2] /= self.fig_size
            anchors_layer[:, 1::2] /= self.fig_size
            
            anchors.append(anchors_layer)
        
        # 合并所有锚框
        anchors = torch.cat(anchors, dim=0)
        
        # 裁剪到图像边界
        anchors = torch.clamp(anchors, 0, 1)
        
        return anchors

def decode_boxes(loc, anchors):
    """解码锚框"""
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    
    dx = loc[:, 0]
    dy = loc[:, 1]
    dw = loc[:, 2]
    dh = loc[:, 3]
    
    ctr_x = dx * widths + ctr_x
    ctr_y = dy * heights + ctr_y
    widths = torch.exp(dw) * widths
    heights = torch.exp(dh) * heights
    
    x1 = ctr_x - 0.5 * widths
    y1 = ctr_y - 0.5 * heights
    x2 = ctr_x + 0.5 * widths
    y2 = ctr_y + 0.5 * heights
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes

def nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制"""
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.int64)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    
    # 按分数降序排序
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        # 选择当前最高分数的框
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # 计算与其他框的IoU
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的框
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep, dtype=torch.int64)

def detect(predicted_locations, predicted_scores, anchors, min_score=0.01, max_overlap=0.45, top_k=200):
    """检测后处理"""
    batch_size = predicted_locations.size(0)
    num_anchors = anchors.size(0)
    num_classes = predicted_scores.size(2)
    
    # 应用softmax到分类分数
    predicted_scores = F.softmax(predicted_scores, dim=2)
    
    detections = []
    
    for i in range(batch_size):
        # 解码锚框
        decoded_boxes = decode_boxes(predicted_locations[i], anchors)
        
        # 每个类别的检测结果
        batch_detections = []
        
        for c in range(1, num_classes):  # 跳过背景类
            # 获取当前类别的分数
            class_scores = predicted_scores[i, :, c]
            
            # 过滤低分数的框
            mask = class_scores > min_score
            class_scores = class_scores[mask]
            
            if class_scores.numel() == 0:
                continue
                
            class_boxes = decoded_boxes[mask]
            
            # 非极大值抑制
            keep = nms(class_boxes, class_scores, iou_threshold=max_overlap)
            
            # 保留前top_k个结果
            keep = keep[:top_k]
            
            # 构造检测结果
            batch_detections.extend([
                {
                    "box": class_boxes[k].tolist(),
                    "score": class_scores[k].item(),
                    "label": c
                }
                for k in keep
            ])
        
        detections.append(batch_detections)
    
    return detections