import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGBase(nn.Module):
    """SSD的基础网络 - VGG16的前10层"""
    def __init__(self):
        super(VGGBase, self).__init__()
        
        # 前四层卷积
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 剩余的网络层
        self.remaining = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        x = self.conv3_3(x)
        conv4_3_features = self.conv4_3(x)
        x = self.remaining(conv4_3_features)
        return conv4_3_features, x

class ExtraLayers(nn.Module):
    """SSD的额外卷积层，用于多尺度特征检测"""
    def __init__(self):
        super(ExtraLayers, self).__init__()
        
        self.extra = nn.ModuleList([
            # Conv6
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            
            # Conv7
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            
            # Conv8
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            ),
            
            # Conv9
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            )
        ])
    
    def forward(self, x):
        features = []
        for layer in self.extra:
            x = layer(x)
            features.append(x)
        return features

class DetectionHead(nn.Module):
    """SSD的检测头 - 用于分类和回归"""
    def __init__(self, num_classes=44, num_anchors=8732):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 回归头 (位置偏移)
        self.regression_head = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),  # Conv4_3
            nn.Conv2d(1024, 4 * 6, kernel_size=3, padding=1),  # FC7
            nn.Conv2d(512, 4 * 6, kernel_size=3, padding=1),  # Conv6
            nn.Conv2d(256, 4 * 6, kernel_size=3, padding=1),  # Conv7
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),  # Conv8
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)   # Conv9
        ])
        
        # 分类头 (类别概率)
        self.classification_head = nn.ModuleList([
            nn.Conv2d(512, num_classes * 4, kernel_size=3, padding=1),  # Conv4_3
            nn.Conv2d(1024, num_classes * 6, kernel_size=3, padding=1),  # FC7
            nn.Conv2d(512, num_classes * 6, kernel_size=3, padding=1),  # Conv6
            nn.Conv2d(256, num_classes * 6, kernel_size=3, padding=1),  # Conv7
            nn.Conv2d(256, num_classes * 4, kernel_size=3, padding=1),  # Conv8
            nn.Conv2d(256, num_classes * 4, kernel_size=3, padding=1)   # Conv9
        ])
    
    def forward(self, features):
        classifications = []
        regressions = []
        
        for i, feature in enumerate(features):
            # 分类预测
            cls = self.classification_head[i](feature)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(cls.size(0), -1, self.num_classes)
            classifications.append(cls)
            
            # 回归预测
            reg = self.regression_head[i](feature)
            reg = reg.permute(0, 2, 3, 1).contiguous()
            reg = reg.view(reg.size(0), -1, 4)
            regressions.append(reg)
        
        # 合并所有预测
        classifications = torch.cat(classifications, dim=1)
        regressions = torch.cat(regressions, dim=1)
        
        return classifications, regressions

class SSD(nn.Module):
    """完整的SSD模型"""
    def __init__(self, num_classes=44):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        
        # 基础网络
        self.base = VGGBase()
        
        # 额外特征层
        self.extra = ExtraLayers()
        
        # 检测头
        self.detection_head = DetectionHead(num_classes=num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        # 基础网络特征
        conv4_3_features, base_features = self.base(x)
        
        # 额外特征层
        extra_features = self.extra(base_features)
        
        # 收集所有特征层
        features = [conv4_3_features] + [base_features] + extra_features
        
        # 检测头预测
        classifications, regressions = self.detection_head(features)
        
        return classifications, regressions
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def build_ssd(num_classes=43):
    """构建SSD模型的工厂函数"""
    model = SSD(num_classes=num_classes)
    return model