import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ssd import build_ssd
from models.utils import detect, AnchorBoxes

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 44
MODEL_PATH = "models/saved/best_ssd_model.pth"

# 加载模型
def load_model():
    model = build_ssd(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 图像预处理
def get_transform(train=False):
    transforms_list = [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transforms_list)

# 加载测试图像
test_image_path = "datasets/GTSRB/Test/00001.png"
if os.path.exists(test_image_path):
    image = Image.open(test_image_path).convert('RGB')
    print(f"Loaded test image: {test_image_path}")
    print(f"Image size: {image.size}")
else:
    # 如果没有测试图像，使用第一个找到的图像
    import glob
    test_images = glob.glob("datasets/GTSRB/Test/*.png")
    if test_images:
        test_image_path = test_images[0]
        image = Image.open(test_image_path).convert('RGB')
        print(f"Loaded test image: {test_image_path}")
        print(f"Image size: {image.size}")
    else:
        # 如果没有测试图像，创建一个简单的测试图像
        image = Image.new('RGB', (300, 300), color=(255, 255, 255))
        print("Created a test image (no test images found)")

# 加载模型和锚框
print("Loading model...")
model = load_model()
print("Model loaded successfully!")

anchor_boxes = AnchorBoxes()
anchors = anchor_boxes(None)
anchors = anchors.to(DEVICE)

# 预处理图像
transform = get_transform(train=False)
processed_image = transform(image).unsqueeze(0).to(DEVICE)
print(f"Processed image shape: {processed_image.shape}")

# 执行推理
print("Running inference...")
with torch.no_grad():
    predicted_scores, predicted_locations = model(processed_image)
    print(f"Predicted scores shape: {predicted_scores.shape}")
    print(f"Predicted locations shape: {predicted_locations.shape}")
    
    # 打印预测分数的最大值，查看模型是否有输出
    max_scores = predicted_scores.max(dim=2)[0].max(dim=1)[0]
    print(f"Max predicted score: {max_scores.item():.4f}")
    
    # 检测
    detections = detect(predicted_locations, predicted_scores, anchors, 
                       min_score=0.01,  # 进一步降低阈值
                       max_overlap=0.45, 
                       top_k=200)
    
    print(f"Number of detections: {len(detections[0])}")
    for i, detection in enumerate(detections[0]):
        print(f"Detection {i+1}: Score={detection['score']:.4f}, Label={detection['label']}, Box={detection['box']}")

print("Test completed!")
