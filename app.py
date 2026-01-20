import os
import uuid
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
import numpy as np

from models.ssd import build_ssd
from models.utils import AnchorBoxes, detect
from datasets.dataset import get_transform

app = Flask(__name__)

# 配置上传文件的存储位置
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 43
MODEL_PATH = "models/saved/best_ssd_model.pth"  # 默认模型路径

# 加载交通标志类别名称
CLASS_NAMES = [
    "限速20", "限速30", "限速50", "限速60", "限速70", "限速80", "结束限速80", "限速100",
    "限速120", "禁止超车", "大型车辆禁止超车", "前方路口有停车让行标志", "优先级道路",
    "让行", "停止", "禁止车辆进入", "禁止大型车辆进入", "禁止驶入", "前方施工",
    "注意危险", "注意左侧来车", "注意右侧来车", "注意儿童", "注意前方有人行横道",
    "注意自行车", "注意路面滑", "注意道路变窄", "注意施工", "注意交通信号灯",
    "注意火车道口", "注意公交车站", "注意学校", "注意前方有避让标志", "注意野生动物",
    "禁止左转", "禁止右转", "禁止直行", "禁止直行和左转", "禁止直行和右转",
    "禁止掉头", "禁止车辆左转或右转", "前方有环岛", "减速让行"
]

# 加载模型函数
def load_model(model_path=MODEL_PATH):
    """加载训练好的SSD模型"""
    model = build_ssd(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 预加载模型
try:
    model = load_model()
    anchor_boxes = AnchorBoxes()
    anchors = anchor_boxes(None)
    anchors = anchors.to(DEVICE)
    transform = get_transform(train=False)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    """首页路由"""
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'image' not in request.files:
            return render_template('index.html', error='没有上传图像')
        
        image_file = request.files['image']
        
        # 检查文件是否为空
        if image_file.filename == '':
            return render_template('index.html', error='没有选择图像文件')
        
        # 检查文件类型
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return render_template('index.html', error='仅支持JPG、JPEG、PNG和BMP格式的图像')
        
        try:
            # 生成唯一的文件名
            filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 保存上传的图像
            image_file.save(image_path)
            
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 预处理图像
            processed_image = transform(image).unsqueeze(0).to(DEVICE)
            
            # 执行推理
            with torch.no_grad():
                predicted_scores, predicted_locations = model(processed_image)
                conf_threshold = float(request.form.get('conf_threshold', 0.5))
                detections = detect(predicted_locations, predicted_scores, anchors, 
                                   min_score=conf_threshold, 
                                   max_overlap=0.45, 
                                   top_k=200)
            
            # 可视化检测结果
            result_image = visualize_detections(processed_image[0], detections[0], original_size, 
                                               class_names=CLASS_NAMES, threshold=conf_threshold)
            
            # 保存结果图像
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            result_image.save(result_path)
            
            # 准备检测信息
            detection_count = len([d for d in detections[0] if d['score'] >= conf_threshold])
            detection_info = []
            for detection in detections[0]:
                if detection['score'] >= conf_threshold:
                    detection_info.append({
                        'class_name': CLASS_NAMES[detection['label'] - 1],
                        'score': detection['score']
                    })
            
            # 返回结果页面
            return render_template('index.html', 
                                  original_image=url_for('static', filename=f'images/{filename}'),
                                  result_image=url_for('static', filename=f'images/{result_filename}'),
                                  detection_count=detection_count,
                                  detections=detection_info)
            
        except Exception as e:
            return render_template('index.html', error=f'处理图像时出错: {str(e)}')
    
    # GET请求返回首页
    return render_template('index.html')

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
    from PIL import ImageDraw, ImageFont
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

@app.route('/static/images/<filename>')
def serve_image(filename):
    """提供静态图像文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)