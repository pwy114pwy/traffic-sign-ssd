import os
import json
import pandas as pd
from PIL import Image
from pathlib import Path

def convert_gtsrb_to_coco(csv_path, img_base_dir, output_json_path, output_img_dir, set_name="train"):
    """
    将GTSRB数据集转换为COCO格式
    
    Args:
        csv_path: CSV文件路径
        img_base_dir: 图像基础目录
        output_json_path: 输出JSON文件路径
        output_img_dir: 输出图像目录
        set_name: 数据集名称 (train/val)
    """
    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # COCO格式数据结构
    coco_data = {
        "info": {
            "description": "GTSRB Dataset",
            "version": "1.0",
            "year": 2023
        },
        "licenses": [
            {
                "id": 1,
                "name": "GTSRB License",
                "url": "https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset"
            }
        ],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 创建类别列表
    class_ids = sorted(df["ClassId"].unique())
    for class_id in class_ids:
        coco_data["categories"].append({
            "id": int(class_id),
            "name": f"class_{class_id}",
            "supercategory": "traffic_sign"
        })
    
    # 处理图像和标注
    image_id = 1
    annotation_id = 1
    
    for _, row in df.iterrows():
        # 获取图像信息
        width = int(row["Width"])
        height = int(row["Height"])
        class_id = int(row["ClassId"])
        roi_x1 = int(row["Roi.X1"])
        roi_y1 = int(row["Roi.Y1"])
        roi_x2 = int(row["Roi.X2"])
        roi_y2 = int(row["Roi.Y2"])
        path = row["Path"]
        
        # 构建完整图像路径
        if set_name == "train":
            full_img_path = os.path.join(img_base_dir, path)
        else:  # test
            full_img_path = os.path.join(img_base_dir, path)
        
        # 检查图像是否存在
        if not os.path.exists(full_img_path):
            print(f"警告: {full_img_path} 不存在")
            continue
        
        # 读取图像
        img = Image.open(full_img_path)
        
        # 生成输出图像文件名
        output_img_name = f"{str(image_id-1).zfill(5)}.jpg"
        output_img_path = os.path.join(output_img_dir, output_img_name)
        
        # 保存图像为JPG格式
        img.convert("RGB").save(output_img_path, "JPEG")
        
        # 添加图像信息到COCO数据
        coco_data["images"].append({
            "id": image_id,
            "file_name": output_img_name,
            "width": width,
            "height": height,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        
        # 计算标注框信息 (COCO格式使用x1, y1, width, height)
        bbox_width = roi_x2 - roi_x1
        bbox_height = roi_y2 - roi_y1
        
        # 添加标注信息到COCO数据
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [roi_x1, roi_y1, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "segmentation": [],
            "iscrowd": 0
        })
        
        # 更新ID
        image_id += 1
        annotation_id += 1
    
    # 保存COCO格式JSON文件
    with open(output_json_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"{set_name}集转换完成: {len(coco_data['images'])} 张图像, {len(coco_data['annotations'])} 个标注")
    print(f"图像保存到: {output_img_dir}")
    print(f"标注保存到: {output_json_path}")

def main():
    # 以下是在本地训练的路径
    # 项目根目录
    project_root = Path(__file__).parent
    
    # GTSRB数据集路径
    gtsrb_dir = project_root / "datasets" / "GTSRB"
    
    # 输出目录
    output_dir = project_root / "datasets" / "GTSRB"
    
    # 转换训练集
    print("正在转换训练集...")
    convert_gtsrb_to_coco(
        csv_path=str(gtsrb_dir / "Train.csv"),
        img_base_dir=str(gtsrb_dir),
        output_json_path=str(output_dir / "annotations" / "train.json"),
        output_img_dir=str(output_dir / "images" / "train"),
        set_name="train"
    )
    
    # 转换测试集
    print("\n正在转换测试集...")
    convert_gtsrb_to_coco(
        csv_path=str(gtsrb_dir / "Test.csv"),
        img_base_dir=str(gtsrb_dir),
        output_json_path=str(output_dir / "annotations" / "val.json"),
        output_img_dir=str(output_dir / "images" / "val"),
        set_name="val"
    )
    
    print("\n✅ 转换完成!")

    # 以下是在服务器上训练的路径
    # GTSRB数据集路径
    # gtsrb_dir = "/kaggle/input/gtsrb-german-traffic-sign"
    
    # # 输出目录
    # output_dir = "/content/traffic-sign-ssd/datasets/GTSRB"
    # # 转换训练集
    # print("正在转换训练集...")
    # convert_gtsrb_to_coco(
    #     csv_path=str(gtsrb_dir +"/"+ "Train.csv"),
    #     img_base_dir=str(gtsrb_dir),
    #     output_json_path=str(output_dir +"/"+ "annotations" +"/"+ "train.json"),
    #     output_img_dir=str(output_dir +"/"+ "images" +"/"+ "train"),
    #     set_name="train"
    # )
    
    # # 转换测试集
    # print("\n正在转换测试集...")
    # convert_gtsrb_to_coco(
    #     csv_path=str(gtsrb_dir +"/"+ "Test.csv"),
    #     img_base_dir=str(gtsrb_dir),
    #     output_json_path=str(output_dir +"/"+ "annotations" +"/"+ "val.json"),
    #     output_img_dir=str(output_dir +"/"+ "images" +"/"+ "val"),
    #     set_name="val"
    # )
    
    # print("\n✅ 转换完成!")

if __name__ == "__main__":
    main()