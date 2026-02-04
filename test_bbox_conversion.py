"""
测试边界框转换的正确性
验证COCO格式到SSD格式的转换和归一化是否正确
"""
import json
import numpy as np

def test_bbox_conversion():
    """测试边界框转换逻辑"""
    print("=" * 60)
    print("测试边界框转换")
    print("=" * 60)
    
    # 测试用例1: 正常情况
    print("\n测试用例1: 正常边界框")
    width, height = 100, 100
    coco_bbox = [10, 20, 30, 40]  # [x, y, w, h]
    
    x, y, w, h = coco_bbox
    x1 = x / width
    y1 = y / height
    x2 = (x + w) / width
    y2 = (y + h) / height
    
    ssd_bbox = [x1, y1, x2, y2]
    print(f"  原始图像尺寸: {width}x{height}")
    print(f"  COCO格式: {coco_bbox} (绝对坐标)")
    print(f"  SSD格式: {ssd_bbox} (归一化坐标)")
    print(f"  预期: [0.1, 0.2, 0.4, 0.6]")
    assert np.allclose(ssd_bbox, [0.1, 0.2, 0.4, 0.6]), "转换错误!"
    print("  ✓ 通过")
    
    # 测试用例2: 边界情况 - 边界框在图像边缘
    print("\n测试用例2: 边界框在图像边缘")
    width, height = 50, 50
    coco_bbox = [0, 0, 50, 50]  # 整个图像
    
    x, y, w, h = coco_bbox
    x1 = x / width
    y1 = y / height
    x2 = (x + w) / width
    y2 = (y + h) / height
    
    ssd_bbox = [x1, y1, x2, y2]
    print(f"  原始图像尺寸: {width}x{height}")
    print(f"  COCO格式: {coco_bbox}")
    print(f"  SSD格式: {ssd_bbox}")
    print(f"  预期: [0.0, 0.0, 1.0, 1.0]")
    assert np.allclose(ssd_bbox, [0.0, 0.0, 1.0, 1.0]), "转换错误!"
    print("  ✓ 通过")
    
    # 测试用例3: 小尺寸图像 (类似GTSRB数据集)
    print("\n测试用例3: 小尺寸图像 (GTSRB实际情况)")
    width, height = 27, 26  # 实际数据集中的尺寸
    coco_bbox = [5, 5, 17, 15]  # 实际标注
    
    x, y, w, h = coco_bbox
    x1 = x / width
    y1 = y / height
    x2 = (x + w) / width
    y2 = (y + h) / height
    
    ssd_bbox = [x1, y1, x2, y2]
    print(f"  原始图像尺寸: {width}x{height}")
    print(f"  COCO格式: {coco_bbox}")
    print(f"  SSD格式: {ssd_bbox}")
    
    # 验证坐标在[0, 1]范围内
    assert all(0 <= coord <= 1 for coord in ssd_bbox), "坐标超出范围!"
    # 验证x2 > x1, y2 > y1
    assert ssd_bbox[2] > ssd_bbox[0], "x2应该大于x1!"
    assert ssd_bbox[3] > ssd_bbox[1], "y2应该大于y1!"
    print("  ✓ 通过")
    
    # 测试用例4: 验证面积比例
    print("\n测试用例4: 验证面积比例")
    width, height = 100, 100
    coco_bbox = [10, 10, 20, 30]
    
    # 原始面积
    original_area = coco_bbox[2] * coco_bbox[3]
    
    # 转换后的面积 (归一化)
    x, y, w, h = coco_bbox
    x1, y1 = x / width, y / height
    x2, y2 = (x + w) / width, (y + h) / height
    normalized_area = (x2 - x1) * (y2 - y1)
    
    # 面积比例应该等于原始面积除以图像面积
    expected_ratio = original_area / (width * height)
    
    print(f"  原始面积: {original_area}")
    print(f"  归一化面积: {normalized_area}")
    print(f"  预期比例: {expected_ratio}")
    assert np.isclose(normalized_area, expected_ratio), "面积比例错误!"
    print("  ✓ 通过")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)

def test_real_dataset():
    """测试真实数据集的边界框"""
    print("\n" + "=" * 60)
    print("测试真实数据集")
    print("=" * 60)
    
    try:
        # 加载真实标注文件
        with open("datasets/GTSRB/annotations/train.json", "r") as f:
            data = json.load(f)
        
        # 获取第一个样本
        image_info = data["images"][0]
        annotation = data["annotations"][0]
        
        print(f"\n图像信息:")
        print(f"  文件名: {image_info['file_name']}")
        print(f"  尺寸: {image_info['width']}x{image_info['height']}")
        
        print(f"\n标注信息:")
        print(f"  类别ID: {annotation['category_id']}")
        print(f"  COCO bbox: {annotation['bbox']}")
        
        # 转换
        width = image_info['width']
        height = image_info['height']
        x, y, w, h = annotation['bbox']
        
        x1 = x / width
        y1 = y / height
        x2 = (x + w) / width
        y2 = (y + h) / height
        
        ssd_bbox = [x1, y1, x2, y2]
        print(f"  SSD bbox: {ssd_bbox}")
        
        # 验证
        assert all(0 <= coord <= 1 for coord in ssd_bbox), "坐标超出范围!"
        assert ssd_bbox[2] > ssd_bbox[0] and ssd_bbox[3] > ssd_bbox[1], "边界框无效!"
        
        print("\n✓ 真实数据集测试通过!")
        
    except FileNotFoundError:
        print("\n⚠ 未找到数据集文件,跳过真实数据测试")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        raise

if __name__ == "__main__":
    test_bbox_conversion()
    test_real_dataset()
    print("\n🎉 所有测试完成!")
