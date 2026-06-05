#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版推理脚本：YOLO 检测 + 独立品牌分类

使用方式：
  1. 先用 train_vehicle_brand.py 训练品牌分类模型
  2. 将本脚本集成到 start_server_lazy.py 中替换原有品牌识别逻辑

核心改进：
  - YOLO 只做车辆检测（输出检测框坐标）
  - 裁剪车辆区域后，送入 ResNet50 品牌分类模型
  - 品牌识别与检测解耦，互不干扰
"""
import os
import json
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 配置 ====================
YOLO_PATH = 'cfg/best.pt'
BRAND_MODEL_PATH = 'cfg/vehicle_brand_cls.pth'
BRAND_CLASSES_PATH = 'cfg/vehicle_brand_classes.json'
INPUT_SIZE = 224

# ==================== 图像预处理 ====================
brand_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== 加载模型 ====================
class BrandClassifier:
    def __init__(self):
        self.model = None
        self.brand_names = {}
        self._load()
    
    def _load(self):
        if not os.path.exists(BRAND_MODEL_PATH):
            print(f"[WARN] 品牌分类模型不存在: {BRAND_MODEL_PATH}")
            print("[WARN] 请先运行 train_vehicle_brand.py 训练模型")
            return
        
        # 加载类别映射
        with open(BRAND_CLASSES_PATH, 'r', encoding='utf-8') as f:
            self.brand_names = json.load(f)
        num_classes = len(self.brand_names)
        
        # 构建模型
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        self.model.load_state_dict(torch.load(BRAND_MODEL_PATH, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        print(f"[MODEL] 品牌分类模型加载完成，支持 {num_classes} 个品牌")
        print(f"[MODEL] 品牌列表: {list(self.brand_names.values())}")
    
    def predict(self, crop_image):
        """
        对车辆裁剪图进行品牌分类
        :param crop_image: OpenCV 读取的 BGR 图像 (numpy array)
        :return: (品牌名称, 置信度)
        """
        if self.model is None:
            return "未知", 0.0
        
        # BGR -> RGB
        rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        
        # 预处理
        tensor = brand_transform(rgb_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        brand_name = self.brand_names.get(str(pred_idx), self.brand_names.get(pred_idx, "未知"))
        
        # 输出 Top-3 结果供调试
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(self.brand_names.get(str(i), self.brand_names.get(i, "?")), float(probs[i])) for i in top3_idx]
        print(f"[BRAND] Top-3: {top3}")
        
        return brand_name, confidence


# ==================== 完整推理流程示例 ====================
def predict_improved(img_path):
    """
    改进版预测流程示例
    """
    # 1. YOLO 检测车辆位置
    yolo = YOLO(YOLO_PATH)
    original = cv2.imread(img_path)
    results = yolo(source=original, conf=0.1, iou=0.45)
    result = results[0]
    
    if len(result.boxes) == 0:
        return "未检测到车辆"
    
    # 取置信度最高的检测框
    best_idx = int(result.boxes.conf.cpu().numpy().argmax())
    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[best_idx].astype(int)
    
    # 裁剪车辆区域
    h, w = original.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = original[y1:y2, x1:x2]
    
    # 2. 品牌分类（独立模型）
    brand_cls = BrandClassifier()
    brand, brand_conf = brand_cls.predict(crop)
    
    print(f"\n检测结果:")
    print(f"  检测框: ({x1}, {y1}, {x2}, {y2})")
    print(f"  品牌: {brand} (置信度: {brand_conf:.2%})")
    
    return brand, brand_conf


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        predict_improved(sys.argv[1])
    else:
        print("用法: python predict_with_brand_cls.py <图片路径>")
        print("\n请先完成以下步骤:")
        print("  1. 准备品牌分类数据集到 datasets/vehicle_brand_cls/")
        print("  2. 运行 python train_vehicle_brand.py 训练模型")
        print("  3. 再运行本脚本测试")
