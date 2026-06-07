#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备品牌识别训练数据集

策略：
1. 使用现有 images 目录图片
2. 用 AI 视觉识别生成伪标签（作为初始标注）
3. 用户可在此基础上修正

YOLO 格式标注：
  class_id x_center y_center width height（均为相对于图片尺寸的归一化值）
"""
import os
import cv2
import json
from pathlib import Path
from ultralytics import YOLO

# 标准品牌列表（与 vehicle-database.csv 一致）
STANDARD_BRANDS = [
    '大众', '别克', '福特', '东风', '长安', '日产', '现代', '一汽', '江淮', '金杯',
    '福田', '江铃', '哈飞', '雪铁龙', '中华', '雪佛兰', '奥迪', '长城', '标致',
    '马自达', '铃木', '五菱', '海马', '宝马', '荣威', '比亚迪', '奇瑞', '本田', '丰田', '起亚', '奔驰'
]

# 创建品牌到ID的映射
BRAND_TO_ID = {brand: i for i, brand in enumerate(STANDARD_BRANDS)}

# 输出目录
OUTPUT_DIR = 'datasets/vehicle_brand'
os.makedirs(f'{OUTPUT_DIR}/images/train', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels/train', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/images/val', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels/val', exist_ok=True)


def detect_and_label(img_path, yolo_model):
    """
    使用YOLO检测车辆位置，生成标注框
    品牌由外部传入（AI识别或人工标注）
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    results = yolo_model(img, conf=0.1, verbose=False)[0]

    if len(results.boxes) == 0:
        return None

    # 取置信度最高的检测框
    best_idx = int(results.boxes.conf.cpu().numpy().argmax())
    box = results.boxes.xyxy.cpu().numpy()[best_idx]
    x1, y1, x2, y2 = box

    # 转换为YOLO格式（归一化）
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return {
        'bbox': [x_center, y_center, bw, bh],
        'img': img,
        'shape': (h, w)
    }


def create_dataset():
    """创建数据集配置文件"""
    data_yaml = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

nc: {len(STANDARD_BRANDS)}
names: {STANDARD_BRANDS}
"""
    with open(f'{OUTPUT_DIR}/data.yaml', 'w', encoding='utf-8') as f:
        f.write(data_yaml)
    print(f"数据集配置已保存: {OUTPUT_DIR}/data.yaml")
    print(f"类别数: {len(STANDARD_BRANDS)}")
    print(f"品牌列表: {STANDARD_BRANDS}")


if __name__ == '__main__':
    create_dataset()
    print("\n请按以下步骤完成品牌数据集准备:")
    print("1. 将图片复制到 datasets/vehicle_brand/images/train/ 和 val/")
    print("2. 为每张图片创建对应的 .txt 标注文件到 labels/train/ 和 labels/val/")
    print("3. 标注格式: <brand_id> <x_center> <y_center> <width> <height>")
    print("4. 运行 python train_yolo.py 重新训练")
