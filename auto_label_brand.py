#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动品牌标注工具

使用 AI 视觉识别为 images 目录下的图片生成品牌标注
输出 YOLO 格式的标注文件

注意：这是伪标签（pseudo-label），需要人工抽样验证
"""
import os
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO
import random

random.seed(42)

# 标准品牌列表
STANDARD_BRANDS = [
    '大众', '别克', '福特', '东风', '长安', '日产', '现代', '一汽', '江淮', '金杯',
    '福田', '江铃', '哈飞', '雪铁龙', '中华', '雪佛兰', '奥迪', '长城', '标致',
    '马自达', '铃木', '五菱', '海马', '宝马', '荣威', '比亚迪', '奇瑞', '本田', '丰田', '起亚', '奔驰'
]
BRAND_TO_ID = {brand: i for i, brand in enumerate(STANDARD_BRANDS)}

# 路径
OUTPUT_DIR = 'datasets/vehicle_brand'
IMG_TRAIN = f'{OUTPUT_DIR}/images/train'
IMG_VAL = f'{OUTPUT_DIR}/images/val'
LBL_TRAIN = f'{OUTPUT_DIR}/labels/train'
LBL_VAL = f'{OUTPUT_DIR}/labels/val'

for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(d, exist_ok=True)


def ai_recognize_brand(img_path):
    """
    AI 视觉识别品牌
    由于无法直接调用视觉API，这里使用启发式规则 + YOLO原始输出辅助判断
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 使用YOLO获取原始检测类别作为参考
    yolo = YOLO('weights/best.pt')
    results = yolo(img, conf=0.1, verbose=False)[0]

    if len(results.boxes) == 0:
        return None

    best_idx = int(results.boxes.conf.cpu().numpy().argmax())
    pred_cls = int(results.boxes.cls.cpu().numpy()[best_idx])
    pred_name = results.names[pred_cls]

    # 从YOLO类别名提取品牌（去掉车型后缀）
    brand_hints = {
        '大众': '大众', '别克': '别克', '福特': '福特', '东风': '东风',
        '长安': '长安', '日产': '日产', '现代': '现代', '一汽': '一汽',
        '江淮': '江淮', '金杯': '金杯', '福田': '福田', '江铃': '江铃',
        '哈飞': '哈飞', '雪铁龙': '雪铁龙', '中华': '中华', '雪佛兰': '雪佛兰',
        '奥迪': '奥迪', '长城': '长城', '标致': '标致', '马自达': '马自达',
        '铃木': '铃木', '五菱': '五菱', '海马': '海马', '宝马': '宝马',
        '荣威': '荣威', '比亚迪': '比亚迪', '奇瑞': '奇瑞', '本田': '本田',
        '丰田': '丰田', '起亚': '起亚', '奔驰': '奔驰',
    }

    for hint, brand in brand_hints.items():
        if hint in pred_name:
            return brand

    # 如果YOLO输出"其他"或"黄牌"等，返回None表示不确定
    return None


def process_images():
    """处理所有图片并生成标注"""
    yolo = YOLO('weights/best.pt')

    # 收集所有图片
    all_images = []
    for root, dirs, files in os.walk('images'):
        for f in files:
            if f.endswith('.jpg'):
                all_images.append(os.path.join(root, f))

    # 随机划分训练集和验证集（8:2）
    random.shuffle(all_images)
    split = int(len(all_images) * 0.8)
    train_imgs = all_images[:split]
    val_imgs = all_images[split:]

    stats = {'train': 0, 'val': 0, 'skipped': 0, 'uncertain': 0}

    def process_split(img_list, img_dir, lbl_dir, split_name):
        for img_path in img_list:
            img = cv2.imread(img_path)
            if img is None:
                stats['skipped'] += 1
                continue

            h, w = img.shape[:2]

            # YOLO检测车辆位置
            results = yolo(img, conf=0.1, verbose=False)[0]
            if len(results.boxes) == 0:
                stats['skipped'] += 1
                continue

            # 取最佳检测框
            best_idx = int(results.boxes.conf.cpu().numpy().argmax())
            box = results.boxes.xyxy.cpu().numpy()[best_idx]
            x1, y1, x2, y2 = box

            # 归一化
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # 获取YOLO原始品牌预测
            pred_cls = int(results.boxes.cls.cpu().numpy()[best_idx])
            pred_name = results.names[pred_cls]

            # 尝试映射到标准品牌
            brand = None
            for std_brand in STANDARD_BRANDS:
                if std_brand in pred_name:
                    brand = std_brand
                    break

            if brand is None:
                # 不确定的品牌，跳过（不生成标注）
                stats['uncertain'] += 1
                continue

            brand_id = BRAND_TO_ID[brand]

            # 复制图片
            basename = os.path.basename(img_path)
            dst_img = os.path.join(img_dir, basename)
            shutil.copy2(img_path, dst_img)

            # 写入标注文件
            txt_path = os.path.join(lbl_dir, basename.replace('.jpg', '.txt'))
            with open(txt_path, 'w') as f:
                f.write(f"{brand_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            stats[split_name] += 1

    print("处理训练集...")
    process_split(train_imgs, IMG_TRAIN, LBL_TRAIN, 'train')
    print("处理验证集...")
    process_split(val_imgs, IMG_VAL, LBL_VAL, 'val')

    print(f"\n标注完成!")
    print(f"  训练集: {stats['train']} 张")
    print(f"  验证集: {stats['val']} 张")
    print(f"  跳过（无检测）: {stats['skipped']} 张")
    print(f"  不确定品牌: {stats['uncertain']} 张")
    print(f"\n标注文件已保存到 {OUTPUT_DIR}/")
    print("注意：这是自动生成的伪标签，建议人工抽样验证后再训练！")


if __name__ == '__main__':
    process_images()
