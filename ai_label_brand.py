#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI辅助品牌标注工具

由于AI无法直接批量处理本地图片文件，本工具提供交互式标注界面：
1. 随机抽取图片
2. 用户（或AI）查看图片并输入真实品牌
3. 保存标注到CSV，累积训练数据
"""
import os
import csv
import cv2
import random
from datetime import datetime

LABEL_FILE = 'data/brand_labels.csv'

# 标准品牌列表
BRANDS = [
    '大众', '别克', '福特', '东风', '长安', '日产', '现代', '一汽', '江淮', '金杯',
    '福田', '江铃', '哈飞', '雪铁龙', '中华', '雪佛兰', '奥迪', '长城', '标致',
    '马自达', '铃木', '五菱', '海马', '宝马', '荣威', '比亚迪', '奇瑞', '本田', '丰田', '起亚', '奔驰', '其他'
]


def load_existing_labels():
    """加载已有的标注"""
    labels = {}
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row['image_path']] = row['brand']
    return labels


def save_label(image_path, brand):
    """保存一条标注"""
    exists = os.path.exists(LABEL_FILE)
    with open(LABEL_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['image_path', 'brand', 'timestamp'])
        writer.writerow([image_path, brand, datetime.now().isoformat()])


def interactive_label(num_images=20):
    """
    交互式标注（供用户手动标注时使用）
    """
    existing = load_existing_labels()

    # 收集所有未标注的图片
    all_images = []
    for root, dirs, files in os.walk('images'):
        for f in files:
            if f.endswith('.jpg'):
                path = os.path.join(root, f)
                if path not in existing:
                    all_images.append(path)

    random.shuffle(all_images)
    to_label = all_images[:num_images]

    print(f"共有 {len(all_images)} 张未标注图片")
    print(f"本次标注 {len(to_label)} 张")
    print("\n可选品牌:", ' '.join(BRANDS))
    print("输入格式: 品牌名 或 'skip'跳过 'quit'退出\n")

    for path in to_label:
        print(f"\n图片: {path}")
        # 这里可以显示图片（如果在有GUI的环境）
        brand = input("真实品牌: ").strip()
        if brand == 'quit':
            break
        if brand == 'skip':
            continue
        if brand in BRANDS:
            save_label(path, brand)
            print(f"  已保存: {brand}")
        else:
            print(f"  未知品牌 '{brand}'，已跳过")


if __name__ == '__main__':
    interactive_label()
