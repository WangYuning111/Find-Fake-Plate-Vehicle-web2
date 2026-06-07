#!/usr/bin/env python3
"""
随机抽5张图片，测试品牌识别（修正后）
对比AI视觉识别的真实品牌
"""
import os
import random
from inference import predict_single

random.seed()

# 从所有图片中随机选5张
all_images = []
for root, dirs, files in os.walk('images'):
    for f in files:
        if f.endswith('.jpg'):
            all_images.append(os.path.join(root, f))

selected = random.sample(all_images, 5)

print("=" * 70)
print("随机抽取5张图片 - 品牌识别测试（修正后）")
print("=" * 70)

for path in selected:
    result = predict_single(path)
    brand = result['vehicle_brand']
    vtype = result['vehicle_type']
    color = result['vehicle_color']
    plate = result['plate_number']
    print(f"\n图片: {path}")
    print(f"  车牌: {plate}")
    print(f"  品牌: {brand}, 车型: {vtype}, 颜色: {color}")
    print("  --- 请告诉我真实品牌，我将对比修正 ---")
