#!/usr/bin/env python3
"""
测试YOLO品牌识别准确率
"""
import os
import cv2
from ultralytics import YOLO
from collections import defaultdict

yolo = YOLO('weights/best.pt')

brand_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})

# 遍历images目录，按文件夹名作为真实品牌
for root, dirs, files in os.walk('images'):
    for f in files:
        if not f.endswith('.jpg'):
            continue
        path = os.path.join(root, f)
        parts = path.replace('\\', '/').split('/')
        if len(parts) < 3:
            continue
        true_brand = parts[-2]

        img = cv2.imread(path)
        if img is None:
            continue

        results = yolo(img, conf=0.1, verbose=False)[0]
        if len(results.boxes) == 0:
            continue

        best_idx = int(results.boxes.conf.cpu().numpy().argmax())
        pred_cls = int(results.boxes.cls.cpu().numpy()[best_idx])
        pred_brand = results.names[pred_cls]

        # 判断是否匹配
        is_match = (true_brand in pred_brand) or (pred_brand in true_brand)

        brand_stats[true_brand]['total'] += 1
        if is_match:
            brand_stats[true_brand]['correct'] += 1
        else:
            brand_stats[true_brand]['errors'].append((f, pred_brand))

print('=' * 70)
print('YOLO品牌识别准确率统计')
print('=' * 70)
total_correct = 0
total_all = 0
for brand, info in sorted(brand_stats.items(), key=lambda x: -x[1]['total']):
    acc = info['correct'] / info['total'] * 100 if info['total'] > 0 else 0
    total_correct += info['correct']
    total_all += info['total']
    err_str = ''
    if info['errors']:
        samples = ', '.join([f'{e[0]}->{e[1]}' for e in info['errors'][:3]])
        err_str = f'  错误: {samples}'
    print(f'{brand:12s}: {info["correct"]:3d}/{info["total"]:3d} ({acc:5.1f}%){err_str}')

print('=' * 70)
print(f'总体: {total_correct}/{total_all} ({total_correct/total_all*100:.1f}%)')
