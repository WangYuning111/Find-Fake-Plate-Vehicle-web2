#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试检测修正效果"""
from inference import load_models, detect_vehicle, preprocess_image
import random

random.seed(42)
all_images = []
import os
for root, dirs, files in os.walk('images'):
    for f in files:
        if f.endswith('.jpg'):
            all_images.append(os.path.join(root, f))

random.shuffle(all_images)
selected = all_images[:50]

models = load_models()

failures = []
successes = []

for path in selected:
    try:
        img = preprocess_image(path)
        detections = detect_vehicle(img, models)
        if len(detections) == 0:
            failures.append((path, 'no_detection'))
        else:
            best = max(detections, key=lambda x: x['confidence'])
            successes.append((path, best['class_name'], best['confidence'], best.get('scale', 'original')))
    except Exception as e:
        failures.append((path, str(e)))

print(f'测试50张图片:')
print(f'  成功: {len(successes)}')
print(f'  失败: {len(failures)}')
print()
if failures:
    print('失败图片:')
    for path, reason in failures:
        print(f'  {path}: {reason}')
    print()
print('成功图片（前10张）:')
for path, name, conf, scale in successes[:10]:
    print(f'  {path}: {name}({conf:.3f}) [{scale}]')
