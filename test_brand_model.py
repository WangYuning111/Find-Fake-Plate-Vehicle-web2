#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试集成后的品牌分类器"""
from inference import predict_single

paths = [
    'images/bus/2412.jpg',
    'images/silver/3540.jpg',
    'images/yellow/Shift_6.jpg',
    'images/black/71.jpg',
    'images/black/239.jpg',
]

print('=' * 70)
print('集成独立品牌分类器后测试')
print('=' * 70)

for path in paths:
    result = predict_single(path)
    print(f'{path}:')
    print(f"  品牌={result['vehicle_brand']}, 车型={result['vehicle_type']}, 颜色={result['vehicle_color']}")
    print()
