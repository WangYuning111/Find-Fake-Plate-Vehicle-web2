#!/usr/bin/env python3
from inference import predict_single

paths = [
    'images/green/2177.jpg',
    'images/red/1190.jpg',
    'images/bus/2535.jpg',
    'images/car/365.jpg',
    'images/white/317.jpg',
]

print('=' * 70)
print('品牌修正后重新测试')
print('=' * 70)

for path in paths:
    result = predict_single(path)
    brand = result['vehicle_brand']
    vtype = result['vehicle_type']
    color = result['vehicle_color']
    print(f'{path}:')
    print(f'  品牌={brand}, 车型={vtype}, 颜色={color}')
    print()
