#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试颜色识别准确率"""
from inference import predict_single

paths = [
    'images/red/470.jpg',
    'images/blue/1235.jpg',
    'images/white/600.jpg',
    'images/black/14.jpg',
    'images/yellow/2611.jpg',
    'images/silver/290.jpg',
    'images/green/1033.jpg',
    'images/brown/116.jpg',
]

for path in paths:
    result = predict_single(path)
    folder_color = path.split('/')[1]
    pred_color = result['vehicle_color']
    correct = folder_color == pred_color
    print(f"{path}: 文件夹={folder_color}, 识别={pred_color}, {'✓' if correct else '✗'}")
