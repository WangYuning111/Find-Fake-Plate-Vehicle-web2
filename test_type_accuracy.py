#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试车型识别准确率"""
from inference import predict_single

paths = [
    'images/bus/2412.jpg',
    'images/car/208.jpg',
    'images/minibus/1287.jpg',
    'images/truck/1323.jpg',
]

for path in paths:
    result = predict_single(path)
    print(f"{path}: 车型={result['vehicle_type']}")
