#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
品牌识别规则库 + 后处理修正

基于观察到的错误模式建立规则，用于修正YOLO品牌识别
"""

# 品牌特征规则（基于车标视觉特征）
# 用于在YOLO输出"其他"或错误时做推断
BRAND_VISUAL_RULES = {
    # 圆形车标
    '日产': {'shapes': ['circle'], 'keywords': ['日产', '尼桑', 'nissan']},
    '宝马': {'shapes': ['circle'], 'keywords': ['宝马', 'bmw']},
    '奔驰': {'shapes': ['circle'], 'keywords': ['奔驰', 'benz']},
    '丰田': {'shapes': ['oval'], 'keywords': ['丰田', 'toyota']},
    # 盾形/特殊形状
    '别克': {'shapes': ['shield'], 'keywords': ['别克', 'buick']},
    '雪佛兰': {'shapes': ['cross'], 'keywords': ['雪佛兰', 'chevrolet']},
    '奥迪': {'shapes': ['rings'], 'keywords': ['奥迪', 'audi']},
    # V型/翼型
    '长安': {'shapes': ['v_shape'], 'keywords': ['长安']},
    '五菱': {'shapes': ['w_shape'], 'keywords': ['五菱']},
    # 文字标识
    '一汽': {'shapes': ['eagle'], 'keywords': ['一汽', '解放']},
    '东风': {'shapes': ['swirl'], 'keywords': ['东风']},
    '本田': {'shapes': ['h_shape'], 'keywords': ['本田', 'honda']},
    '现代': {'shapes': ['h_slant'], 'keywords': ['现代', 'hyundai']},
    '起亚': {'shapes': ['k_shape'], 'keywords': ['起亚', 'kia']},
    '福特': {'shapes': ['oval'], 'keywords': ['福特', 'ford']},
    '大众': {'shapes': ['vw'], 'keywords': ['大众', 'vw', 'volkswagen']},
}

# 品牌-车型强关联（某些品牌只做特定车型）
BRAND_TYPE_STRONG = {
    '五菱': ['minibus'],
    '金杯': ['minibus'],
    '福田': ['truck'],
    '江铃': ['truck'],
    '哈飞': ['minibus'],
    '江淮': ['truck', 'bus', 'minibus'],
    '东风': ['truck', 'bus'],
    '一汽': ['truck', 'bus'],
}

# 常见品牌颜色分布（用于辅助判断）
BRAND_COLOR_PREF = {
    '大众出租车': ['green', 'blue'],
    '丰田': ['black', 'white', 'silver'],
    '本田': ['black', 'white'],
    '日产': ['silver', 'black', 'white'],
    '别克': ['black', 'silver'],
    '福特': ['black', 'white'],
    '奥迪': ['black', 'white', 'silver'],
    '宝马': ['black', 'white', 'silver'],
    '奔驰': ['black', 'silver', 'white'],
}


def infer_brand_from_type_color(vehicle_type, vehicle_color, yolo_brand_raw):
    """
    当YOLO品牌识别为"其他"或置信度低时，根据车型+颜色推断品牌
    """
    # 出租车特征
    if vehicle_color in ['green', 'blue'] and vehicle_type == 'car':
        return '大众'  # 大多数出租车是大众

    # 面包车特征
    if vehicle_type == 'minibus':
        if vehicle_color == 'silver':
            return '五菱'
        return '长安'

    # 大巴
    if vehicle_type == 'bus':
        return '其他'

    # 卡车
    if vehicle_type == 'truck':
        if vehicle_color == 'blue':
            return '一汽'
        return '东风'

    return '其他'


def correct_brand_advanced(yolo_brand, vehicle_type, vehicle_color, confidence=0.0):
    """
    高级品牌修正：结合YOLO输出、车型、颜色做综合判断
    """
    # 如果YOLO置信度高且品牌明确，直接返回
    if confidence > 0.5 and yolo_brand not in ['其他', '黄牌大巴', '黄牌卡车', '公交车']:
        # 提取标准品牌名
        brand_map = {
            '大众家用车': '大众', '大众出租车': '大众',
            '别克商务车': '别克', '别克': '别克',
            '福特面包车': '福特', '福特': '福特',
            '东风卡车': '东风', '东风面包车': '东风', '东风': '东风',
            '长安面包车': '长安', '长安': '长安',
            '日产越野车': '日产', '日产': '日产',
            '现代越野车': '现代', '现代面包车': '现代', '现代': '现代',
            '一汽卡车': '一汽', '一汽面包车': '一汽', '一汽': '一汽',
            '江淮面包车': '江淮', '江淮卡车': '江淮', '江淮': '江淮',
            '金杯卡车': '金杯', '金杯面包车': '金杯',
            '福田小卡': '福田', '福田大卡': '福田',
            '雪佛兰': '雪佛兰', '奥迪': '奥迪', '长城': '长城',
            '标致': '标致', '马自达': '马自达', '铃木': '铃木',
            '五菱': '五菱', '海马': '海马', '宝马': '宝马',
            '荣威': '荣威', '比亚迪': '比亚迪', '奇瑞': '奇瑞',
            '本田': '本田', '丰田': '丰田', '起亚': '起亚',
            '奔驰': '奔驰', '江铃': '江铃', '哈飞': '哈飞',
            '雪铁龙': '雪铁龙', '中华': '中华',
        }
        return brand_map.get(yolo_brand, yolo_brand)

    # 置信度低或品牌不明确，用规则推断
    return infer_brand_from_type_color(vehicle_type, vehicle_color, yolo_brand)


if __name__ == '__main__':
    # 测试
    test_cases = [
        ('其他', 'car', 'green', 0.2),
        ('黄牌大巴', 'bus', 'white', 0.8),
        ('一汽卡车', 'truck', 'blue', 0.6),
        ('其他', 'minibus', 'silver', 0.15),
        ('福特面包车', 'minibus', 'white', 0.45),
    ]
    print("品牌推断测试:")
    for yolo, vtype, color, conf in test_cases:
        result = correct_brand_advanced(yolo, vtype, color, conf)
        print(f"  YOLO={yolo:12s} 车型={vtype:8s} 颜色={color:6s} 置信度={conf:.2f} -> {result}")
