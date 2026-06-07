#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
品牌识别后处理修正模块

问题：YOLO 模型同时做检测+品牌分类，但品牌类别设计混乱
解决：通过后处理映射 + 规则修正，将 YOLO 输出映射到标准品牌

标准品牌列表（与 vehicle-database.csv 一致）：
大众, 别克, 福特, 东风, 长安, 日产, 现代, 一汽, 江淮, 金杯,
福田, 江铃, 哈飞, 雪铁龙, 中华, 雪佛兰, 奥迪, 长城, 标致,
马自达, 铃木, 五菱, 海马, 宝马, 荣威, 比亚迪, 奇瑞, 本田, 丰田, 起亚, 奔驰
"""

# YOLO 原始类别 -> 标准品牌映射
# 注意：YOLO 类别包含车型信息（如"大众家用车"），需要提取品牌名
YOLO_TO_STANDARD_BRAND = {
    # 大众系列
    '大众家用车': '大众',
    '大众出租车': '大众',
    # 别克系列
    '别克商务车': '别克',
    '别克': '别克',
    # 福特系列
    '福特面包车': '福特',
    '福特': '福特',
    # 东风系列
    '东风卡车': '东风',
    '东风面包车': '东风',
    '东风': '东风',
    # 长安系列
    '长安面包车': '长安',
    '长安': '长安',
    # 日产系列
    '日产越野车': '日产',
    '日产': '日产',
    # 现代系列
    '现代越野车': '现代',
    '现代面包车': '现代',
    '现代': '现代',
    # 一汽系列
    '一汽卡车': '一汽',
    '一汽面包车': '一汽',
    '一汽': '一汽',
    # 江淮系列
    '江淮面包车': '江淮',
    '江淮卡车': '江淮',
    '江淮': '江淮',
    # 金杯系列
    '金杯卡车': '金杯',
    '金杯面包车': '金杯',
    # 福田系列
    '福田小卡': '福田',
    '福田大卡': '福田',
    # 黄牌系列（无法确定品牌，按车型处理）
    '黄牌大巴': '其他',
    '黄牌卡车': '其他',
    '公交车': '其他',
    # 其他已存在品牌
    '江铃': '江铃',
    '哈飞': '哈飞',
    '雪铁龙': '雪铁龙',
    '中华': '中华',
    '雪佛兰': '雪佛兰',
    '奥迪': '奥迪',
    '长城': '长城',
    '标致': '标致',
    '马自达': '马自达',
    '铃木': '铃木',
    '五菱': '五菱',
    '海马': '海马',
    '宝马': '宝马',
    '荣威': '荣威',
    '比亚迪': '比亚迪',
    '奇瑞': '奇瑞',
    '本田': '本田',
    '丰田': '丰田',
    '起亚': '起亚',
    '奔驰': '奔驰',
    '其他': '其他',
}

# 品牌 -> 常见车型的对应关系（用于交叉验证）
BRAND_VEHICLE_TYPE_HINTS = {
    '大众': ['car', 'minibus'],
    '别克': ['car', 'minibus'],
    '福特': ['car', 'minibus'],
    '雪佛兰': ['car'],
    '丰田': ['car', 'minibus'],
    '本田': ['car', 'minibus'],
    '日产': ['car', 'minibus'],
    '现代': ['car', 'minibus'],
    '起亚': ['car'],
    '马自达': ['car'],
    '宝马': ['car'],
    '奥迪': ['car', 'minibus'],
    '奔驰': ['car', 'minibus'],
    '标致': ['car'],
    '雪铁龙': ['car'],
    '铃木': ['car', 'minibus'],
    '奇瑞': ['car'],
    '比亚迪': ['car', 'minibus'],
    '长城': ['car', 'minibus'],
    '荣威': ['car'],
    '中华': ['car'],
    '海马': ['car'],
    '长安': ['car', 'minibus', 'truck'],
    '五菱': ['minibus'],
    '东风': ['truck', 'bus', 'minibus'],
    '一汽': ['truck', 'car', 'bus'],
    '江淮': ['truck', 'bus', 'minibus'],
    '福田': ['truck'],
    '金杯': ['minibus', 'truck'],
    '江铃': ['truck'],
    '哈飞': ['minibus'],
}


def correct_brand(yolo_brand, vehicle_type=None, vehicle_color=None):
    """
    将 YOLO 输出的品牌修正为标准品牌

    :param yolo_brand: YOLO 原始输出品牌
    :param vehicle_type: 车型分类结果（bus/car/minibus/truck），用于交叉验证
    :param vehicle_color: 颜色，辅助判断（如出租车通常是绿色/蓝色）
    :return: 修正后的标准品牌
    """
    if not yolo_brand:
        return '其他'

    # 1. 直接映射
    standard = YOLO_TO_STANDARD_BRAND.get(yolo_brand)
    if standard:
        return standard

    # 2. 模糊匹配：检查 yolo_brand 是否包含某个标准品牌名
    for yolo_name, std_name in YOLO_TO_STANDARD_BRAND.items():
        if yolo_name in yolo_brand or yolo_brand in yolo_name:
            return std_name

    # 3. 如果还是找不到，返回原始值（可能是新品牌）
    return yolo_brand


def validate_brand_type_consistency(brand, vehicle_type):
    """
    检查品牌和车型是否一致，不一致时给出警告或修正建议
    :return: (is_consistent, suggested_brand)
    """
    if brand == '其他' or not vehicle_type:
        return True, brand

    expected_types = BRAND_VEHICLE_TYPE_HINTS.get(brand, [])
    if not expected_types:
        return True, brand

    if vehicle_type not in expected_types:
        # 不一致，但不做强制修正（因为有些品牌确实有多种车型）
        return False, brand

    return True, brand


if __name__ == '__main__':
    # 测试映射
    test_cases = [
        ('大众家用车', 'car'),
        ('大众出租车', 'car'),
        ('别克商务车', 'minibus'),
        ('福特面包车', 'minibus'),
        ('东风卡车', 'truck'),
        ('黄牌大巴', 'bus'),
        ('雪佛兰', 'car'),
        ('其他', 'car'),
    ]
    print("品牌修正测试:")
    for yolo_brand, vtype in test_cases:
        corrected = correct_brand(yolo_brand, vtype)
        consistent, _ = validate_brand_type_consistency(corrected, vtype)
        print(f"  {yolo_brand:12s} -> {corrected:8s} (车型={vtype}, 一致={consistent})")
