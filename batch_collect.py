#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量数据采集工具
- 从本地图片文件夹批量分析
- 自动保存到反馈数据库
- 支持按子文件夹自动标注（子文件夹名作为标签）

用法:
    python batch_collect.py --src "D:/车辆图片"
    python batch_collect.py --src "D:/车辆图片" --label-folder  # 子文件夹名作为品牌标签
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import load_models, predict_single
from feedback_db import save_feedback


def collect_from_folder(src_dir, use_folder_label=False):
    """
    从文件夹批量采集数据
    :param src_dir: 图片文件夹路径
    :param use_folder_label: 是否使用子文件夹名作为品牌标签
    """
    models = load_models()
    print(f"[COLLECT] 开始采集: {src_dir}")

    # 支持的图片格式
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if use_folder_label:
        # 遍历子文件夹
        for subdir in sorted(Path(src_dir).iterdir()):
            if not subdir.is_dir():
                continue
            brand_label = subdir.name
            print(f"\n[COLLECT] 处理品牌文件夹: {brand_label}")
            images = [f for f in subdir.iterdir() if f.suffix.lower() in exts]
            for img_path in images:
                try:
                    result = predict_single(str(img_path))
                    if result['error']:
                        print(f"  [SKIP] {img_path.name}: {result['error']}")
                        continue
                    # 强制将品牌设为文件夹名
                    result['vehicle_brand'] = brand_label
                    fid = save_feedback(str(img_path), result['plate_number'],
                                        result['vehicle_type'], result['vehicle_color'],
                                        brand_label, result['is_fake'])
                    print(f"  [OK] {img_path.name} -> {result['plate_number']} ({brand_label})")
                except Exception as e:
                    print(f"  [ERR] {img_path.name}: {e}")
    else:
        # 直接遍历所有图片
        images = [f for f in Path(src_dir).iterdir() if f.suffix.lower() in exts]
        print(f"[COLLECT] 共找到 {len(images)} 张图片")
        for img_path in images:
            try:
                result = predict_single(str(img_path))
                if result['error']:
                    print(f"  [SKIP] {img_path.name}: {result['error']}")
                    continue
                fid = save_feedback(str(img_path), result['plate_number'],
                                    result['vehicle_type'], result['vehicle_color'],
                                    result['vehicle_brand'], result['is_fake'])
                print(f"  [OK] {img_path.name} -> {result['plate_number']} | {result['vehicle_brand']} | {result['vehicle_type']} | {result['vehicle_color']}")
            except Exception as e:
                print(f"  [ERR] {img_path.name}: {e}")

    print("\n[COLLECT] 采集完成！请访问网页 '数据修正' 页面核对并修正标签。")


def auto_label_plate_dataset(src_dir, brand_hint=None):
    """
    专用于车牌识别数据集构建
    假设图片文件名就是车牌号（如 京A12345.jpg）
    """
    models = load_models()
    print(f"[PLATE-DATA] 构建车牌数据集: {src_dir}")
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    correct = 0
    wrong = 0
    for img_path in Path(src_dir).iterdir():
        if img_path.suffix.lower() not in exts:
            continue
        # 从文件名提取真实车牌
        true_plate = img_path.stem.upper().replace('_', '').replace('-', '')
        try:
            result = predict_single(str(img_path))
            pred_plate = result.get('plate_number', '')
            if result['error']:
                print(f"  [ERR] {img_path.name}: 识别失败")
                wrong += 1
                continue

            # 保存到反馈库
            fid = save_feedback(str(img_path), pred_plate,
                                result['vehicle_type'], result['vehicle_color'],
                                result['vehicle_brand'], result['is_fake'])

            if pred_plate == true_plate:
                print(f"  [OK] {img_path.name}: {pred_plate} ✓")
                correct += 1
            else:
                print(f"  [WRONG] {img_path.name}: 真实={true_plate}, 预测={pred_plate}")
                wrong += 1
        except Exception as e:
            print(f"  [ERR] {img_path.name}: {e}")
            wrong += 1

    total = correct + wrong
    acc = correct / total * 100 if total > 0 else 0
    print(f"\n[PLATE-DATA] 车牌识别准确率: {correct}/{total} ({acc:.1f}%)")
    print("[PLATE-DATA] 请在'数据修正'页面修正错误样本，然后运行 build_training_dataset()")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量数据采集工具')
    parser.add_argument('--src', required=True, help='图片文件夹路径')
    parser.add_argument('--label-folder', action='store_true',
                        help='使用子文件夹名作为品牌标签（用于品牌分类数据集）')
    parser.add_argument('--plate-mode', action='store_true',
                        help='车牌数据集模式（文件名作为真实车牌）')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"[ERR] 路径不存在: {args.src}")
        sys.exit(1)

    if args.plate_mode:
        auto_label_plate_dataset(args.src)
    else:
        collect_from_folder(args.src, use_folder_label=args.label_folder)
