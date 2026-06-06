#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度提升工具集
包含:
1. 图像增强预处理 (去噪/锐化/对比度) —— 提升车牌识别率
2. 反馈数据修正工具 —— 人工修正错误样本
3. 训练数据集构建 —— 从反馈生成训练数据
4. 快速数据增强 —— 扩充训练样本
"""
import os
import cv2
import numpy as np
import csv
from feedback_db import list_pending_feedback, update_feedback, build_training_dataset, get_feedback_stats


# ============== 1. 图像增强（推理前预处理） ==============

def enhance_image_for_plate(img):
    """
    针对车牌识别的图像增强
    - 对比度增强 (CLAHE)
    - 轻微锐化
    - 去噪
    """
    if img is None:
        return None

    # 转换到 LAB 颜色空间，对 L 通道做 CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 轻微锐化
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # 快速去噪
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    return enhanced


def enhance_plate_region(plate_img):
    """专门针对车牌小图的增强（二值化、resize）"""
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # 放大到固定高度
    h, w = plate_img.shape[:2]
    target_h = 96
    scale = target_h / h
    new_w = int(w * scale)
    resized = cv2.resize(plate_img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 灰度 + 自适应二值化
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # 转回 3 通道
    binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary_3ch


# ============== 2. 反馈数据人工修正工具 ==============

def interactive_correct():
    """
    交互式修正反馈数据
    运行后逐个显示待修正样本，用户输入正确标签
    """
    records = list_pending_feedback(limit=100)
    if not records:
        print("[CORRECT] 没有待修正的反馈数据。请先上传一些图片进行分析。")
        return

    print(f"[CORRECT] 共 {len(records)} 条待修正记录")
    print("提示: 直接回车表示保持原值不变，输入 'skip' 跳过此条，输入 'quit' 退出\n")

    for rec in records:
        fid = rec['id']
        img_path = rec['image_path']
        print(f"\n{'='*50}")
        print(f"ID: {fid}")
        print(f"图片: {img_path}")
        print(f"当前识别 -> 车牌: {rec['plate']} | 品牌: {rec['brand']} | 类型: {rec['v_type']} | 颜色: {rec['v_color']}")

        # 显示图片（如果可以用 cv2）
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            show_w = min(800, w)
            show_h = int(h * show_w / w)
            show_img = cv2.resize(img, (show_w, show_h))
            cv2.imshow(f"Feedback {fid}", show_img)
            cv2.waitKey(200)

        # 交互输入
        c_plate = input(f"  正确车牌 [{rec['plate']}]: ").strip()
        c_brand = input(f"  正确品牌 [{rec['brand']}]: ").strip()
        c_type = input(f"  正确类型 [{rec['v_type']}]: ").strip()
        c_color = input(f"  正确颜色 [{rec['v_color']}]: ").strip()

        if c_plate.lower() == 'quit' or c_brand.lower() == 'quit':
            print("[CORRECT] 已退出")
            break
        if c_plate.lower() == 'skip' or c_brand.lower() == 'skip':
            print("[CORRECT] 已跳过")
            continue

        update_feedback(fid, corrected_plate=c_plate, corrected_brand=c_brand,
                        corrected_type=c_type, corrected_color=c_color)
        print(f"[CORRECT] 已修正: {fid}")

    cv2.destroyAllWindows()
    print("[CORRECT] 修正完成")


def quick_correct(fid, plate='', brand='', v_type='', color=''):
    """快速修正单条记录（命令行参数方式）"""
    ok = update_feedback(fid, corrected_plate=plate, corrected_brand=brand,
                         corrected_type=v_type, corrected_color=color)
    if ok:
        print(f"[CORRECT] 修正成功: {fid}")
    else:
        print(f"[CORRECT] 修正失败，未找到记录: {fid}")


# ============== 3. 数据增强（生成训练样本） ==============

def augment_image(img, num_variants=3):
    """
    对单张图片做数据增强，生成多个变体
    返回: list of augmented images
    """
    aug_imgs = [img.copy()]
    h, w = img.shape[:2]

    for _ in range(num_variants):
        aug = img.copy()

        # 随机亮度
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.randint(-30, 30)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        # 随机水平翻转 (50% 概率)
        if np.random.random() > 0.5:
            aug = cv2.flip(aug, 1)

        # 随机高斯噪声
        noise = np.random.normal(0, 5, aug.shape).astype(np.uint8)
        aug = cv2.add(aug, noise)

        # 随机轻微旋转
        angle = np.random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderValue=(128, 128, 128))

        # 随机饱和度/色调微调 (HSV)
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
        hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        aug_imgs.append(aug)

    return aug_imgs


# ============== 4. 主入口 ==============

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python improve_accuracy.py stats          # 查看反馈统计")
        print("  python improve_accuracy.py correct        # 交互式修正反馈数据")
        print("  python improve_accuracy.py build          # 构建训练数据集")
        print("  python improve_accuracy.py quick FID --plate 京A12345 --brand 大众  # 快速修正")
        print("\n精度提升流程:")
        print("  1. 正常使用系统上传图片分析")
        print("  2. 运行: python improve_accuracy.py correct  # 修正识别错误的样本")
        print("  3. 运行: python improve_accuracy.py build    # 生成训练数据集")
        print("  4. 用生成的数据集 fine-tune 模型")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'stats':
        stats = get_feedback_stats()
        print(f"反馈统计: 总计 {stats['total']}, 待修正 {stats['pending']}, 已修正 {stats['corrected']}")

    elif cmd == 'correct':
        interactive_correct()

    elif cmd == 'build':
        out = build_training_dataset()
        if out:
            print(f"训练数据集已生成: {out}")
        else:
            print("无可用的已修正数据")

    elif cmd == 'quick':
        if len(sys.argv) < 3:
            print("请提供反馈ID")
            sys.exit(1)
        fid = sys.argv[2]
        kwargs = {'plate': '', 'brand': '', 'v_type': '', 'color': ''}
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--plate' and i + 1 < len(sys.argv):
                kwargs['plate'] = sys.argv[i + 1]; i += 2
            elif sys.argv[i] == '--brand' and i + 1 < len(sys.argv):
                kwargs['brand'] = sys.argv[i + 1]; i += 2
            elif sys.argv[i] == '--type' and i + 1 < len(sys.argv):
                kwargs['v_type'] = sys.argv[i + 1]; i += 2
            elif sys.argv[i] == '--color' and i + 1 < len(sys.argv):
                kwargs['color'] = sys.argv[i + 1]; i += 2
            else:
                i += 1
        quick_correct(fid, **kwargs)

    else:
        print(f"未知命令: {cmd}")
