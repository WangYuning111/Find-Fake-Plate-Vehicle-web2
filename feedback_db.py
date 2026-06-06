#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反馈数据管理模块
- 保存每次识别结果和原图，供后续人工修正
- 形成数据闭环，持续优化模型精度
"""
import os
import csv
import shutil
import uuid
from datetime import datetime
from config import Config

FEEDBACK_CSV = os.path.join(os.path.dirname(Config.DATABASE_PATH), "feedback.csv")
FEEDBACK_IMG_DIR = os.path.join(os.path.dirname(Config.DATABASE_PATH), "feedback_images")

os.makedirs(FEEDBACK_IMG_DIR, exist_ok=True)

# CSV 列: id, timestamp, image_path, plate, v_type, v_color, brand, is_fake, corrected_plate, corrected_brand, corrected_type, corrected_color, status

def init_feedback_csv():
    if not os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'timestamp', 'image_path', 'plate', 'v_type', 'v_color', 'brand',
                             'is_fake', 'corrected_plate', 'corrected_brand', 'corrected_type',
                             'corrected_color', 'status'])

def save_feedback(image_path, plate, v_type, v_color, brand, is_fake):
    """保存一次识别结果到反馈库"""
    init_feedback_csv()
    fid = str(uuid.uuid4())[:8]
    ext = os.path.splitext(image_path)[1]
    dst_name = f"{fid}{ext}"
    dst_path = os.path.join(FEEDBACK_IMG_DIR, dst_name)
    shutil.copy(image_path, dst_path)

    with open(FEEDBACK_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            fid, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), dst_path,
            plate, v_type, v_color, brand, int(is_fake),
            '', '', '', '', 'pending'
        ])
    return fid


def list_pending_feedback(limit=50):
    """列出待修正的反馈记录"""
    if not os.path.exists(FEEDBACK_CSV):
        return []
    records = []
    with open(FEEDBACK_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'pending':
                records.append(row)
                if len(records) >= limit:
                    break
    return records


def update_feedback(fid, corrected_plate='', corrected_brand='', corrected_type='', corrected_color=''):
    """人工修正一条反馈记录"""
    if not os.path.exists(FEEDBACK_CSV):
        return False

    rows = []
    updated = False
    with open(FEEDBACK_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['id'] == fid:
                if corrected_plate:
                    row['corrected_plate'] = corrected_plate
                if corrected_brand:
                    row['corrected_brand'] = corrected_brand
                if corrected_type:
                    row['corrected_type'] = corrected_type
                if corrected_color:
                    row['corrected_color'] = corrected_color
                # 如果任一字段被修正，标记为 corrected
                if any([corrected_plate, corrected_brand, corrected_type, corrected_color]):
                    row['status'] = 'corrected'
                updated = True
            rows.append(row)

    if updated:
        with open(FEEDBACK_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return updated


def get_feedback_stats():
    """获取反馈统计"""
    if not os.path.exists(FEEDBACK_CSV):
        return {'total': 0, 'pending': 0, 'corrected': 0}
    total = pending = corrected = 0
    with open(FEEDBACK_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get('status') == 'pending':
                pending += 1
            elif row.get('status') == 'corrected':
                corrected += 1
    return {'total': total, 'pending': pending, 'corrected': corrected}


def build_training_dataset(output_dir='datasets/improved'):
    """
    将已修正的反馈数据构建为训练数据集
    输出结构:
        output_dir/
            images/
            labels.csv  (image_path, plate, brand, type, color)
    """
    if not os.path.exists(FEEDBACK_CSV):
        print("[FEEDBACK] 暂无反馈数据")
        return None

    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    label_path = os.path.join(output_dir, 'labels.csv')

    count = 0
    with open(FEEDBACK_CSV, 'r', encoding='utf-8') as f_in, \
         open(label_path, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(['image_path', 'plate', 'brand', 'type', 'color'])

        for row in reader:
            if row.get('status') != 'corrected':
                continue
            src_img = row['image_path']
            if not os.path.exists(src_img):
                continue
            # 使用修正值，若未修正则使用原始值
            plate = row.get('corrected_plate') or row['plate']
            brand = row.get('corrected_brand') or row['brand']
            v_type = row.get('corrected_type') or row['v_type']
            color = row.get('corrected_color') or row['v_color']

            dst_img = os.path.join(img_dir, os.path.basename(src_img))
            if not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)
            writer.writerow([dst_img, plate, brand, v_type, color])
            count += 1

    print(f"[FEEDBACK] 构建训练数据集完成: {count} 条记录 -> {output_dir}")
    return output_dir


if __name__ == '__main__':
    print("反馈数据管理模块")
    stats = get_feedback_stats()
    print(f"统计: 总计 {stats['total']}, 待修正 {stats['pending']}, 已修正 {stats['corrected']}")
