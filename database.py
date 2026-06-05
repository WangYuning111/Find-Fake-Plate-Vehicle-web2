#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite 数据库管理
- 识别历史记录
- 车辆信息 CRUD
"""
import os
import sqlite3
import json
import time
from datetime import datetime
from threading import Lock
from config import Config

# 线程锁，防止并发写入
_db_lock = Lock()


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(Config.DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH) or '.', exist_ok=True)
    with get_db() as conn:
        # 识别历史记录表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                plate_number TEXT,
                vehicle_type TEXT,
                vehicle_color TEXT,
                vehicle_brand TEXT,
                is_fake INTEGER DEFAULT 0,
                true_brand TEXT,
                confidence REAL,
                result_image TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 车辆信息表（替代 CSV，支持增删改查）
        conn.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_no TEXT UNIQUE NOT NULL,
                car_brand TEXT NOT NULL,
                car_type TEXT,
                owner_name TEXT,
                registration_date TEXT,
                status TEXT DEFAULT '正常',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        print("[DB] 数据库初始化完成")


def import_csv_to_db(csv_path=None):
    """将 CSV 中的车辆数据导入 SQLite"""
    import pandas as pd
    csv_path = csv_path or Config.VEHICLE_CSV
    if not os.path.exists(csv_path):
        print(f"[DB] CSV 文件不存在: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    with _db_lock, get_db() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute('''
                    INSERT OR IGNORE INTO vehicle_info (plate_no, car_brand)
                    VALUES (?, ?)
                ''', (str(row['plateNo']).strip(), str(row['carBrand']).strip()))
            except Exception as e:
                print(f"[DB] 导入行失败: {row}, 错误: {e}")
        conn.commit()
    print(f"[DB] CSV 数据导入完成: {csv_path}")


def add_record(image_path, plate, v_type, v_color, brand, is_fake, true_brand, confidence, result_image):
    """添加识别记录"""
    with _db_lock, get_db() as conn:
        conn.execute('''
            INSERT INTO detection_records
            (image_path, plate_number, vehicle_type, vehicle_color, vehicle_brand,
             is_fake, true_brand, confidence, result_image, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image_path, plate, v_type, v_color, brand,
              1 if is_fake else 0, true_brand, confidence, result_image,
              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()


def get_records(limit=50, offset=0):
    """分页查询识别记录"""
    with get_db() as conn:
        rows = conn.execute('''
            SELECT * FROM detection_records
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset)).fetchall()
        return [dict(row) for row in rows]


def get_record_count():
    """获取记录总数"""
    with get_db() as conn:
        row = conn.execute('SELECT COUNT(*) as cnt FROM detection_records').fetchone()
        return row['cnt'] if row else 0


def get_stats():
    """获取统计数据"""
    with get_db() as conn:
        total = conn.execute('SELECT COUNT(*) as cnt FROM detection_records').fetchone()['cnt']
        fake = conn.execute('SELECT COUNT(*) as cnt FROM detection_records WHERE is_fake=1').fetchone()['cnt']
        today = datetime.now().strftime('%Y-%m-%d')
        today_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM detection_records WHERE created_at LIKE ?",
            (f'{today}%',)
        ).fetchone()['cnt']
        return {'total': total, 'fake': fake, 'today': today_count}


# ============== 车辆信息 CRUD ==============

def get_vehicle_by_plate(plate_no):
    """根据车牌查询车辆信息"""
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM vehicle_info WHERE plate_no = ?',
            (plate_no,)
        ).fetchone()
        return dict(row) if row else None


def add_vehicle(plate_no, car_brand, car_type=None, owner_name=None, registration_date=None, status='正常'):
    """添加车辆信息"""
    with _db_lock, get_db() as conn:
        try:
            conn.execute('''
                INSERT INTO vehicle_info (plate_no, car_brand, car_type, owner_name, registration_date, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (plate_no, car_brand, car_type, owner_name, registration_date, status,
                  datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def update_vehicle(plate_no, **kwargs):
    """更新车辆信息"""
    allowed = {'car_brand', 'car_type', 'owner_name', 'registration_date', 'status'}
    fields = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not fields:
        return False

    fields['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    set_clause = ', '.join(f'{k}=?' for k in fields.keys())
    values = list(fields.values()) + [plate_no]

    with _db_lock, get_db() as conn:
        conn.execute(f'UPDATE vehicle_info SET {set_clause} WHERE plate_no=?', values)
        conn.commit()
        return conn.total_changes > 0


def delete_vehicle(plate_no):
    """删除车辆信息"""
    with _db_lock, get_db() as conn:
        conn.execute('DELETE FROM vehicle_info WHERE plate_no = ?', (plate_no,))
        conn.commit()
        return conn.total_changes > 0


def list_vehicles(search=None, limit=100, offset=0):
    """分页查询车辆列表"""
    with get_db() as conn:
        if search:
            like = f'%{search}%'
            rows = conn.execute('''
                SELECT * FROM vehicle_info
                WHERE plate_no LIKE ? OR car_brand LIKE ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            ''', (like, like, limit, offset)).fetchall()
            total = conn.execute('''
                SELECT COUNT(*) as cnt FROM vehicle_info
                WHERE plate_no LIKE ? OR car_brand LIKE ?
            ''', (like, like)).fetchone()['cnt']
        else:
            rows = conn.execute('''
                SELECT * FROM vehicle_info
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset)).fetchall()
            total = conn.execute('SELECT COUNT(*) as cnt FROM vehicle_info').fetchone()['cnt']
        return [dict(row) for row in rows], total


def check_fake_plate(plate_no, detected_brand):
    """
    套牌检测：对比数据库中的真实品牌
    返回: (is_fake, true_brand, vehicle_info)
    """
    if detected_brand == '未知' or not detected_brand:
        return False, None, None

    vehicle = get_vehicle_by_plate(plate_no)
    if vehicle is None:
        return False, None, None

    true_brand = vehicle['car_brand']
    # 支持品牌别名匹配，例如"大众家用车"匹配"大众"
    is_match = detected_brand in true_brand or true_brand in detected_brand
    is_fake = not is_match
    return is_fake, true_brand, vehicle


# 初始化
init_db()
import_csv_to_db()
