#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型权重一键下载脚本（Windows / Linux / macOS 通用）
支持从 GitHub Release 自动下载所有模型权重文件

用法:
    python download_models.py

如果没有找到模型文件，会自动从 GitHub Release 下载。
"""

import os
import sys
import urllib.request
import urllib.error

# ==================== 配置 ====================
REPO_OWNER = "WangYuning111"
REPO_NAME = "Find-Fake-Plate-Vehicle-web"
RELEASE_TAG = "v1.0.0"
BASE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}"

# 模型文件列表: (Release中的文件名, 本地保存路径)
MODELS = [
    ("best.pt", "weights/best.pt"),
    ("vehicle_type.pth", "weights/vehicle_type.pth"),
    ("vehicle_color.pth", "weights/vehicle_color.pth"),
    ("vehicle_brand_resnet18.pth", "weights/vehicle_brand_resnet18.pth"),
]

# 本地 cfg/ 目录中可能的备用来源
CFG_DIR = "cfg"


def download_file(url, output_path):
    """下载文件并显示进度"""
    try:
        print(f"  [下载] {os.path.basename(output_path)} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"OK ({size_kb:.1f} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"失败 (HTTP {e.code})")
        return False
    except Exception as e:
        print(f"失败 ({e})")
        return False


def copy_file(src, dst):
    """复制文件"""
    try:
        import shutil
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"  [复制] {os.path.basename(dst)} (从 cfg/) OK ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [复制] 失败: {e}")
        return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 50)
    print("  套牌车检测系统 - 模型权重获取")
    print("=" * 50)

    # 确保 weights/ 目录存在
    os.makedirs("weights", exist_ok=True)

    all_ok = True
    for asset_name, local_path in MODELS:
        # 如果已存在则跳过
        if os.path.exists(local_path):
            size_kb = os.path.getsize(local_path) / 1024
            print(f"  [跳过] {asset_name} 已存在 ({size_kb:.1f} KB)")
            continue

        # 策略1: 从本地 cfg/ 复制（训练产出的模型）
        cfg_path = os.path.join(CFG_DIR, asset_name)
        if os.path.exists(cfg_path):
            copy_file(cfg_path, local_path)
            continue

        # 策略2: 从 GitHub Release 下载
        url = f"{BASE_URL}/{asset_name}"
        if not download_file(url, local_path):
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("  所有模型就绪")
    else:
        print("  部分模型缺失，请检查上方错误信息")
        print(f"  手动下载地址: https://github.com/{REPO_OWNER}/{REPO_NAME}/releases")
    print("=" * 50)

    # 显示最终文件列表
    print("\n  本地模型文件:")
    for _, local_path in MODELS:
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"    {local_path:40s} {size_mb:6.1f} MB")
        else:
            print(f"    {local_path:40s} [缺失]")


if __name__ == "__main__":
    main()
