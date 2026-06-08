#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动创建 GitHub Release 并上传模型权重
用法:
    python upload_release.py --token YOUR_GITHUB_TOKEN

如果没有 Token，请前往:
    https://github.com/settings/tokens/new
勾选 "repo" 权限，生成后复制 Token。
"""

import os
import sys
import argparse
import requests

REPO = "WangYuning111/Find-Fake-Plate-Vehicle-web"
RELEASE_TAG = "v1.0.0"
RELEASE_NAME = "v1.0.0 模型权重"
RELEASE_BODY = """模型权重文件（首次运行必需）

包含:
- best.pt — YOLO 车辆检测模型
- vehicle_type.pth — 车型分类模型 (MiniVGGNet, 4类)
- vehicle_color.pth — 颜色分类模型 (MiniVGGNet, 8色)
- vehicle_brand_resnet18.pth — 品牌分类模型 (ResNet18, 26品牌)

下载方式:
```bash
# Linux / macOS / Git Bash
bash download_model.sh

# Windows
python download_models.py
```

> 注意：模型权重文件较大，未纳入源代码仓库，请从本 Release 下载。
> 数据集图片请从网盘或另行获取，也不在仓库中。
"""

MODELS = [
    ("weights/best.pt", "best.pt"),
    ("weights/vehicle_type.pth", "vehicle_type.pth"),
    ("weights/vehicle_color.pth", "vehicle_color.pth"),
    ("cfg/vehicle_brand_resnet18.pth", "vehicle_brand_resnet18.pth"),
]


def get_release_by_tag(token, tag):
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    return None


def create_release(token):
    url = f"https://api.github.com/repos/{REPO}/releases"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    data = {
        "tag_name": RELEASE_TAG,
        "name": RELEASE_NAME,
        "body": RELEASE_BODY,
        "draft": False,
        "prerelease": False,
    }
    r = requests.post(url, headers=headers, json=data)
    if r.status_code == 201:
        print(f"[OK] Release {RELEASE_TAG} 创建成功")
        return r.json()
    else:
        print(f"[ERROR] 创建 Release 失败: {r.status_code}")
        print(r.json())
        sys.exit(1)


def upload_asset(token, release, file_path, asset_name):
    upload_url = release["upload_url"].replace("{?name,label}", f"?name={asset_name}")
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/octet-stream",
    }
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"[上传] {asset_name} ({size_mb:.1f} MB) ...", end=" ", flush=True)
    with open(file_path, "rb") as f:
        r = requests.post(upload_url, headers=headers, data=f)
    if r.status_code == 201:
        print("OK")
        return True
    else:
        print(f"失败 ({r.status_code})")
        print(r.json())
        return False


def main():
    parser = argparse.ArgumentParser(description="上传模型权重到 GitHub Release")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    args = parser.parse_args()

    token = args.token

    # 检查模型文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for local_path, _ in MODELS:
        full_path = os.path.join(base_dir, local_path)
        if not os.path.exists(full_path):
            # 尝试从 cfg/ 找
            alt_path = os.path.join(base_dir, "cfg", os.path.basename(local_path))
            if os.path.exists(alt_path):
                print(f"[提示] 将使用 cfg/{os.path.basename(local_path)} 上传")
            else:
                print(f"[错误] 找不到模型文件: {local_path}")
                sys.exit(1)

    # 获取或创建 Release
    release = get_release_by_tag(token, RELEASE_TAG)
    if release:
        print(f"[提示] Release {RELEASE_TAG} 已存在，将追加上传")
    else:
        release = create_release(token)

    # 上传模型
    all_ok = True
    for local_path, asset_name in MODELS:
        full_path = os.path.join(base_dir, local_path)
        if not os.path.exists(full_path):
            alt_path = os.path.join(base_dir, "cfg", os.path.basename(local_path))
            if os.path.exists(alt_path):
                full_path = alt_path
        if not upload_asset(token, release, full_path, asset_name):
            all_ok = False

    if all_ok:
        print("\n[完成] 所有模型已上传到 Release，download_model.sh 可直接使用")
    else:
        print("\n[警告] 部分上传失败，请检查上方错误")


if __name__ == "__main__":
    main()
