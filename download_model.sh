#!/usr/bin/env bash
# ============================================================
# 模型权重一键获取脚本
# 优先级: 1) 本地 cfg/ 复制  2) GitHub Release 下载
# 支持: wget / curl
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
CFG_DIR="${SCRIPT_DIR}/cfg"
mkdir -p "${WEIGHTS_DIR}"

echo "=========================================="
echo "  套牌车检测系统 - 模型权重获取"
echo "=========================================="

# 下载函数：优先 wget，备选 curl
download_file() {
    local url="$1"
    local output="$2"
    if command -v wget &> /dev/null; then
        wget --show-progress -O "${output}" "${url}"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "${output}" "${url}"
    else
        echo "错误: 未找到 wget 或 curl，请手动下载"
        return 1
    fi
}

# 模型文件列表
MODELS=(
    "best.pt"
    "vehicle_type.pth"
    "vehicle_color.pth"
    "vehicle_brand_resnet18.pth"
)

# GitHub Release 直链（创建 Release 后自动生效）
BASE_URL="https://github.com/WangYuning111/Find-Fake-Plate-Vehicle-web/releases/download/v1.0.0"

all_ok=true
for model in "${MODELS[@]}"; do
    dest="${WEIGHTS_DIR}/${model}"
    if [[ -f "${dest}" ]]; then
        echo "[跳过] ${model} 已存在"
        continue
    fi

    # 策略1: 从本地 cfg/ 复制（训练产出的模型）
    src="${CFG_DIR}/${model}"
    if [[ -f "${src}" ]]; then
        echo "[复制] ${model} (从 cfg/)"
        cp "${src}" "${dest}"
        continue
    fi

    # 策略2: 从 GitHub Release 下载
    echo "[下载] ${model} ..."
    if download_file "${BASE_URL}/${model}" "${dest}"; then
        echo "[完成] ${model}"
    else
        echo "[失败] ${model} 下载失败"
        all_ok=false
    fi
done

echo "=========================================="
if ${all_ok}; then
    echo "  所有模型就绪"
else
    echo "  部分模型缺失，请检查上方错误信息"
    echo "  手动下载地址: ${BASE_URL}"
fi
echo "  存放目录: ${WEIGHTS_DIR}"
echo "=========================================="
