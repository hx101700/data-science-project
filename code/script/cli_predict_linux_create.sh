#!/bin/bash
# 设置严格模式
set -euo pipefail

# ----------- 配置区 -----------
APP_EXE_ROOT="../../executable/cli_predict_linux"
DIST_DIR="$APP_EXE_ROOT"
BUILD_DIR="$APP_EXE_ROOT/build"
# ----------------------------

# 检查是否在Conda环境中
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "错误：未检测到Conda环境，请先激活py311环境"
    echo "使用方法：conda activate py311"
    exit 1
fi

# 检查必要工具
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "错误：未找到命令 $1"
        exit 1
    fi
}

check_command pyinstaller
check_command patchelf
check_command upx

# 目录处理
mkdir -p "$DIST_DIR" "$BUILD_DIR"
rm -rf "$DIST_DIR"/* "$BUILD_DIR"/*

# 调试信息
echo "=== 环境信息 ==="
echo "Python路径: $(which python)"
echo "Python版本: $(python --version 2>&1)"
echo "XGBoost路径: $(python -c "import xgboost; print(xgboost.__file__)")"
echo "================"

# 执行打包
echo "开始构建..."
SECONDS=0  # 计时开始

pyinstaller ../cli_predict_for_linux.spec \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --clean \
    --noconfirm

# 设置可执行权限
chmod +x "$DIST_DIR/cli_predict"

# 结果报告
echo "========================================"
echo "成功构建Linux版本!"
echo "位置: $(realpath "$DIST_DIR/cli_predict")"
echo "大小: $(du -sh "$DIST_DIR/cli_predict" | cut -f1)"
echo "耗时: $SECONDS 秒"
echo "========================================"

# 验证构建
if ! "$DIST_DIR/cli_predict" --version &> /dev/null; then
    echo "警告：生成的二进制文件可能有问题，请检查依赖"
    exit 1
fi