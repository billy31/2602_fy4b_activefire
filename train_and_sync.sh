#!/bin/bash
# 训练脚本包装器 - 自动启动TensorBoard

set -e

cd /root/codes/fire0226/selfCodes

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 保存传入的参数
PY_ARGS="$*"

echo "======================================"
echo "Fire Detection Training Launcher"
echo "======================================"
echo "Timestamp: $TIMESTAMP"
echo "Args: $PY_ARGS"
echo ""

# 1. 启动TensorBoard（后台运行）
echo ""
echo "📊 启动 TensorBoard..."
# 查找并终止已有的tensorboard进程
pkill -f "tensorboard" 2>/dev/null || true
sleep 1

# 启动新的tensorboard
tensorboard --logdir=/root/tf-logs --port=6006 --bind_all &
TENSORBOARD_PID=$!
echo "✅ TensorBoard 已启动 (PID: $TENSORBOARD_PID)"
echo "   访问地址: http://$(hostname -I | awk '{print $1}'):6006"
echo ""

# 2. 运行训练
echo "======================================"
echo "🏃 启动训练..."
echo "======================================"
echo ""

# 捕获训练退出状态
python train_landsat.py $PY_ARGS || TRAIN_EXIT_CODE=$?
TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE:-0}

# 3. 终止TensorBoard
kill $TENSORBOARD_PID 2>/dev/null || true

echo ""
echo "======================================"

# 4. 检查是否需要手动提交训练结果（Python内部已处理大部分提交）
if [ -n "$(git status --porcelain)" ]; then
    echo "📦 发现未提交的变更，自动提交..."
    git add -A
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        COMMIT_MSG="Post-training: completed @ $TIMESTAMP"
    else
        COMMIT_MSG="Post-training: exited with code $TRAIN_EXIT_CODE @ $TIMESTAMP"
    fi
    
    git commit -m "$COMMIT_MSG" || true
    git push origin main 2>/dev/null && echo "✅ 已推送" || echo "⚠️ 推送失败"
fi

echo ""
echo "======================================"
echo "✨ 训练流程完成 (exit code: $TRAIN_EXIT_CODE)"
echo "======================================"

exit $TRAIN_EXIT_CODE
