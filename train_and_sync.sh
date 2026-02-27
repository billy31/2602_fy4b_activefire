#!/bin/bash
# 训练脚本包装器 - 自动启动TensorBoard并同步到GitHub

set -e

cd /root/codes/fire0226/selfCodes

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
EXP_NAME="fire_$(date '+%Y%m%d_%H%M%S')"

echo "======================================"
echo "Fire Detection Training with Auto-Sync"
echo "======================================"
echo "Experiment: $EXP_NAME"
echo ""

# 1. 自动提交代码变更
echo "📦 提交代码变更..."
git add -A
git commit -m "Auto-commit before training @ $TIMESTAMP

- Starting new training session
- Config: $@" || true

# 2. 推送到GitHub
echo ""
echo "🚀 推送到 GitHub..."
if git push origin main 2>/dev/null; then
    echo "✅ 已同步到 GitHub"
else
    echo "⚠️ 推送失败，稍后请手动运行: git push origin main"
fi

# 3. 启动TensorBoard（后台运行）
echo ""
echo "📊 启动 TensorBoard..."
# 查找并终止已有的tensorboard进程
pkill -f "tensorboard" 2>/dev/null || true
sleep 1

# 启动新的tensorboard，指向最新的日志目录
tensorboard --logdir=/root/tf-logs --port=6006 --bind_all &
TENSORBOARD_PID=$!
echo "✅ TensorBoard 已启动 (PID: $TENSORBOARD_PID)"
echo "   访问地址: http://$(hostname -I | awk '{print $1}'):6006"
echo ""

# 4. 运行训练
echo "======================================"
echo "🏃 启动训练..."
echo "======================================"
echo ""

# 捕获训练退出状态
python train_landsat.py "$@" || TRAIN_EXIT_CODE=$?
TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE:-0}

# 5. 终止TensorBoard
kill $TENSORBOARD_PID 2>/dev/null || true

echo ""
echo "======================================"

# 6. 训练后提交结果
if [ -n "$(git status --porcelain)" ]; then
    echo "📦 提交训练结果..."
    git add -A
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        COMMIT_MSG="Post-training: completed @ $TIMESTAMP

Training finished successfully
Args: $@"
    else
        COMMIT_MSG="Post-training: exited with code $TRAIN_EXIT_CODE @ $TIMESTAMP

Training encountered issues
Args: $@"
    fi
    
    git commit -m "$COMMIT_MSG" || true
    
    echo "🚀 推送结果到 GitHub..."
    git push origin main 2>/dev/null && echo "✅ 已推送" || echo "⚠️ 推送失败"
fi

echo ""
echo "======================================"
echo "✨ 训练流程完成"
echo "======================================"

exit $TRAIN_EXIT_CODE
