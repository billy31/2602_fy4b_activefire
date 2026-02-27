#!/bin/bash
# 训练脚本包装器 - 自动提交代码变更到GitHub

set -e  # 遇到错误立即退出

cd /root/codes/fire0226/selfCodes

echo "======================================"
echo "Fire Detection Training with Git Sync"
echo "======================================"
echo ""

# 获取当前时间戳
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 检查是否有未提交的代码变更
if [ -n "$(git status --porcelain)" ]; then
    echo "📦 发现未提交的代码变更，正在提交..."
    git add -A
    git commit -m "Auto-commit before training @ $TIMESTAMP

Changes:
- Update train_landsat.py with latest modifications
- Sync before starting training session" || true
    echo "✅ 代码已提交到本地"
else
    echo "ℹ️ 没有未提交的代码变更"
fi

# 尝试推送到远程（如果失败不中断）
echo ""
echo "🚀 尝试同步到 GitHub..."
if git push origin main 2>/dev/null; then
    echo "✅ 已成功同步到 GitHub"
else
    echo "⚠️ GitHub 同步失败（可能需要手动认证）"
    echo "   稍后请运行: git push origin main"
fi

echo ""
echo "======================================"
echo "🏃 启动训练..."
echo "======================================"
echo ""

# 运行训练脚本，传递所有参数
python train_landsat.py "$@"

# 获取训练退出码
TRAIN_EXIT_CODE=$?

echo ""
echo "======================================"

# 训练完成后提交结果
if [ -n "$(git status --porcelain)" ]; then
    echo "📦 训练完成，提交结果..."
    git add -A
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        COMMIT_MSG="Post-training: completed successfully @ $TIMESTAMP

Results:
- Training finished with exit code 0
- Model saved to output directory"
    else
        COMMIT_MSG="Post-training: exited with code $TRAIN_EXIT_CODE @ $TIMESTAMP

Note:
- Training encountered issues
- Check logs for details"
    fi
    
    git commit -m "$COMMIT_MSG" || true
    
    # 尝试推送
    echo "🚀 同步训练结果到 GitHub..."
    if git push origin main 2>/dev/null; then
        echo "✅ 结果已同步到 GitHub"
    else
        echo "⚠️ 推送失败，请稍后手动运行: git push origin main"
    fi
else
    echo "ℹ️ 没有新的训练结果需要提交"
fi

echo ""
echo "======================================"
echo "✨ 训练流程完成"
echo "======================================"

exit $TRAIN_EXIT_CODE
