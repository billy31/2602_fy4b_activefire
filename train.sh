#!/bin/bash
# 简化的训练启动脚本 - 推荐使用

cd /root/codes/fire0226/selfCodes

# 默认参数
REGION="${1:-Asia1}"

echo "======================================"
echo "Fire Detection Training"
echo "======================================"
echo "Region: $REGION"
echo ""

# 启动TensorBoard（后台）
pkill -f "tensorboard" 2>/dev/null || true
sleep 1
tensorboard --logdir=/root/tf-logs --port=6006 --bind_all &
echo "TensorBoard started at http://$(hostname -I | awk '{print $1}'):6006"
echo ""

# 运行训练 - 使用优化后的参数
python train_landsat.py "$REGION" \
    --model mamba_vision_S \
    --pretrained \
    --bands 7 6 2 \
    --batch-size 16 \
    --epochs 100 \
    --lr 5e-5 \
    --warmup-epochs 10 \
    --freeze-backbone-epochs 10 \
    --min-fg-pixels 50 \
    --early-stop-patience 3 \
    --early-stop-min-f1 40.0 \
    --use-amp \
    --tensorboard \
    --visualize

# 训练结束，终止TensorBoard
pkill -f "tensorboard" 2>/dev/null || true

echo ""
echo "Training completed!"
