#!/bin/bash
#SBATCH -o job.%j.out          # 输出日志保存路径
#SBATCH --partition=titan      # 指定分区队列
#SBATCH --qos=titan            # 指定QOS
#SBATCH -J myDistributedJob    # 作业名称
#SBATCH --nodes=1              # 申请1个节点
#SBATCH --ntasks-per-node=6    # 每个节点运行6个任务
#SBATCH --gres=gpu:4           # 申请4张GPU

# 设置使用的 GPU 设备数量
export NGPUS=4  # 设置使用的 GPU 数量
export MASTER_ADDR=$(hostname)  # 设置分布式主节点的地址
export MASTER_PORT=12345        # 设置分布式主节点的端口

# 使用 PyTorch 分布式工具运行训练脚本
torchrun  --nproc_per_node=$NGPUS train.py \
    --model fcn8s \
    --backbone vgg16 \
    --dataset citys \
    --lr 0.001 \
    --epochs 40 \
    --skip-val 
