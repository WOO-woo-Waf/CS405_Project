#!/bin/bash
# Bash script for training FCN16s model

#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan      # 作业提交的指定分区队列为titan
#SBATCH --qos=titan            # 指定作业的QOS
#SBATCH -J TEST       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:1           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 

# 设置使用的 GPU 设备
export CUDA_VISIBLE_DEVICES=0

# 指定训练脚本和参数
python demo.py --model fcn8s_vgg16_voc --input-pic /home/wangdx_lab/cse12210928/.torch/datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png  --outdir ./1111 