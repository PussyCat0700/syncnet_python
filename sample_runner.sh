#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=AVSync
#SBATCH --partition=RTX3090,RTX4090,A100,ADA6000 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -e
runname=revise
python calculate_scores_LRS.py --data_root /data1/yfliu/outputs/hifigandev/output433h/revise/433h_upsample/test_samples --postfix vc --runname $runname
runname=divise_400k
python calculate_scores_LRS.py --data_root /data1/yfliu/outputs/baseline/433h_8x_fixed/test_samples --postfix vc --runname $runname
runname=divise_gf
python calculate_scores_LRS.py --data_root /data1/yfliu/outputs/baseline/433h_8x_fixed/test_samples --postfix gf --runname $runname