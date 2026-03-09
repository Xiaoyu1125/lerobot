#!/usr/bin/env python3
"""
用于循环运行不同 shuffle_buffer_size 参数的训练脚本
shuffle_buffer_size 从 100 到 2000，间隔 100
"""

import subprocess
import sys
from pathlib import Path

OUTPUTS_ROOT = Path.home()/ "git" / "lerobot" / "outputs"

def run_training():

    # 指定验证集 episodes (从数据集中选取几个 episode 作为验证集)
    # 例如: val_episodes = [0, 1, 2]  # 使用前3个 episode 作为验证
    val_episodes = [0, 1, 2]

    base_cmd = [
        "python",
        "src/lerobot/scripts/lerobot_train.py",
        "--policy.type=smolvla",
        "--dataset.repo_id=/data/lerobot/svla_so100_stacking",
        # "--dataset.repo_id=/public/xiaoyu/svla_so100_stacking",
        "--batch_size=4",
        "--steps=200000",
        "--policy.repo_id=~/git/lerobot/lerobot/outputs",
        # "--policy.repo_id=/home/xiaoyu/lerobot/lerobot/outputs",
        "--wandb.enable=true",
        "--wandb.mode=offline",
        "--dataset.streaming=true",
        "--num_workers=1",
        "--dataset.root=/data/lerobot/svla_so100_stacking",
        # "--dataset.root=/public/xiaoyu/svla_so100_stacking",
        "--dataset.video_backend=pyav",
        "--dataset.val_episodes=" + str(val_episodes),
    ]

    buffer_sizes = range(100, 2100, 100)
    total_runs = len(list(buffer_sizes))

    print(f"total {total_runs} training times")
    print(f"outputs root: {OUTPUTS_ROOT}")
    print("-" * 80)

    for idx, buffer_size in enumerate(buffer_sizes, 1):
        run_id = "shuffle"+str(buffer_size)
        output_dir = OUTPUTS_ROOT / run_id

        cmd = base_cmd + [
            f"--dataset.shuffle_buffer_size={buffer_size}",
            f"--wandb.run_id={run_id}",
            f"--output_dir={output_dir}",
        ]

        print(f"\n[{idx}/{total_runs}], shuffle_buffer_size={buffer_size}, output_dir={output_dir}")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n错误: 训练失败 (shuffle_buffer_size={buffer_size})")
            print(f"返回码: {result.returncode}")
            sys.exit(1)

        print(f"✓ 训练完成 (buffer_size={buffer_size})")

    print("所有训练任务完成！")

if __name__ == "__main__":
    run_training()
