import time
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

import os
import glob
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_shuffle_csv(output_dir="outputs/shuffle_debug"):
    csv_paths = glob.glob(os.path.join(output_dir, "shuffle_buffer*_shards*.csv"))

    for csv_path in csv_paths:
        png_path = csv_path.replace(".csv", ".png")

        if os.path.exists(png_path):
            print(f"Skip (exists): {png_path}")
            continue
        xs, ys = [], []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(int(row["output_sample_i"]))
                ys.append(int(row["data_index"]))

        if len(xs) == 0:
            continue


        plt.figure(figsize=(12, 6))
        plt.scatter(xs, ys, s=1)
        plt.xlabel("Output sample i after shuffle buffer")
        plt.ylabel("Original dataset index")
        plt.title(os.path.basename(csv_path))
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        print(f"Saved plot: {png_path}")


def plot_shuffle_csv_combined(output_dir="outputs/shuffle_debug", pattern="shuffle_buffer1_shards*.csv"):
    csv_paths = sorted(glob.glob(os.path.join(output_dir, pattern)))

    if not csv_paths:
        print(f"No CSV files found for pattern: {os.path.join(output_dir, pattern)}")
        return

    output_name = pattern.replace("*", "all").replace(".csv", "_combined.png")
    output_path = os.path.join(output_dir, output_name)

    plt.figure(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab10", len(csv_paths))

    for idx, csv_path in enumerate(csv_paths):
        xs, ys = [], []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(int(row["output_sample_i"]))
                ys.append(int(row["data_index"]))

        if not xs:
            continue

        label = os.path.splitext(os.path.basename(csv_path))[0]
        plt.scatter(xs, ys, s=4, color=cmap(idx), alpha=0.8, label=label)

    plt.xlabel("Global output sample i after shuffle buffer")
    plt.ylabel("Original dataset index")
    plt.title("Shuffle buffer output distribution across files")
    plt.xlim(0, 4000)
    plt.ylim(0, 3000)
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved combined plot: {output_path}")

def main():
    # ===== SmolVLA policy config =====
    policy_cfg = SmolVLAConfig()

    # SmolVLA:
    # observation_delta_indices = [0]
    # action_delta_indices = [0, 1, ..., 49]
    # reward_delta_indices = None
    delta_timestamps = {
        "observation.images.image": [0],
        "observation.images.image2": [0],
        "observation.state": [0],
        "action": list(range(50)),
    }

    if policy_cfg.reward_delta_indices is not None:
        delta_timestamps["reward"] = policy_cfg.reward_delta_indices

    # ===== dataloader-only debug args =====
    dataset_repo_id = "/public/xiaoyu/le_libero"
    dataset_root = "/public/xiaoyu/le_libero"

    num_workers = 8
    batch_size = 16
    max_steps = 1000
    max_num_shards = [8]
    shuffle_buffer_sizes = [1]

    for max_num_shard in max_num_shards:
        for shuffle_buffer_size in shuffle_buffer_sizes:
            dataset = StreamingLeRobotDataset(
                dataset_repo_id,
                root=dataset_root,
                episodes=None,
                delta_timestamps=delta_timestamps,
                image_transforms=None,
                revision=None,
                max_num_shards=max_num_shard,
                tolerance_s=1e-4,
                buffer_size=shuffle_buffer_size,
            )

            dataset_streaming = True
            sampler = None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                shuffle=(sampler is None) and (not dataset_streaming),
                sampler=sampler,
                pin_memory=device.type == "cuda",
                drop_last=False,
                prefetch_factor=2 if num_workers > 0 else None,
                collate_fn=None,
            )
            start = time.time()

            for step, batch in enumerate(dataloader):
                if step == 0:
                    print("First batch keys:")
                    print(batch.keys())

                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(k, v.shape, v.dtype)
                        else:
                            print(k, type(v))

                if step % 50 == 0:
                    print(f"[step {step}]")

                if step >= max_steps:
                    break
            plot_shuffle_csv("outputs/shuffle_debug")
            # plot_shuffle_csv_combined("outputs/shuffle_debug")
            print("Finished dataloader-only run.")
            print(f"Time cost: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
