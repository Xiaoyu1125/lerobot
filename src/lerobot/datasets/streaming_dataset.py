#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    Backtrackable,
    LookAheadError,
    LookBackError,
    check_version_compatibility,
    find_float_index,
    get_delta_indices,
    is_float_in_list,
    item_to_torch,
    safe_shard,
)
from lerobot.datasets.video_utils import (
    VideoDecoderCache,
)
from lerobot.utils.constants import HF_LEROBOT_HOME, LOOKAHEAD_BACKTRACKTABLE, LOOKBACK_BACKTRACKTABLE


import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class ShuffleOutputRecorder:
    """
    Record output order after shuffle buffer.

    x-axis: global output sample i after shuffle buffer
    y-axis: original dataset index from make_frame(): result["index"]
    """

    def __init__(
        self,
        enable: bool = False,
        output_dir: str = "outputs/shuffle_debug",
        filename_prefix: str = "shuffle_output",
        worker_id: int | None = None,
        num_workers: int = 1,
        max_points: int | None = 200_000,
        index_key: str = "index",
        flush_every: int = 1000,
    ):
        self.enable = enable
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.worker_id = worker_id
        self.num_workers = max(1, num_workers)
        self.max_points = max_points
        self.index_key = index_key
        self.flush_every = flush_every

        self.worker_output_sample_i = 0
        self.rows = []

        if self.enable:
            os.makedirs(self.output_dir, exist_ok=True)

            self.csv_path = os.path.join(
                self.output_dir,
                f"{self.filename_prefix}.csv",
            )
            self.png_path = os.path.join(
                self.output_dir,
                f"{self.filename_prefix}.png",
            )

            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["output_sample_i", "worker_output_sample_i", "data_index"])

    def _worker_suffix(self) -> str:
        return f" [worker {self.worker_id}]" if self.worker_id is not None else ""

    def _global_output_sample_i(self) -> int:
        if self.worker_id is None or self.num_workers <= 1:
            return self.worker_output_sample_i
        return self.worker_output_sample_i * self.num_workers + self.worker_id

    def _to_int(self, value):
        """
        Convert torch.Tensor / numpy scalar / python int to int.
        make_frame() gives result["index"] from item["index"].
        """
        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "item"):
            value = value.item()

        return int(value)

    def _get_data_index(self, sample):
        """
        According to make_frame(), sample should contain:

            sample["index"]

        This is the original dataset index.
        """
        if self.index_key not in sample:
            return None

        return self._to_int(sample[self.index_key])

    def record(self, sample):
        if not self.enable:
            return

        if self.max_points is not None and self.worker_output_sample_i >= self.max_points:
            self.worker_output_sample_i += 1
            return

        data_index = self._get_data_index(sample)

        if data_index is None:
            self.worker_output_sample_i += 1
            return

        self.rows.append((self._global_output_sample_i(), self.worker_output_sample_i, data_index))
        self.worker_output_sample_i += 1

        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.enable or len(self.rows) == 0:
            return

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)

        self.rows.clear()

    def save_plot(self):
        if not self.enable:
            return

        self.flush()

        xs = []
        ys = []

        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(int(row["output_sample_i"]))
                ys.append(int(row["data_index"]))

        if len(xs) == 0:
            print("[ShuffleOutputRecorder] No data recorded.")
            return

        plt.figure(figsize=(12, 6))
        plt.scatter(xs, ys, s=1)
        plt.xlabel("Global output sample i after shuffle buffer")
        plt.ylabel("Original dataset index")
        plt.title(f"Shuffle buffer output distribution{self._worker_suffix()}")
        plt.tight_layout()
        plt.savefig(self.png_path, dpi=200)
        plt.close()

        print(f"[ShuffleOutputRecorder]{self._worker_suffix()} Recorded {len(xs)} samples.")
        print(f"[ShuffleOutputRecorder]{self._worker_suffix()} Saved csv to: {self.csv_path}")
        print(f"[ShuffleOutputRecorder]{self._worker_suffix()} Saved plot to: {self.png_path}")

class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """LeRobotDataset with streaming capabilities.

    This class extends LeRobotDataset to add streaming functionality, allowing data to be streamed
    rather than loaded entirely into memory. This is especially useful for large datasets that may
    not fit in memory or when you want to quickly explore a dataset without downloading it completely.

    The key innovation is using a Backtrackable iterator that maintains a bounded buffer of recent
    items, allowing us to access previous frames for delta timestamps without loading the entire
    dataset into memory.

    Example:
        Basic usage:
        ```python
        from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

        # Create a streaming dataset with delta timestamps
        delta_timestamps = {
            "observation.image": [-1.0, -0.5, 0.0],  # 1 sec ago, 0.5 sec ago, current
            "action": [0.0, 0.1, 0.2],  # current, 0.1 sec future, 0.2 sec future
        }

        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=1000,
        )

        # Iterate over the dataset
        for i, item in enumerate(dataset):
            print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
            # item will contain stacked frames according to delta_timestamps
            if i >= 10:
                break
        ```
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
        video_backend: str = "pyav",
    ):
        """Initialize a StreamingLeRobotDataset.

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset.
            root (Path | None, optional): Local directory to use for downloading/writing files.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list.
            image_transforms (Callable | None, optional): Transform to apply to image data.
            tolerance_s (float, optional): Tolerance in seconds for timestamp matching.
            revision (str, optional): Git revision id (branch name, tag, or commit hash).
            force_cache_sync (bool, optional): Flag to sync and refresh local files first.
            streaming (bool, optional): Whether to stream the dataset or load it all. Defaults to True.
            buffer_size (int, optional): Buffer size for shuffling when streaming. Defaults to 1000.
            max_num_shards (int, optional): Number of shards to re-shard the input dataset into. Defaults to 16.
            seed (int, optional): Reproducibility random seed.
            rng (np.random.Generator | None, optional): Random number generator.
            shuffle (bool, optional): Whether to shuffle the dataset across exhaustions. Defaults to True.
            video_backend (str, optional): Video backend to use for decoding. Defaults to "pyav".
                Options: "pyav" (recommended), "torchcodec" (may have issues with AV1 on some platforms).
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.shuffle = shuffle

        self.streaming = streaming
        self.buffer_size = buffer_size
        self.video_backend = video_backend
        self.max_num_shards = max_num_shards

        # We cache the video decoders to avoid re-initializing them at each frame (avoiding a ~10x slowdown)
        self.video_decoder_cache = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # Check version
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        self.delta_timestamps = None
        self.delta_indices = None

        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)  # raises ValueError if invalid
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.hf_dataset: datasets.IterableDataset = load_dataset(
            self.repo_id if not self.streaming_from_local else str(self.root),
            split="train",
            streaming=self.streaming,
            data_files="data/*/*.parquet",
            revision=self.revision,
        )

        import logging
        logging.info(f"Streaming dataset loaded: streaming={self.streaming}, streaming_from_local={self.streaming_from_local}")
        logging.info(f"Dataset root: {self.root}")

        # num_shards is only available for IterableDataset (streaming=True)
        if self.streaming:
            self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)
            logging.info(f"Dataset num_shards: {self.hf_dataset.num_shards}, using max_num_shards: {max_num_shards}")
            # Ensure at least 1 shard so the iterator loop can run (num_workers=0 => max_num_shards=0)
            if self.num_shards == 0:
                self.num_shards = 1
                logging.warning(f"num_shards is 0, setting to 1")
        else:
            # For non-streaming, use a single shard
            self.num_shards = 1

        logging.info(f"Final num_shards for iteration: {self.num_shards}")

    @property
    def num_frames(self):
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def fps(self):
        return self.meta.fps

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        import torch
        import numpy as np

        collated = {}

        for key in batch[0].keys():
            # Do not pass raw task strings through accelerate
            if key == "task":
                continue

            values = [item[key] for item in batch]

            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                try:
                    arr = np.stack(values)
                    collated[key] = torch.from_numpy(arr)
                except Exception:
                    collated[key] = values

        return collated    
    
    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator, buffer_size: int, random_batch_size=100
    ) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    @staticmethod
    def _infinite_generator_over_elements(rng: np.random.Generator, elements: list[int]) -> Iterator[int]:
        while True:
            yield rng.choice(elements)

    # TODO(fracapuano): Implement multi-threaded prefetching to accelerate data loading.
    # The current sequential iteration is a bottleneck. A producer-consumer pattern
    # could be used with a ThreadPoolExecutor to run `make_frame` (especially video decoding)
    # in parallel, feeding a queue from which this iterator will yield processed items.
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.video_decoder_cache is None:
            self.video_decoder_cache = VideoDecoderCache()

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        has_multiple_workers = worker_info is not None and worker_info.num_workers > 1
        prefix = f"shuffle_buffer{self.buffer_size}_shards{self.max_num_shards}"
        if has_multiple_workers:
            prefix = f"{prefix}_worker{worker_id}"
        self.shuffle_output_recorder = ShuffleOutputRecorder(
            enable=True,
            output_dir="outputs/shuffle_debug",
            filename_prefix=prefix,
            worker_id=worker_id if has_multiple_workers else None,
            num_workers=worker_info.num_workers if worker_info is not None else 1,
            max_points=200_000,
            index_key="index",
        )

        # Infinite iteration for training with accelerator.prepare()
        while True:
            # keep the same seed across exhaustions if shuffle is False, otherwise shuffle data across exhaustions
            rng = np.random.default_rng(self.seed) if not self.shuffle else self.rng

            buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)

            idx_to_backtrack_dataset = {
                idx: self._make_backtrackable_dataset(safe_shard(self.hf_dataset, idx, self.num_shards))
                for idx in range(self.num_shards)
            }

            # This buffer is populated while iterating on the dataset's shards
            # the logic is to add 2 levels of randomness:
            # (1) sample one shard at random from the ones available, and
            # (2) sample one frame from the shard sampled at (1)
            frames_buffer = []

            while available_shards := list(idx_to_backtrack_dataset.keys()):
                shard_key = next(self._infinite_generator_over_elements(rng, available_shards))
                backtrack_dataset = idx_to_backtrack_dataset[shard_key]  # selects which shard to iterate on

                try:
                    for frame in self.make_frame(backtrack_dataset):
                        if len(frames_buffer) == self.buffer_size:
                            i = next(buffer_indices_generator)

                            out_frame = frames_buffer[i]
                            self.shuffle_output_recorder.record(out_frame)

                            yield out_frame

                            frames_buffer[i] = frame
                        else:
                            frames_buffer.append(frame)
                        break  # random shard sampled, switch shard
                except StopIteration:
                    del idx_to_backtrack_dataset[shard_key]
                except (FileNotFoundError, Exception) as e:
                    import logging

                    logging.error(
                        f"Error loading from shard {shard_key}: "
                        f"{type(e).__name__}: {e}"
                    )
                    del idx_to_backtrack_dataset[shard_key]

            # Once shards are all exhausted, shuffle the buffer and yield the remaining frames
            rng.shuffle(frames_buffer)

            if len(frames_buffer) == 0:
                import logging

                logging.error(
                    "No frames collected from any shard! "
                    "This indicates a problem with the dataset loading."
                )
                logging.error(
                    f"Available shards: {list(idx_to_backtrack_dataset.keys())}"
                )
                raise RuntimeError(
                    "Failed to load any frames from the streaming dataset. "
                    "Check logs for details."
                )

            for out_frame in frames_buffer:
                self.shuffle_output_recorder.record(out_frame)
                yield out_frame

            # 每轮数据遍历结束后保存一次图
            self.shuffle_output_recorder.save_plot()
            
    def _get_window_steps(
        self, delta_timestamps: dict[str, list[float]] | None = None, dynamic_bounds: bool = False
    ) -> tuple[int, int]:
        if delta_timestamps is None:
            return 1, 1

        if not dynamic_bounds:
            # Fix the windows
            lookback = LOOKBACK_BACKTRACKTABLE
            lookahead = LOOKAHEAD_BACKTRACKTABLE
        else:
            # Dynamically adjust the windows based on the given delta_timesteps
            all_timestamps = sum(delta_timestamps.values(), [])
            lookback = min(all_timestamps) * self.fps
            lookahead = max(all_timestamps) * self.fps

            # When lookback is >=0 it means no negative timesteps have been provided
            lookback = 0 if lookback >= 0 else (lookback * -1)

        return lookback, lookahead

    def _make_backtrackable_dataset(self, dataset: datasets.IterableDataset) -> Backtrackable:
        lookback, lookahead = self._get_window_steps(self.delta_timestamps)
        return Backtrackable(dataset, history=lookback, lookahead=lookahead)

    def _make_timestamps_from_indices(
        self, start_ts: float, indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        if indices is not None:
            return {
                key: (
                    start_ts + torch.tensor(indices[key]) / self.fps
                ).tolist()  # NOTE: why not delta_timestamps directly?
                for key in self.delta_timestamps
            }
        else:
            return dict.fromkeys(self.meta.video_keys, [start_ts])

    def _make_padding_camera_frame(self, camera_key: str):
        """Variable-shape padding frame for given camera keys, given in (H, W, C)"""
        return torch.zeros(self.meta.info["features"][camera_key]["shape"]).permute(-1, 0, 1)

    def _get_video_frame_padding_mask(
        self,
        video_frames: dict[str, torch.Tensor],
        query_timestamps: dict[str, list[float]],
        original_timestamps: dict[str, list[float]],
    ) -> dict[str, torch.BoolTensor]:
        padding_mask = {}

        for video_key, timestamps in original_timestamps.items():
            if video_key not in video_frames:
                continue  # only padding on video keys that are available
            frames = []
            mask = []
            padding_frame = self._make_padding_camera_frame(video_key)
            for ts in timestamps:
                if is_float_in_list(ts, query_timestamps[video_key]):
                    idx = find_float_index(ts, query_timestamps[video_key])
                    frames.append(video_frames[video_key][idx, :])
                    mask.append(False)
                else:
                    frames.append(padding_frame)
                    mask.append(True)

            padding_mask[f"{video_key}_is_pad"] = torch.BoolTensor(mask)

        return padding_mask

    def make_frame(self, dataset_iterator: Backtrackable) -> Generator:
        """Makes a frame starting from a dataset iterator"""
        with torch.profiler.record_function("dataloader_parquet_read"):
            item = next(dataset_iterator)
            item = item_to_torch(item)

        updates = []  # list of "updates" to apply to the item retrieved from hf_dataset (w/o camera features)

        # Get episode index from the item
        ep_idx = item["episode_index"]

        # "timestamp" restarts from 0 for each episode, whereas we need a global timestep within the single .mp4 file (given by index/fps)
        current_ts = item["index"] / self.fps

        episode_boundaries_ts = {
            key: (
                self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        # Apply delta querying logic if necessary
        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.append(query_result)
            updates.append(padding)

        # Load video frames, when needed
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)

            # Some timestamps might not result available considering the episode's boundaries
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )

            video_frames = self._query_videos(query_timestamps, ep_idx)

            if self.image_transforms is not None:
                with torch.profiler.record_function("dataloader_image_transforms"):
                    image_keys = self.meta.camera_keys
                    for cam in image_keys:
                        video_frames[cam] = self.image_transforms(video_frames[cam])

            updates.append(video_frames)

            if self.delta_indices is not None:
                # We always return the same number of frames. Unavailable frames are padded.
                padding_mask = self._get_video_frame_padding_mask(
                    video_frames, query_timestamps, original_timestamps
                )
                updates.append(padding_mask)

        result = item.copy()
        for update in updates:
            result.update(update)

        # Keep original dataset index for shuffle-buffer visualization
        result["index"] = item["index"]

        # Keep 'task' as string - this will be handled specially in collate
        task_index = item["task_index"]
        if isinstance(task_index, torch.Tensor):
            task_index = int(task_index.item())
        else:
            task_index = int(task_index)

        result["task_index"] = task_index
        result["task"] = self.meta.tasks.iloc[task_index].name

        yield result

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        episode_boundaries_ts: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        keys_to_timestamps = self._make_timestamps_from_indices(current_ts, query_indices)
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = keys_to_timestamps[key]
                # Clamp out timesteps outside of episode boundaries
                query_timestamps[key] = torch.clamp(
                    torch.tensor(timestamps), *episode_boundaries_ts[key]
                ).tolist()

            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        from lerobot.datasets.video_utils import decode_video_frames

        item = {}
        for video_key, query_ts in query_timestamps.items():
            # Determine the correct root path:
            # - If streaming from local (root is provided), use local path
            # - If streaming from HF Hub, use url_root
            if self.streaming_from_local:
                root = self.root
            elif self.streaming:
                root = self.meta.url_root
            else:
                root = self.root

            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"

            # Check if file exists
            from pathlib import Path
            if not Path(video_path).exists():
                error_msg = f"Video file not found: {video_path}"
                import logging
                logging.error(f"{error_msg} (episode_index={ep_idx}, video_key={video_key})")
                raise FileNotFoundError(error_msg)

            # Use the specified video backend (pyav or torchcodec)
            try:
                with torch.profiler.record_function("dataloader_video_decode"):
                    frames = decode_video_frames(
                        video_path, query_ts, self.tolerance_s, backend=self.video_backend
                    )
            except Exception as e:
                import logging
                logging.error(f"Failed to decode video {video_path}: {type(e).__name__}: {e}")
                raise

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames

        return item

    def _get_delta_frames(self, dataset_iterator: Backtrackable, current_item: dict):
        # TODO(fracapuano): Modularize this function, refactor the code
        """Get frames with delta offsets using the backtrackable iterator.

        Args:
            current_item (dict): Current item from the iterator.
            ep_idx (int): Episode index.

        Returns:
            tuple: (query_result, padding) - frames at delta offsets and padding info.
        """
        current_episode_idx = current_item["episode_index"]

        # Prepare results
        query_result = {}
        padding = {}

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue  # visual frames are decoded separately

            target_frames = []
            is_pad = []

            # Create a results dictionary to store frames in processing order, then reconstruct original order for stacking
            delta_results = {}

            # Separate and sort deltas by difficulty (easier operations first)
            negative_deltas = sorted([d for d in delta_indices if d < 0], reverse=True)  # [-1, -2, -3, ...]
            positive_deltas = sorted([d for d in delta_indices if d > 0])  # [1, 2, 3, ...]
            zero_deltas = [d for d in delta_indices if d == 0]

            # Process zero deltas (current frame)
            for delta in zero_deltas:
                delta_results[delta] = (
                    current_item[key],
                    False,
                )

            # Process negative deltas in order of increasing difficulty
            lookback_failed = False

            last_successful_frame = current_item[key]

            for delta in negative_deltas:
                if lookback_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    steps_back = abs(delta)
                    if dataset_iterator.can_peek_back(steps_back):
                        past_item = dataset_iterator.peek_back(steps_back)
                        past_item = item_to_torch(past_item)

                        if past_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (past_item[key], False)
                            last_successful_frame = past_item[key]

                        else:
                            raise LookBackError("Retrieved frame is from different episode!")
                    else:
                        raise LookBackError("Cannot go back further than the history buffer!")

                except LookBackError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookback_failed = True  # All subsequent negative deltas will also fail

            # Process positive deltas in order of increasing difficulty
            lookahead_failed = False
            last_successful_frame = current_item[key]

            for delta in positive_deltas:
                if lookahead_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    if dataset_iterator.can_peek_ahead(delta):
                        future_item = dataset_iterator.peek_ahead(delta)
                        future_item = item_to_torch(future_item)

                        if future_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (future_item[key], False)
                            last_successful_frame = future_item[key]

                        else:
                            raise LookAheadError("Retrieved frame is from different episode!")
                    else:
                        raise LookAheadError("Cannot go ahead further than the lookahead buffer!")

                except LookAheadError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookahead_failed = True  # All subsequent positive deltas will also fail

            # Reconstruct original order for stacking
            for delta in delta_indices:
                frame, is_padded = delta_results[delta]

                # add batch dimension for stacking
                target_frames.append(frame)  # frame.unsqueeze(0))
                is_pad.append(is_padded)

            # Stack frames and add to results
            if target_frames:
                query_result[key] = torch.stack(target_frames)
                padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

        return query_result, padding

    def _validate_delta_timestamp_keys(self, delta_timestamps: dict[list[float]]) -> None:
        """
        Validate that all keys in delta_timestamps correspond to actual features in the dataset.

        Raises:
            ValueError: If any delta timestamp key doesn't correspond to a dataset feature.
        """
        if delta_timestamps is None:
            return

        # Get all available feature keys from the dataset metadata
        available_features = set(self.meta.features.keys())

        # Get all keys from delta_timestamps
        delta_keys = set(delta_timestamps.keys())

        # Find any keys that don't correspond to features
        invalid_keys = delta_keys - available_features

        if invalid_keys:
            raise ValueError(
                f"The following delta_timestamp keys do not correspond to dataset features: {invalid_keys}. "
                f"Available features are: {sorted(available_features)}"
            )
