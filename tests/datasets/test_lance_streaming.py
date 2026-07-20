#!/usr/bin/env python

import torch

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset


def test_streaming_collate_stacks_values_and_drops_task_strings():
    batch = [
        {"action": torch.tensor([1.0]), "task_index": 0, "task": "pick"},
        {"action": torch.tensor([2.0]), "task_index": 1, "task": "place"},
    ]

    collated = StreamingLeRobotDataset.collate_fn(batch)

    assert "task" not in collated
    assert torch.equal(collated["action"], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(collated["task_index"], torch.tensor([0, 1]))


def test_lance_iterator_configures_worker_and_filters_episodes(monkeypatch):
    class FakeLanceReader:
        def __init__(self):
            self.calls = []
            self.batches = [
                [{"episode_index": torch.tensor(0), "index": torch.tensor(0)}],
                [{"episode_index": torch.tensor(1), "index": torch.tensor(1)}],
            ]

        def configure_worker_shard(self, worker_id, num_workers):
            self.calls.append(("configure", worker_id, num_workers))

        def reset(self):
            self.calls.append(("reset",))

        def load_next_batch(self):
            if not self.batches:
                raise StopIteration
            return self.batches.pop(0)

    dataset = object.__new__(StreamingLeRobotDataset)
    dataset.lance_dataset = FakeLanceReader()
    dataset.episodes = [1]
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)

    frames = list(dataset._iter_lance())

    assert [int(frame["index"]) for frame in frames] == [1]
    assert dataset.lance_dataset.calls == [("configure", 0, 1), ("reset",)]
