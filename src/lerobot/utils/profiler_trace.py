#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

_TRACE_DIR: Path | None = None
_TRACE_ENABLED = False


def configure_dataloader_trace(trace_dir: Path | str | None, enabled: bool) -> None:
    global _TRACE_DIR, _TRACE_ENABLED

    _TRACE_ENABLED = enabled and trace_dir is not None
    _TRACE_DIR = Path(trace_dir) if trace_dir is not None else None
    if _TRACE_ENABLED and _TRACE_DIR is not None:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)


def dataloader_worker_init(worker_id: int, trace_dir: Path | str | None, enabled: bool) -> None:
    configure_dataloader_trace(trace_dir, enabled)
    record_dataloader_event(
        "dataloader_worker_init",
        start_time_ns=time.time_ns(),
        duration_ns=1,
        args={"worker_id": worker_id},
    )


def _get_worker_id() -> int | str:
    worker_info = torch.utils.data.get_worker_info()
    return worker_info.id if worker_info is not None else "main"


def record_dataloader_event(
    name: str,
    start_time_ns: int,
    duration_ns: int,
    args: dict[str, Any] | None = None,
) -> None:
    if not _TRACE_ENABLED or _TRACE_DIR is None:
        return

    event = {
        "name": name,
        "cat": "dataloader",
        "ph": "X",
        "ts": start_time_ns / 1000,
        "dur": duration_ns / 1000,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "args": {"worker_id": _get_worker_id(), **(args or {})},
    }
    trace_path = _TRACE_DIR / f"dataloader_worker_trace_{os.getpid()}.jsonl"
    try:
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
    except OSError as exc:
        logging.debug("Could not write dataloader trace event %s: %s", name, exc)


@contextmanager
def dataloader_trace(name: str, **args: Any):
    start_time_ns = time.time_ns()
    start_perf_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        record_dataloader_event(
            name,
            start_time_ns=start_time_ns,
            duration_ns=time.perf_counter_ns() - start_perf_ns,
            args=args,
        )


def merge_dataloader_traces(trace_dir: Path | str, output_name: str = "dataloader_multiprocess.trace.json") -> Path | None:
    trace_dir = Path(trace_dir)
    events = []
    for path in sorted(trace_dir.glob("dataloader_worker_trace_*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not read dataloader trace file %s: %s", path, exc)

    if not events:
        return None

    output_path = trace_dir / output_name
    payload = {
        "traceEvents": sorted(events, key=lambda event: event["ts"]),
        "displayTimeUnit": "ms",
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    return output_path
