from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class StageStartEvent:
    layer: int
    total_layers: int
    stage: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class StageEvent:
    layer: int
    total_layers: int
    stage: str
    elapsed_s: float
    timings_so_far: dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class LayerEvent:
    layer: int
    total_layers: int
    layer_timings: dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineEvent:
    total_timings: dict[str, float]
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class ProgressCallback(Protocol):
    def on_stage_start(self, *, layer: int, total_layers: int, stage: str) -> None: ...

    def on_stage_complete(
        self, *, layer: int, total_layers: int, stage: str, elapsed_s: float, timings_so_far: dict[str, float]
    ) -> None: ...

    def on_layer_complete(self, *, layer: int, total_layers: int, layer_timings: dict[str, float]) -> None: ...

    def on_pipeline_complete(self, *, total_timings: dict[str, float]) -> None: ...


class PrintProgressCallback:
    def on_stage_start(self, *, layer: int, total_layers: int, stage: str) -> None:
        print(f"  Layer {layer + 1}/{total_layers} | {stage:<25s} starting...", flush=True)

    def on_stage_complete(
        self, *, layer: int, total_layers: int, stage: str, elapsed_s: float, timings_so_far: dict[str, float]
    ) -> None:
        ms = elapsed_s * 1000
        print(f"  Layer {layer + 1}/{total_layers} | {stage:<25s} {ms:>8.1f} ms", flush=True)

    def on_layer_complete(self, *, layer: int, total_layers: int, layer_timings: dict[str, float]) -> None:
        total = sum(layer_timings.values())
        print(f"  Layer {layer + 1}/{total_layers} complete — {total:.2f}s", flush=True)

    def on_pipeline_complete(self, *, total_timings: dict[str, float]) -> None:
        total = sum(total_timings.values())
        print(f"  Pipeline complete — {total:.2f}s total", flush=True)


class QueueProgressCallback:
    def __init__(self, q: queue.Queue | None = None):
        self.q: queue.Queue = q or queue.Queue()

    def on_stage_start(self, *, layer: int, total_layers: int, stage: str) -> None:
        self.q.put(
            StageStartEvent(
                layer=layer,
                total_layers=total_layers,
                stage=stage,
            )
        )

    def on_stage_complete(
        self, *, layer: int, total_layers: int, stage: str, elapsed_s: float, timings_so_far: dict[str, float]
    ) -> None:
        self.q.put(
            StageEvent(
                layer=layer,
                total_layers=total_layers,
                stage=stage,
                elapsed_s=elapsed_s,
                timings_so_far=dict(timings_so_far),
            )
        )

    def on_layer_complete(self, *, layer: int, total_layers: int, layer_timings: dict[str, float]) -> None:
        self.q.put(
            LayerEvent(
                layer=layer,
                total_layers=total_layers,
                layer_timings=dict(layer_timings),
            )
        )

    def on_pipeline_complete(self, *, total_timings: dict[str, float]) -> None:
        self.q.put(PipelineEvent(total_timings=dict(total_timings)))
