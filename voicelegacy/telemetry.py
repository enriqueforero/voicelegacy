"""Runtime timing and hardware telemetry helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from voicelegacy.logging_config import get_logger

logger = get_logger()


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Small, JSON-friendly runtime telemetry snapshot."""

    cuda_available: bool
    cuda_device_name: str | None = None
    cuda_allocated_mb: float | None = None
    cuda_reserved_mb: float | None = None
    cuda_max_allocated_mb: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "cuda_allocated_mb": self.cuda_allocated_mb,
            "cuda_reserved_mb": self.cuda_reserved_mb,
            "cuda_max_allocated_mb": self.cuda_max_allocated_mb,
        }


def runtime_snapshot() -> RuntimeSnapshot:
    """Capture torch/CUDA memory state when torch is installed."""
    try:
        import torch
    except ImportError:
        return RuntimeSnapshot(cuda_available=False)

    if not torch.cuda.is_available():
        return RuntimeSnapshot(cuda_available=False)

    device_idx = torch.cuda.current_device()
    return RuntimeSnapshot(
        cuda_available=True,
        cuda_device_name=torch.cuda.get_device_name(device_idx),
        cuda_allocated_mb=round(torch.cuda.memory_allocated(device_idx) / 1e6, 1),
        cuda_reserved_mb=round(torch.cuda.memory_reserved(device_idx) / 1e6, 1),
        cuda_max_allocated_mb=round(torch.cuda.max_memory_allocated(device_idx) / 1e6, 1),
    )


@contextmanager
def timed_step(label: str) -> Iterator[None]:
    """Log elapsed wall-clock time and CUDA memory around a pipeline step."""
    start = time.perf_counter()
    before = runtime_snapshot()
    logger.info("▶ {} | runtime_before={}", label, before.to_dict())
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        after = runtime_snapshot()
        logger.info("✓ {} | elapsed_s={:.2f} | runtime_after={}", label, elapsed, after.to_dict())
