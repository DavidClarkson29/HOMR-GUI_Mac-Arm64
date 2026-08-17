from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from homr.type_definitions import NDArray


@lru_cache(maxsize=2)
def _load_model(model_path: str) -> Any:
    import coremltools as ct  # noqa: PLC0415 - optional macOS runtime dependency

    return ct.models.MLModel(
        model_path,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


class NativeCoreMLSegnet:
    """Static batch-8 SegNet executed by Apple's native Core ML runtime."""

    batch_size = 8

    def __init__(self, model_path: Path) -> None:
        self.model: Any = _load_model(str(model_path.resolve()))

    def run(self, input_data: NDArray) -> NDArray:
        data = input_data.astype(np.float32, copy=False)
        outputs: list[NDArray] = []
        for start in range(0, len(data), self.batch_size):
            chunk = data[start : start + self.batch_size]
            valid_count = len(chunk)
            if valid_count < self.batch_size:
                padded = np.zeros((self.batch_size, *chunk.shape[1:]), dtype=np.float32)
                padded[:valid_count] = chunk
                chunk = padded
            prediction = self.model.predict({"input": chunk})["output"]
            outputs.append(np.asarray(prediction, dtype=np.float32)[:valid_count])
        return np.concatenate(outputs, axis=0)
