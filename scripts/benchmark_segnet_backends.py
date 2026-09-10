from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
import onnxruntime as ort


def _measure(function, iterations: int) -> tuple[np.ndarray, float]:
    for _ in range(2):
        output = function()
    started = perf_counter()
    for _ in range(iterations):
        output = function()
    return output, (perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU ONNX against native Core ML SegNet."
    )
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("coreml_model", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    input_data = np.random.default_rng(7).random((8, 3, 320, 320), dtype=np.float32)
    onnx_model = ort.InferenceSession(
        str(args.onnx_model.expanduser().resolve()),
        providers=["CPUExecutionProvider"],
    )
    coreml_model = ct.models.MLModel(
        str(args.coreml_model.expanduser().resolve()),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    onnx_output, onnx_seconds = _measure(
        lambda: onnx_model.run(None, {"input": input_data})[0],
        args.iterations,
    )
    coreml_output, coreml_seconds = _measure(
        lambda: coreml_model.predict({"input": input_data})["output"],
        args.iterations,
    )
    difference = np.abs(onnx_output - coreml_output)
    result = {
        "batch_size": 8,
        "iterations": args.iterations,
        "onnx_cpu_seconds": onnx_seconds,
        "native_coreml_seconds": coreml_seconds,
        "speedup": onnx_seconds / coreml_seconds,
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "argmax_agreement": float(
            np.mean(np.argmax(onnx_output, axis=1) == np.argmax(coreml_output, axis=1))
        ),
    }
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
