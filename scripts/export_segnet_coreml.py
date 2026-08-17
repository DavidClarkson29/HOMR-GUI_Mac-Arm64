from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from training.architecture.segmentation.model import create_segnet


class InferenceSegnet(torch.nn.Module):
    def __init__(self, source: torch.nn.Module) -> None:
        super().__init__()
        self.network = source.model
        self.register_buffer("mean", source.mean.detach().clone())
        self.register_buffer("std", source.std.detach().clone())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network((image - self.mean) / self.std)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HOMR SegNet as a static ANE model.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    source = create_segnet(skip_weights_download=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source.load_state_dict(state, strict=True)
    model = InferenceSegnet(source).eval()
    sample = torch.zeros(args.batch_size, 3, 320, 320)
    traced = torch.jit.trace(model, sample)

    coreml_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=sample.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    coreml_model.short_description = "HOMR SegNet optimized for Apple Neural Engine"
    output.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(output))
    sys.stdout.write(f"Saved {output}\n")


if __name__ == "__main__":
    main()
