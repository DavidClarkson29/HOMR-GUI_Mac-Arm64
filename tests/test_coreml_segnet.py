# ruff: noqa: S101

import numpy as np

from homr.segmentation import config
from homr.segmentation.coreml_segnet import NativeCoreMLSegnet


class _FakeModel:
    def __init__(self) -> None:
        self.batch_shapes: list[tuple[int, ...]] = []

    def predict(self, values):
        input_data = values["input"]
        self.batch_shapes.append(input_data.shape)
        return {"output": input_data[:, :1]}


def test_static_batch_is_padded_and_trimmed() -> None:
    segnet = NativeCoreMLSegnet.__new__(NativeCoreMLSegnet)
    segnet.model = _FakeModel()
    input_data = np.ones((3, 3, 320, 320), dtype=np.float32)

    output = segnet.run(input_data)

    assert output.shape == (3, 1, 320, 320)
    assert segnet.model.batch_shapes == [(8, 3, 320, 320)]


def test_multiple_static_batches_are_concatenated() -> None:
    segnet = NativeCoreMLSegnet.__new__(NativeCoreMLSegnet)
    segnet.model = _FakeModel()
    input_data = np.ones((10, 3, 320, 320), dtype=np.float32)

    output = segnet.run(input_data)

    assert output.shape == (10, 1, 320, 320)
    assert segnet.model.batch_shapes == [(8, 3, 320, 320), (8, 3, 320, 320)]


def test_exported_model_filename_matches_runtime_lookup(monkeypatch, tmp_path) -> None:
    model = tmp_path / "segnet_ane_batch8.mlpackage"
    model.mkdir()
    monkeypatch.setenv("HOMR_COREML_SEGNET_PATH", str(model))

    assert config.coreml_model_name == model.name
    assert config.find_coreml_segnet_path() == model.resolve()
