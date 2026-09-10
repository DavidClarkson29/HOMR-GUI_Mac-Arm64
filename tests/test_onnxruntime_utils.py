# ruff: noqa: S101

from homr import onnxruntime_utils as runtime


def _providers(monkeypatch, providers: list[str], device: str = "CPU") -> None:
    monkeypatch.setattr(runtime, "get_available_providers", lambda: providers)
    monkeypatch.setattr(runtime.ort, "get_device", lambda: device)


def test_coreml_is_preferred_over_cpu(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOMR_CACHE_DIR", str(tmp_path))
    _providers(monkeypatch, [runtime.COREML_EP, runtime.CPU_EP])

    providers = runtime.execution_providers(use_gpu_inference=True)

    assert providers[0][0] == runtime.COREML_EP
    assert providers[0][1]["MLComputeUnits"] == "ALL"
    assert providers[-1] == runtime.CPU_EP
    assert not runtime.uses_fp16_models(use_accelerated_inference=True)
    assert not runtime.should_accelerate_transformer(use_accelerated_inference=True)


def test_cuda_keeps_fp16_models_and_takes_priority(monkeypatch) -> None:
    _providers(
        monkeypatch,
        [runtime.COREML_EP, runtime.CUDA_EP, runtime.CPU_EP],
        device="GPU",
    )

    providers = runtime.execution_providers(use_gpu_inference=True)

    assert providers[0] == runtime.CUDA_EP
    assert runtime.uses_fp16_models(use_accelerated_inference=True)
    assert runtime.should_accelerate_transformer(use_accelerated_inference=True)


def test_acceleration_can_be_disabled(monkeypatch) -> None:
    _providers(monkeypatch, [runtime.COREML_EP, runtime.CPU_EP])

    assert runtime.execution_providers(use_gpu_inference=False) == [runtime.CPU_EP]
    assert not runtime.uses_fp16_models(use_accelerated_inference=False)
