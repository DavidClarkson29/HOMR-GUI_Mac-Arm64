import os
from pathlib import Path

from homr.app_paths import bundled_resource_path, model_dir

script_location = str(model_dir())

model_name = "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f"

segnet_path_onnx = os.path.join(script_location, f"{model_name}.onnx")

segnet_path_onnx_fp16 = os.path.join(script_location, f"{model_name}_fp16.onnx")

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    f"{model_name}.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version

coreml_model_name = "segnet_ane_batch8.mlpackage"


def find_coreml_segnet_path() -> Path | None:
    configured = os.environ.get("HOMR_COREML_SEGNET_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.exists() else None

    candidates = (
        bundled_resource_path(Path("homr") / "models" / coreml_model_name),
        model_dir() / coreml_model_name,
    )
    return next((path for path in candidates if path.exists()), None)
