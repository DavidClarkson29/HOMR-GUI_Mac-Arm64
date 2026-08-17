import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project_root = Path(SPECPATH).parents[1]

datas = [
    (str(path), "homr/transformer")
    for path in (project_root / "homr" / "transformer").glob("tokenizer_*.json")
]
datas += collect_data_files("rapidocr")
datas += collect_data_files(
    "music21",
    excludes=[
        "**/audioSearch/**",
        "**/corpus/**",
        "**/figuredBass/demo.xml",
        "**/languageExcerpts/**",
        "**/musicxml/lilypondTestSuite/**",
        "**/musicxml/test*.xml",
        "**/omr/**",
        "**/scale/**",
        "**/stream/*test.xml",
    ],
)
datas += collect_data_files("musicxml")

model_source_dir = Path(
    os.environ.get("HOMR_BUNDLE_MODEL_DIR", project_root / "work" / "runtime" / "models")
).expanduser().resolve()
required_models = (
    "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
    "encoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1.onnx",
    "decoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1.onnx",
    "segnet_ane_batch8.mlpackage",
)
for model_name in required_models:
    model_path = model_source_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Required bundled model does not exist: {model_path}")
    destination = f"homr/models/{model_name}" if model_path.is_dir() else "homr/models"
    datas.append((str(model_path), destination))

coreml_model = os.environ.get("HOMR_COREML_SEGNET_PATH")
if (
    coreml_model
    and Path(coreml_model).expanduser().resolve()
    != model_source_dir / "segnet_ane_batch8.mlpackage"
):
    coreml_model_path = Path(coreml_model).expanduser().resolve()
    if not coreml_model_path.exists():
        raise FileNotFoundError(f"Core ML SegNet does not exist: {coreml_model_path}")
    datas.append((str(coreml_model_path), f"homr/models/{coreml_model_path.name}"))

unused_rapidocr_backends = (
    "rapidocr.inference_engine.mnn",
    "rapidocr.inference_engine.openvino",
    "rapidocr.inference_engine.paddle",
    "rapidocr.inference_engine.pytorch",
    "rapidocr.inference_engine.tensorrt",
)
hiddenimports = collect_submodules(
    "rapidocr",
    filter=lambda name: not name.startswith(unused_rapidocr_backends),
)

analysis = Analysis(
    [str(project_root / "homr_gui.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / "packaging" / "macos" / "runtime_hook_opencv.py")],
    excludes=[
        "MNN",
        "matplotlib",
        "openvino",
        "paddle",
        "pytest",
        "tensorrt",
        "torch",
        "torchvision",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="HOMR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="HOMR",
)

app = BUNDLE(
    collection,
    name="HOMR.app",
    icon=str(project_root / "packaging" / "macos" / "assets" / "HOMR-icon-white.png"),
    bundle_identifier="com.quackone.homr",
    info_plist={
        "CFBundleDisplayName": "HOMR",
        "CFBundleShortVersionString": "0.5.10",
        "CFBundleVersion": "19",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
    },
)
