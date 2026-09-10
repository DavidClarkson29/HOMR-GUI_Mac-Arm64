# macOS / Apple Silicon

HOMR supports a native Apple Silicon application bundle and a measured hybrid
inference path. The SegNet segmentation stage uses a native FP16 Core ML
MLProgram with static batch size 8 and requests `CPU_AND_NE`; the dynamic
Transformer encoder and decoder remain on ONNX Runtime CPU because their
current Core ML conversions are slower or unsupported.

## Runtime layout

The repository is self-contained in one project directory. Generated runtime
data is kept out of the source tree:

- Models: `~/Library/Application Support/HOMR/models`
- Cache and temporary screenshots: `~/Library/Caches/HOMR`
- Core ML compilation cache: managed by macOS

`HOMR_DATA_DIR`, `HOMR_CACHE_DIR`, `HOMR_MODEL_DIR`, and
`HOMR_COREML_SEGNET_PATH` can override these locations for tests and automation.

## Export the native Core ML model

Export from the matching upstream PyTorch SegNet checkpoint:

```bash
python -m pip install -r packaging/macos/requirements-coreml-export.txt
python scripts/export_segnet_coreml.py \
  path/to/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.pth \
  work/runtime/models/segnet_ane_batch8.mlpackage
```

The exported model has shape `[8, 3, 320, 320]`. Runtime code pads the last
partial batch and discards padded outputs.

## Build

Use an arm64 Python 3.11 or newer environment with the project dependencies:

```bash
python -m pip install -e .
python -m pip install -r packaging/macos/requirements-build.txt
HOMR_COREML_SEGNET_PATH="$PWD/work/runtime/models/segnet_ane_batch8.mlpackage" \
HOMR_BUILD_PYTHON="$(command -v python)" scripts/build_macos_app.sh
```

The result is `dist/HOMR.app`. The build script verifies that it is arm64,
works around the OpenCV frozen-loader layout, applies an ad-hoc signature, and
checks the final bundle. The native Core ML model is embedded in the app;
Transformer ONNX models are still downloaded on first use.

In the GUI, clicking **框选乐谱** starts macOS's native interactive screenshot
tool. Drag over the score region; the tool writes directly to HOMR's cache and
recognition starts when the capture process exits. The clipboard listener is
only a compatibility fallback for keyboard-driven screenshots. Press Escape to
cancel the selection.

On first use, macOS may ask for Screen Recording access. Install one fixed copy
of `HOMR.app` in `~/Applications`, grant access to that copy once, and do not
replace it with development rebuilds. An ad-hoc signature's code hash changes
on every rebuild, so repeatedly overwriting the app can make TCC request access
again. A Developer ID signature is the long-term identity for public releases.

GUI diagnostics are written to `~/Library/Caches/HOMR/homr-gui.log` with two
rotated backups.

After a successful conversion, a normal click starts another selection. To
export the current MIDI to a DAW, press the card and move the pointer far enough
to begin an actual drag; a stationary click is never treated as a file drag.

## ANE verification and performance

The claim of Neural Engine use is based on a real desktop process, not merely
on selecting a Core ML provider. `MLComputePlan` reports 80 measurable SegNet
operations whose preferred device is `MLNeuralEngineComputeDevice`; the other
241 entries are constants or have no preferred-device result.

Warm raw batch-8 SegNet measurements on the validation Mac:

| Backend | Seconds per batch | Relative speed |
| --- | ---: | ---: |
| ONNX Runtime CPU | 0.367366 | 1.00× |
| Native Core ML CPU + Neural Engine | 0.012416 | 29.59× |

The FP16 Core ML output has 100% argmax agreement with the CPU output on the
fixture. Maximum absolute tensor difference is `0.08624` and mean absolute
difference is `0.00562`. End-to-end MusicXML is byte-for-byte identical.

Cold one-shot end-to-end runs are approximately neutral (6.79 seconds CPU,
6.85 seconds native path) because Python imports and first-use model compilation
dominate a single recognition. A packaged run completed in about 5.44 seconds;
repeated recognition in the persistent GUI benefits from the warm model cache.

The ONNX CoreML provider and CPU remain automatic fallbacks if the native model
cannot be loaded. Set `HOMR_NATIVE_COREML=0` to disable the native route for a
regression comparison.

To inspect a compiled model's preferred devices:

```bash
swiftc scripts/profile_coreml.swift -o /tmp/profile_coreml \
  -framework CoreML -framework Foundation
/tmp/profile_coreml path/to/model.mlpackage
```

## Accuracy and backend regression

With all ONNX models already downloaded, compare the same score image on both
paths:

```bash
python -m scripts.benchmark_backends path/to/score.png
```

The benchmark reports wall-clock time, a SHA-256 digest of each MusicXML
output, exact-output equality, and speedup. Exact equality and 100% segmentation
argmax agreement on the fixture are regression checks, not a claim that every
possible score has been exhaustively validated.

## Signing and notarization

The default build is ad-hoc signed for local development. Public distribution
requires a Developer ID Application certificate and a stored `notarytool`
keychain profile:

```bash
export HOMR_SIGNING_IDENTITY='Developer ID Application: Example (TEAMID)'
export HOMR_NOTARY_PROFILE='homr-notary'
scripts/sign_and_notarize_macos.sh
```

The script applies hardened-runtime signing, verifies nested code, submits and
waits for notarization, staples and validates the ticket, runs Gatekeeper
assessment, then recreates the ZIP so the archive contains the stapled app.
