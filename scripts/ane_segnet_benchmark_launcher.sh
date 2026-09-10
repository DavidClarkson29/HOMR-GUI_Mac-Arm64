#!/bin/zsh

set -u

PROJECT_ROOT="${HOMR_ANE_BENCHMARK_PROJECT_ROOT:-${0:A:h:h:h:h:h}}"
PYTHON_BIN="/private/tmp/homr-mac-venv/bin/python"
REPORT_PATH="/private/tmp/homr-ane-segnet-benchmark.json"

cd "${PROJECT_ROOT}" || exit 1
MPLCONFIGDIR="${PROJECT_ROOT}/work/runtime/matplotlib" \
PYTHONPYCACHEPREFIX="/private/tmp/homr-pycache" \
"${PYTHON_BIN}" -m scripts.benchmark_segnet_backends \
  "${PROJECT_ROOT}/work/runtime/models/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx" \
  "${PROJECT_ROOT}/work/runtime/models/segnet_ane_batch8.mlpackage" \
  --iterations 10 \
  >"${REPORT_PATH}" 2>&1
print "exit_status=$?" >>"${REPORT_PATH}"

sleep 5
