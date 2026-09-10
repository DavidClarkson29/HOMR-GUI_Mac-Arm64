#!/bin/zsh

set -u

PROJECT_ROOT="${HOMR_ANE_BENCHMARK_PROJECT_ROOT:-${0:A:h:h:h:h:h}}"
PYTHON_BIN="/private/tmp/homr-mac-venv/bin/python"
REPORT_PATH="/private/tmp/homr-ane-e2e-benchmark.json"

cd "${PROJECT_ROOT}" || exit 1
MPLCONFIGDIR="${PROJECT_ROOT}/work/runtime/matplotlib" \
HOMR_COREML_SEGNET_PATH="${PROJECT_ROOT}/work/runtime/models/segnet_ane_batch8.mlpackage" \
HOMR_MODEL_DIR="${PROJECT_ROOT}/work/runtime/models" \
HOMR_DATA_DIR="${PROJECT_ROOT}/work/runtime/data" \
HOMR_CACHE_DIR="${PROJECT_ROOT}/work/runtime/cache" \
PYTHONPYCACHEPREFIX="/private/tmp/homr-pycache" \
"${PYTHON_BIN}" -m scripts.benchmark_backends \
  "${PROJECT_ROOT}/work/fixtures/tabi.png" \
  >"${REPORT_PATH}" 2>&1
print "exit_status=$?" >>"${REPORT_PATH}"

sleep 5
