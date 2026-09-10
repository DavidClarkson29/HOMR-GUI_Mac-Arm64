#!/bin/zsh

set -u

BUNDLE_CONTENTS="${0:A:h:h}"
RESOURCES="${BUNDLE_CONTENTS}/Resources"
REPORT_PATH="/private/tmp/homr-packaged-ane-smoke.json"
LOG_PATH="/private/tmp/homr-packaged-ane-smoke.log"

HOMR_HEADLESS_IMAGE="${RESOURCES}/tabi.png" \
HOMR_HEADLESS_REPORT="${REPORT_PATH}" \
HOMR_MODEL_DIR="${RESOURCES}/models" \
HOMR_DATA_DIR="/private/tmp/homr-packaged-ane-data" \
HOMR_CACHE_DIR="/private/tmp/homr-packaged-ane-cache" \
MPLCONFIGDIR="/private/tmp/homr-packaged-matplotlib" \
PYTHONPYCACHEPREFIX="/private/tmp/homr-packaged-pycache" \
"${RESOURCES}/HOMR.app/Contents/MacOS/HOMR" >"${LOG_PATH}" 2>&1
print "exit_status=$?" >>"${REPORT_PATH}"

sleep 5
