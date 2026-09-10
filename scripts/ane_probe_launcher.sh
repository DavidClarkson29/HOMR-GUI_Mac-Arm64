#!/bin/zsh

set -u

BUNDLE_CONTENTS="${0:A:h:h}"
REPORT_PATH="/private/tmp/homr-ane-probe.txt"

"${BUNDLE_CONTENTS}/Resources/profile_coreml" \
  "${BUNDLE_CONTENTS}/Resources/model.mlmodelc" \
  >"${REPORT_PATH}" 2>&1
print "exit_status=$?" >>"${REPORT_PATH}"

# Keep the process alive briefly so desktop automation can observe launch.
sleep 5
