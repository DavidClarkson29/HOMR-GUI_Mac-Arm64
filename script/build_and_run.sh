#!/bin/zsh

set -euo pipefail

MODE="${1:-run}"
PROJECT_ROOT="${0:A:h:h}"
APP_NAME="HOMR"
APP_PATH="${PROJECT_ROOT}/dist/HOMR.app"
APP_BINARY="${APP_PATH}/Contents/MacOS/HOMR"
BUNDLE_ID="com.quackone.homr"

pkill -x "${APP_NAME}" >/dev/null 2>&1 || true
"${PROJECT_ROOT}/scripts/build_macos_app.sh"

open_app() {
  /usr/bin/open -n "${APP_PATH}"
}

case "${MODE}" in
  run)
    open_app
    ;;
  --debug|debug)
    lldb -- "${APP_BINARY}"
    ;;
  --logs|logs)
    open_app
    /usr/bin/log stream --info --style compact --predicate "process == \"${APP_NAME}\""
    ;;
  --telemetry|telemetry)
    open_app
    /usr/bin/log stream --info --style compact --predicate "subsystem == \"${BUNDLE_ID}\""
    ;;
  --verify|verify)
    open_app
    sleep 2
    pgrep -x "${APP_NAME}" >/dev/null
    ;;
  *)
    print -u2 "usage: $0 [run|--debug|--logs|--telemetry|--verify]"
    exit 2
    ;;
esac
