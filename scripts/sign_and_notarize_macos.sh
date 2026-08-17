#!/bin/zsh

set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
APP_PATH="${1:-${PROJECT_ROOT}/dist/HOMR.app}"
OUTPUT_ZIP="${2:-${PROJECT_ROOT}/dist/HOMR-macOS-arm64-notarized.zip}"

: "${HOMR_SIGNING_IDENTITY:?Set HOMR_SIGNING_IDENTITY to a Developer ID Application identity}"
: "${HOMR_NOTARY_PROFILE:?Set HOMR_NOTARY_PROFILE to an xcrun notarytool keychain profile}"

if [[ ! -d "${APP_PATH}" ]]; then
  print -u2 "Application bundle does not exist: ${APP_PATH}"
  exit 1
fi

codesign --force --deep --strict --options runtime --timestamp \
  --sign "${HOMR_SIGNING_IDENTITY}" "${APP_PATH}"
codesign --verify --deep --strict --verbose=2 "${APP_PATH}"
codesign -dvvv "${APP_PATH}"

ditto -c -k --sequesterRsrc --keepParent "${APP_PATH}" "${OUTPUT_ZIP}"
xcrun notarytool submit "${OUTPUT_ZIP}" \
  --keychain-profile "${HOMR_NOTARY_PROFILE}" --wait
xcrun stapler staple "${APP_PATH}"
xcrun stapler validate "${APP_PATH}"
spctl -a -t exec -vv "${APP_PATH}"

# Recreate the archive so the delivered copy contains the stapled ticket.
ditto -c -k --sequesterRsrc --keepParent "${APP_PATH}" "${OUTPUT_ZIP}"
shasum -a 256 "${OUTPUT_ZIP}"
