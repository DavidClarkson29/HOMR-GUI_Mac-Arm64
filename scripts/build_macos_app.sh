#!/bin/zsh

set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
PYTHON_BIN="${HOMR_BUILD_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
APP_PATH="${PROJECT_ROOT}/dist/HOMR.app"
export PYINSTALLER_CONFIG_DIR="${PROJECT_ROOT}/build/pyinstaller-config"
export MPLCONFIGDIR="${PROJECT_ROOT}/build/matplotlib-config"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "HOMR's current macOS package target requires Apple Silicon."
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  print -u2 "Python environment not found: ${PYTHON_BIN}"
  print -u2 "Set HOMR_BUILD_PYTHON to a Python 3.11+ environment with project dependencies and PyInstaller."
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PYINSTALLER_CONFIG_DIR}" "${MPLCONFIGDIR}"
"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "${PROJECT_ROOT}/dist" \
  --workpath "${PROJECT_ROOT}/build/pyinstaller" \
  "${PROJECT_ROOT}/packaging/macos/HOMR.spec"

# OpenCV's loader resolves its source symlink into Contents/Resources/cv2,
# while PyInstaller keeps the native extension in Contents/Frameworks so it
# can be signed correctly. Some OpenCV/PyInstaller combinations omit the
# reverse link that the loader needs at runtime.
CV2_RESOURCE_DIR="${APP_PATH}/Contents/Resources/cv2"
CV2_FRAMEWORK_DIR="${APP_PATH}/Contents/Frameworks/cv2"
CV2_EXTENSIONS=("${CV2_FRAMEWORK_DIR}"/cv2*.so(N))
if [[ -d "${CV2_RESOURCE_DIR}" && ${#CV2_EXTENSIONS} -eq 1 ]]; then
  CV2_EXTENSION_NAME="${CV2_EXTENSIONS[1]:t}"
  ln -sfn "../../Frameworks/cv2/${CV2_EXTENSION_NAME}" \
    "${CV2_RESOURCE_DIR}/${CV2_EXTENSION_NAME}"
fi

# The GUI never opens PDFs and provides all of its own Chinese text. PyInstaller
# pulls QtPdf and the complete Qt translation catalog through optional image
# plugins; removing them keeps the local bundle lean without changing HOMR's
# PNG capture, Core ML, ONNX, MusicXML, or MIDI paths.
PRUNED_BUNDLE_PATHS=(
  "${APP_PATH}/Contents/Frameworks/PyQt6/Qt6/lib/QtPdf.framework"
  "${APP_PATH}/Contents/Frameworks/PyQt6/Qt6/plugins/imageformats/libqpdf.dylib"
  "${APP_PATH}/Contents/Frameworks/QtPdf"
  "${APP_PATH}/Contents/Resources/QtPdf"
  "${APP_PATH}/Contents/Frameworks/PyQt6/Qt6/translations"
  "${APP_PATH}/Contents/Resources/PyQt6/Qt6/translations"
)
for path_to_prune in "${PRUNED_BUNDLE_PATHS[@]}"; do
  if [[ -e "${path_to_prune}" || -L "${path_to_prune}" ]]; then
    find "${path_to_prune}" -depth -delete
  fi
done

codesign --force --deep --sign - "${APP_PATH}"
codesign --verify --deep --strict --verbose=2 "${APP_PATH}"
plutil -lint "${APP_PATH}/Contents/Info.plist"

print "Built ${APP_PATH}"
