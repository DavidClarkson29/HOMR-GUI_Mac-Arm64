# ruff: noqa: INP001, I001

import sys

# The frozen app has the native cv2 extension beside the loader sources in
# Contents/Resources. Make OpenCV put that directory ahead of PyInstaller's
# Contents/Frameworks path while it replaces the Python loader package.
sys.OpenCV_REPLACE_SYS_PATH_0 = True
