# ruff: noqa: S101

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication, QLabel

import homr_gui


class _FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in self.callbacks:
            callback(*args)


class _FakeProcess:
    def __init__(self, _parent) -> None:
        self.finished = _FakeSignal()
        self.errorOccurred = _FakeSignal()
        self.program = ""
        self.arguments = []
        self.started = False

    def setProgram(self, program: str) -> None:
        self.program = program

    def setArguments(self, arguments: list[str]) -> None:
        self.arguments = arguments

    def start(self) -> None:
        self.started = True


def test_idle_button_starts_native_macos_capture(monkeypatch, tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(homr_gui, "QProcess", _FakeProcess)
    monkeypatch.setattr(homr_gui.sys, "platform", "darwin")
    monkeypatch.setenv("HOMR_CACHE_DIR", str(tmp_path))
    gui = homr_gui.HomrGui()

    gui.btn.click()

    process = gui._capture_process
    assert isinstance(process, _FakeProcess)
    assert process.program == "/usr/sbin/screencapture"
    assert process.arguments == ["-i", str(tmp_path / "capture_score.png")]
    assert process.started
    assert gui.current_state_name == "capturing"

    process.finished.emit(1, None)
    assert gui._capture_process is None
    assert gui.current_state_name == "idle"
    gui.close()
    app.processEvents()


def test_success_button_can_start_another_capture(monkeypatch, tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(homr_gui, "QProcess", _FakeProcess)
    monkeypatch.setattr(homr_gui.sys, "platform", "darwin")
    monkeypatch.setenv("HOMR_CACHE_DIR", str(tmp_path))
    gui = homr_gui.HomrGui()
    previous_midi = tmp_path / "previous.mid"
    previous_midi.touch()
    gui.current_output = str(previous_midi)
    gui._apply_state("success")

    gui.btn.click()

    assert isinstance(gui._capture_process, _FakeProcess)
    assert gui.current_state_name == "capturing"
    assert not previous_midi.exists()
    gui._capture_process.finished.emit(1, None)
    gui.close()
    app.processEvents()


def test_finished_capture_file_starts_recognition(monkeypatch, tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(homr_gui, "QProcess", _FakeProcess)
    monkeypatch.setattr(homr_gui.sys, "platform", "darwin")
    monkeypatch.setenv("HOMR_CACHE_DIR", str(tmp_path))
    gui = homr_gui.HomrGui()
    recognized = []
    monkeypatch.setattr(gui, "_start_recognition", recognized.append)

    gui.btn.click()
    capture_path = tmp_path / "capture_score.png"
    capture_path.write_bytes(b"PNG test payload")
    gui._capture_process.finished.emit(0, None)

    assert recognized == [str(capture_path)]
    assert gui._capture_process is None
    gui.close()
    app.processEvents()


def test_image_content_id_tracks_pixels_not_qt_cache_identity() -> None:
    first = QImage(12, 12, QImage.Format.Format_RGBA8888)
    first.fill(QColor("white"))
    same_pixels = first.copy()
    changed = first.copy()
    changed.setPixelColor(4, 4, QColor("black"))

    assert homr_gui.image_content_id(first) == homr_gui.image_content_id(same_pixels)
    assert homr_gui.image_content_id(first) != homr_gui.image_content_id(changed)


def test_xcode_style_uses_native_window_and_no_developer_credit() -> None:
    app = QApplication.instance() or QApplication([])
    gui = homr_gui.HomrGui()

    labels = [label.text() for label in gui.findChildren(QLabel)]

    assert not any("GUI by" in label or "尤里" in label for label in labels)
    assert "HOMR Engine · 临时文件在退出时自动清除" in labels
    assert "  ANE 优先  " in labels
    assert not gui.windowFlags() & Qt.WindowType.FramelessWindowHint
    gui.close()
    app.processEvents()
