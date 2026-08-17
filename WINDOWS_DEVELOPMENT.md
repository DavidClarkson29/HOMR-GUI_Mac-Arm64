# HOMR Windows 开发交接说明

这个目录是 2026-08-17 的 Windows 移植开发基线，已包含当前 macOS 版的识别核心、节拍修正、单/双钢琴结构判断、MIDI 后处理、测试和三个 FP32 ONNX 模型。

## 当前基线

- Python：建议 Windows x64 Python 3.12。
- GUI：PyQt6。
- CPU 推理：ONNX Runtime + FP32 ONNX 模型。
- NVIDIA 推理：代码已有 `CUDAExecutionProvider` 路径，但 Windows CUDA 依赖和打包还需在真实 Windows/NVIDIA 机器上验证。
- 模型位置：`work/runtime/models/`。
- 回归基线：121 个单元测试在 macOS 交接时全部通过。

## Windows 快速启动

1. 安装 64 位 Python 3.12，并确保 Python Launcher `py` 可用。
2. 在 PowerShell 中进入本目录。
3. 如果 PowerShell 禁止脚本，只对当前进程允许：

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   ```

4. CPU 环境：

   ```powershell
   .\setup_windows_cpu.ps1
   ```

5. 启动 GUI：

   ```powershell
   .\run_gui_windows.bat
   ```

CPU 版可用来先完成截图和打包移植，不需要 CUDA Toolkit。

## NVIDIA CUDA 环境

先确认 NVIDIA 驱动和 ONNX Runtime GPU 要求的 CUDA/cuDNN 运行库版本。然后在新环境中执行：

```powershell
.\setup_windows_cuda.ps1
```

验证 Provider：

```powershell
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

结果应包含 `CUDAExecutionProvider`。如果只有 `CPUExecutionProvider`，程序仍可运行，但没有使用 NVIDIA GPU。

## Windows 移植的首要任务

### 1. 实现跨平台框选截图

当前 `homr_gui.py` 调用 macOS 的 `/usr/sbin/screencapture -i`，Windows 上会明确拒绝启动。建议用 PyQt6 实现一个独立的 `CaptureOverlay`：

- 无边框、置顶、半透明，覆盖整个虚拟桌面。
- 鼠标拖动选择区域，松开后使用 `QScreen.grabWindow()` 获取像素。
- 按 Escape 取消，不触发识别。
- 正确处理多显示器、负坐标和 100%/125%/150%/200% DPI 缩放。
- 不保存永久截图，继续使用 `homr.app_paths.session_dir()` 的一次性临时目录。
- 将 macOS 和 Windows 实现拆成平台 Capture Backend，不要在识别线程里散落平台判断。

现有 `tests/test_gui_capture.py` 是截图行为的回归起点，Windows 实现应增加针对坐标换算、取消和重复框选的测试。

### 2. 不要改动识别后处理语义

以下是已经验证的跨平台逻辑，Windows 移植应直接复用：

- `homr/rhythm_correction.py`
- `homr/barline_recovery.py`
- `homr/midi_timing.py`
- `homr/midi_arpeggio.py`
- `homr/midi_tremolo.py`
- `homr/staff_parsing.py`
- `homr/music_xml_generator.py`

平台适配不应加入针对某一页谱子、曲名、小节号或固定图像坐标的特化规则。

### 3. Windows 打包

建议新增 `packaging/windows/HOMR.spec` 和 `scripts/build_windows_app.ps1`：

- 先做 PyInstaller `onedir` 版，不要一开始使用 `onefile`。
- 包含 `homr/transformer/*.json`、`work/runtime/models/*.onnx` 和 RapidOCR 模型。
- 排除 CoreML、`.mlpackage`、macOS `.app` 和 macOS 签名脚本。
- CPU 和 CUDA 建议发布为两个独立包，避免 GPU DLL 增大所有用户的包体。
- 在干净 Windows 10/11 x64 机器或虚拟机上验证，不能只在开发机测试。
- 对外发布时再制作 Inno Setup/MSIX 安装包和代码签名。

## 必须保持的回归条件

每次平台修改后运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

最低验收项：

- 现有 121 个测试不得因 Windows 移植失败。
- 单钢琴输出两条手部轨道的逻辑不变。
- 双钢琴保持两个 Part/四条乐谱轨道，不得合并为一架钢琴。
- 左右手小节边界对齐，但不强行把两个独立声部改成相同音符序列。
- 框选成功、取消和连续框选都必须测试。
- 识别完成后不永久保存截图、MusicXML 或 MIDI 中间文件。

## Git 基线

交接包内已保留 Git 历史，并在 `windows-dev-baseline` 分支上将当前未发布的修复收敛为一个交接提交。开发者应在此基线之后工作，不要从 GitHub 原始上游重新开始，否则会丢失本项目已完成的节拍和多钢琴修复。
