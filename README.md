# HOMR GUI (Music Score to MIDI)

本项目是基于 [liebharc/homr](https://github.com/liebharc/homr) 的 GUI 封装与增强版本。

### 主要改进：
* **可视化界面**：增加了基于 PyQt6 的交互界面，告别命令行。
* **模式切换**：支持 CPU 兼容模式与 GPU 加速模式一键切换。
* **国内优化**：优化了模型下载逻辑，使用镜像代理提升下载速度。
* **Apple Silicon**：支持原生 arm64 `.app`，并自动使用经过回归验证的
  CoreML 分割网络 + CPU Transformer 混合推理。

### macOS

macOS 构建、运行目录、性能基准和 Apple Neural Engine 实测结论见
[docs/MACOS.md](docs/MACOS.md)。原生 Core ML SegNet 已由实际桌面进程的
`MLComputePlan` 验证调度 Apple Neural Engine；动态 Transformer 仍保留在 CPU。

### 开源协议：
本项目遵循 **GNU AGPLv3** 协议。
