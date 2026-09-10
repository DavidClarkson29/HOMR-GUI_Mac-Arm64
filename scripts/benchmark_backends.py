from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_backend(source: Path, destination: Path, gpu_mode: str) -> dict[str, object]:
    shutil.copy2(source, destination)
    environment = os.environ.copy()
    environment.setdefault("HOMR_COREML_COMPUTE_UNITS", "CPUAndNeuralEngine")

    started = perf_counter()
    process = subprocess.run(  # noqa: S603 - arguments are not evaluated by a shell
        [sys.executable, "-m", "homr.main", str(destination), "--gpu", gpu_mode],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = perf_counter() - started
    output = destination.with_suffix(".musicxml")
    if process.returncode != 0 or not output.exists():
        raise RuntimeError(
            f"Backend {gpu_mode} failed with exit code {process.returncode}:\n"
            f"{process.stdout}\n{process.stderr}"
        )

    return {
        "elapsed_seconds": round(elapsed, 4),
        "musicxml_sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HOMR CPU output against the macOS CoreML/CPU hybrid pipeline."
    )
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    source = args.image.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Image does not exist: {source}")

    with tempfile.TemporaryDirectory(prefix="homr-backend-benchmark-") as directory:
        temporary = Path(directory)
        cpu = _run_backend(source, temporary / f"cpu{source.suffix}", "no")
        coreml = _run_backend(source, temporary / f"coreml{source.suffix}", "force")

    cpu_time = float(cpu["elapsed_seconds"])
    coreml_time = float(coreml["elapsed_seconds"])
    result = {
        "cpu": cpu,
        "coreml_hybrid": coreml,
        "outputs_identical": cpu["musicxml_sha256"] == coreml["musicxml_sha256"],
        "speedup": round(cpu_time / coreml_time, 4),
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
