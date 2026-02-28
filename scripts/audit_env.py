from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e}"


def module_version(name: str) -> str:
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "NOT_INSTALLED"


def main() -> None:
    data = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "gpu_processes": run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"]),
        "disk_root": run(["df", "-h", "/"]),
        "memory": run(["free", "-h"]),
        "modules": {
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "trl": module_version("trl"),
            "peft": module_version("peft"),
            "datasets": module_version("datasets"),
            "accelerate": module_version("accelerate"),
            "bitsandbytes": module_version("bitsandbytes"),
        },
        "models_present": {
            "Qwen3-8B": Path("/root/Qwen3-8B").exists(),
            "Qwen3-14B": Path("/root/Qwen3-14B").exists(),
        },
        "docker": shutil.which("docker") is not None,
    }
    print(json.dumps(data, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
