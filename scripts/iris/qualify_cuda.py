"""Qualify eager FP32 IRIS numerics and sampling on one CUDA GPU.

Run from the checkout root with python -m scripts.iris.qualify_cuda. This is
correctness qualification, not a timing benchmark or full-server test.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.iris.reference_harness import REFERENCE_COMMIT, download_checkpoint, run, verify_checkpoint


def git_output(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def indexed_cuda_device(device="cuda"):
    """Resolve the same current GPU `.to('cuda')` uses for engine device selection.

    The engine's graph backend calls set_device_index, which requires an ordinal.
    Keep explicit ordinals intact, including logical CUDA_VISIBLE_DEVICES indices.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def require_cuda():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("A working NVIDIA CUDA PyTorch build and GPU are required; no CPU fallback is used")
    import triton
    import triton.compiler  # noqa: F401

    if not getattr(triton, "__file__", None):
        raise RuntimeError("A real Triton installation is required, not a test stub")
    # Detect an unusable driver/wheel combination before downloading weights.
    torch.ones(1, device="cuda").add_(1).cpu()


def prepare_build_tools():
    # Running a venv's Python directly does not activate its scripts directory
    # for JIT compilers launching tools by name in child processes.
    scripts = str(Path(sys.executable).parent)
    os.environ["PATH"] = os.pathsep.join((scripts, os.environ.get("PATH", os.defpath)))
    ninja = shutil.which("ninja")
    if ninja is None:
        raise RuntimeError(
            f"ninja is not installed or executable; install it with: uv pip install --python {sys.executable} ninja"
        )
    version = subprocess.check_output([ninja, "--version"], text=True).strip()
    return {"ninja": ninja, "ninja_version": version}


def qualify(args, report, save):
    report["stage"] = "cuda_preflight"
    save()
    require_cuda()
    report["hardware"] = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "platform": platform.platform(),
    }
    try:
        report["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
    except OSError as exc:
        report["nvidia_smi"] = str(exc)
    report["stage"] = "build_tools_preflight"
    save()
    report["build_tools"] = prepare_build_tools()
    save()
    from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope
    from scripts.iris.qualification import compare_trace, sampling_smoke

    report["stage"] = "sampling_kernel_preflight"
    save()
    from mstar.engine.resources.sampler.utils import sample_tokens

    report["hardware"]["flashinfer"] = importlib.metadata.version("flashinfer-python")
    for vocab in (2, 3, 512):
        sampled = sample_tokens(
            torch.zeros(2, vocab, device="cuda"),
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            any_greedy=False,
            top_k_zero_count=2,
            seed=torch.tensor([1, 2], device="cuda"),
            rand_offset=torch.zeros(2, dtype=torch.long, device="cuda"),
        )
        assert sampled.shape == (2,) and ((sampled >= 0) & (sampled < vocab)).all()

    report["stage"] = "reference_and_checkpoint"
    save()
    commit = git_output("rev-parse", "HEAD", cwd=args.reference_root)
    if commit != REFERENCE_COMMIT:
        raise ValueError(f"official source must be pinned at {REFERENCE_COMMIT}, got {commit}")
    if args.download_checkpoint:
        download_checkpoint(args.checkpoint)
    report["checkpoint_sha256"] = verify_checkpoint(args.checkpoint)
    torch.set_num_threads(2)
    # M* imports set a process default; both implementations explicitly use
    # full FP32 for this comparison, including the reference trace generation.
    with fp32_precision_scope(Fp32Precision("highest", False)):
        report["parity"] = []
        for batch in (1, 2):
            reference_output = args.output / f"official-batch-{batch}"
            report["stage"] = f"official_cuda_trace_batch_{batch}"
            save()
            run(
                SimpleNamespace(
                    reference_root=args.reference_root,
                    checkpoint=args.checkpoint,
                    device="cuda",
                    threads=2,
                    seed=0,
                    observations=None,
                    actions=None,
                    batch_size=batch,
                    horizon=20,
                    output=reference_output,
                    compare=None,
                )
            )
            with np.load(reference_output / "trace.npz", allow_pickle=False) as data:
                trace = dict(data)
            report["stage"] = f"mstar_teacher_forced_batch_{batch}"
            save()
            report["parity"].append(compare_trace(args.checkpoint, trace, device="cuda", batch_size=batch))
        report["sampling"] = []
        for batch in (1, 2):
            report["stage"] = f"mstar_real_sampling_batch_{batch}"
            save()
            report["sampling"].append(sampling_smoke(args.checkpoint, batch_size=batch))
    report.update(status="passed", stage="complete")
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path(".ci-results/iris/reference"))
    parser.add_argument("--checkpoint", type=Path, default=Path(".ci-results/iris/Breakout.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "stage": "initialization",
        "started_utc": datetime.now(UTC).isoformat(),
        "purpose": "eager FP32 numerical and sampling qualification; no performance claim",
        "reference_commit": REFERENCE_COMMIT,
        "python": platform.python_version(),
        "capture": "disabled: eager-only reference attention",
        "server": "not exercised",
        "tolerances": {
            "world_rtol": 2e-5,
            "world_atol": 1e-5,
            "rgb_rtol": 2e-5,
            "rgb_atol": 5e-5,
            "codes_rewards_endings": "exact",
            "rgb24_max_absolute_error": 1,
        },
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        report["mstar_commit"] = git_output("rev-parse", "HEAD")
        report["working_tree_status"] = git_output("status", "--porcelain", "--untracked-files=no")
        qualify(args, report, save)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        print(report["traceback"])
        print(f"FAILED at {report['stage']}; send {args.output / 'report.json'}")
        return 1
    print(f"PASSED; send {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
