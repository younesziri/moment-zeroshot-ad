"""Step-2 feasibility check: can this machine run MOMENT zero-shot?

Downloads the real MOMENT-1 weights from Hugging Face (small ~0.15 GB,
base ~0.45 GB, large ~1.4 GB, cached in ~/.cache/huggingface), then for each size:
  - checks the model is in eval mode and the config does not mask patches at inference,
  - reconstructs the tiltAPB2 test series (154 windows of 512) on CPU and,
    if available, on the Apple GPU (MPS),
  - reports run time, peak memory, and whether MPS agrees with CPU.
No anomaly scores are computed here; that is step 3.

Usage:  uv run python scripts/check_setup.py                 # all three sizes
        uv run python scripts/check_setup.py --sizes base
"""

import argparse
import gc
import platform
import resource
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline

DATA = "data/198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out"
N_TRAIN = 50_000  # end of the anomaly-free training part, from the UCR file name
WINDOW = 512


def load_test_windows() -> torch.Tensor:
    """Test part of tiltAPB2, z-scored with train statistics, cut into
    non-overlapping windows (the last one aligned to the end of the series)."""
    values = pd.read_csv(DATA, header=None)[0].to_numpy(dtype=np.float32)
    train, test = values[:N_TRAIN], values[N_TRAIN:]
    test = (test - train.mean()) / train.std()
    starts = list(range(0, len(test) - WINDOW + 1, WINDOW))
    if starts[-1] != len(test) - WINDOW:
        starts.append(len(test) - WINDOW)
    return torch.from_numpy(np.stack([test[s : s + WINDOW] for s in starts]))[:, None, :]


def reconstruct(model, windows: torch.Tensor, device: str, batch_size: int = 32) -> torch.Tensor:
    model.to(device)
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(windows), batch_size):
            x = windows[i : i + batch_size].to(device)
            input_mask = torch.ones(x.shape[0], WINDOW, device=device)
            outputs.append(model(x_enc=x, input_mask=input_mask).reconstruction.cpu())
    return torch.cat(outputs)


def timed_reconstruct(model, windows: torch.Tensor, device: str):
    reconstruct(model, windows[:4], device)  # warm-up (first MPS call compiles kernels)
    start = time.perf_counter()
    out = reconstruct(model, windows, device)
    return out, time.perf_counter() - start


def peak_memory_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, KB on Linux


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["small", "base", "large"], choices=["small", "base", "large"])
    args = parser.parse_args(argv)

    print(f"{platform.platform()} | Python {platform.python_version()}")
    print(" | ".join(f"{p} {version(p)}" for p in ["torch", "numpy", "transformers", "huggingface-hub", "momentfm"]))
    has_mps = torch.backends.mps.is_available()
    print(f"MPS (Apple GPU) available: {has_mps}")

    windows = load_test_windows()
    print(f"tiltAPB2 test part: {windows.shape[0]} windows of {WINDOW} points")

    for size in args.sizes:  # small -> large, so peak memory after each line is that size's
        start = time.perf_counter()
        model = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{size}", model_kwargs={"task_name": "reconstruction"}
        )
        model.init()
        # huggingface-hub 0.24 (pinned by momentfm 0.1.4) only calls eval() when loading
        # pickle weights, not safetensors, so the model comes back with dropout ON.
        left_in_training_mode = model.training
        model.eval()
        load_s = time.perf_counter() - start
        n_params = sum(p.numel() for p in model.parameters())
        mask_ratio = model.config.getattr("mask_ratio", None)

        print(f"\n== MOMENT-1-{size}: {n_params / 1e6:.1f}M parameters, loaded in {load_s:.0f}s (first run includes download)")
        print(f"   from_pretrained left dropout on: {left_in_training_mode} -> called model.eval(); now in eval mode: {not model.training}")
        print(f"   mask_ratio in config: {mask_ratio}  (None -> momentfm default 0.0: no patch is masked at inference)")

        cpu_out, cpu_s = timed_reconstruct(model, windows, "cpu")
        a, b = reconstruct(model, windows[:8], "cpu"), reconstruct(model, windows[:8], "cpu")
        print(f"   CPU: {cpu_s:.1f}s for the series ({1000 * cpu_s / len(windows):.0f} ms/window); "
              f"two identical calls give identical output: {torch.equal(a, b)}")

        if has_mps:
            try:
                mps_out, mps_s = timed_reconstruct(model, windows, "mps")
                diff = (mps_out - cpu_out).abs().max().item()
                print(f"   MPS: {mps_s:.1f}s ({1000 * mps_s / len(windows):.0f} ms/window); max |MPS - CPU| = {diff:.1e}")
            except Exception as e:  # report, don't hide: an unsupported op on MPS is a finding
                print(f"   MPS failed: {type(e).__name__}: {e}")
            model.to("cpu")
            torch.mps.empty_cache()

        print(f"   peak process memory so far: {peak_memory_gb():.2f} GB")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
