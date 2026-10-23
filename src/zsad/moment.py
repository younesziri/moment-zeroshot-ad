"""MOMENT zero-shot anomaly scores: reconstruct each 512-point window, nothing masked,
score = squared reconstruction error (as in the authors' detect_anomalies)."""

import numpy as np
import torch
from momentfm import MOMENTPipeline

from zsad.data import WINDOW, unwindow, windows


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_moment(size: str, device: str, positional_embedding: bool = True):
    # The released configs don't set add_positional_embedding, so momentfm adds an absolute
    # sinusoidal embedding by default (True). The paper (§3.2) only mentions a relative scheme
    # and the authors' zero-shot config sets it to False, but the released weights need it:
    # without it the reconstruction no longer follows the input (see scripts/diagnose_moment.py).
    model = MOMENTPipeline.from_pretrained(
        f"AutonLab/MOMENT-1-{size}",
        model_kwargs={"task_name": "reconstruction", "add_positional_embedding": positional_embedding},
    )
    model.init()
    # huggingface-hub 0.24 (pinned by momentfm 0.1.4) leaves safetensors models in training
    # mode, so dropout would make scores change on every call.
    model.eval()
    return model.to(device)


def moment_scores(model, x: np.ndarray, device: str, batch_size: int = 32):
    """Returns (score, reconstruction), one value per point of x."""
    x_windows = windows(x.astype(np.float32))
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(x_windows), batch_size):
            batch = torch.from_numpy(x_windows[i : i + batch_size])[:, None, :].to(device)
            input_mask = torch.ones(batch.shape[0], WINDOW, device=device)  # all points observed
            out = model(x_enc=batch, input_mask=input_mask).reconstruction
            outputs.append(out[:, 0, :].cpu().numpy())
    reconstruction = unwindow(np.concatenate(outputs).astype(np.float64), len(x))
    return (x - reconstruction) ** 2, reconstruction


def masked_scores(model, x: np.ndarray, device: str, k: int = 4, batch_size: int = 32):
    """Masked scoring (analysis plan, section 2): k passes per window; pass j hides patches
    j, j + k, j + 2k, ... (mask = 0 there, input_mask all ones), and each point is reconstructed by
    the pass that hid its patch. Returns (score, reconstruction)."""
    x_windows = windows(x.astype(np.float32))
    patch = np.arange(WINDOW) // model.config.patch_len
    out = np.zeros_like(x_windows)
    with torch.inference_mode():
        for j in range(k):
            hidden = patch % k == j
            for i in range(0, len(x_windows), batch_size):
                batch = torch.from_numpy(x_windows[i : i + batch_size])[:, None, :].to(device)
                ones = torch.ones(batch.shape[0], WINDOW, device=device)
                mask = ones * torch.from_numpy(~hidden).float().to(device)
                pred = model(x_enc=batch, input_mask=ones, mask=mask).reconstruction[:, 0, :].cpu().numpy()
                out[i : i + batch_size, hidden] = pred[:, hidden]
    reconstruction = unwindow(out.astype(np.float64), len(x))
    return (x - reconstruction) ** 2, reconstruction
