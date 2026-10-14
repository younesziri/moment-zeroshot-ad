"""MOMENT zero-shot anomaly scores: reconstruct each 512-point window, nothing masked,
score = squared reconstruction error (as in the authors' detect_anomalies)."""

import numpy as np
import torch
from momentfm import MOMENTPipeline

from zsad.windows import make_windows, stitch

WINDOW = 512 


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_moment(size: str, device: str):
    model = MOMENTPipeline.from_pretrained(
        f"AutonLab/MOMENT-1-{size}", model_kwargs={"task_name": "reconstruction"}
    )
    model.init()
    # huggingface-hub 0.24 (pinned by momentfm 0.1.4) leaves safetensors models in training
    # mode, so dropout would make scores change on every call.
    model.eval()
    return model.to(device)


def moment_scores(model, x: np.ndarray, device: str, batch_size: int = 32):
    """Returns (score, reconstruction), one value per point of x."""
    windows, starts = make_windows(x.astype(np.float32), WINDOW)
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size])[:, None, :].to(device)
            input_mask = torch.ones(batch.shape[0], WINDOW, device=device)  # all points observed
            out = model(x_enc=batch, input_mask=input_mask).reconstruction
            outputs.append(out[:, 0, :].cpu().numpy())
    reconstruction = stitch(np.concatenate(outputs).astype(np.float64), starts, len(x))
    return (x - reconstruction) ** 2, reconstruction
