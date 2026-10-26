"""Timestral zero-shot anomaly scores (analysis plan, D11). Timestral is a forecaster, so the score is
the forecast error: x is cut into blocks of 64 points, each block is forecast from the 2048 points
before it (taken from the history for the first blocks), score = (x − median forecast)²."""

import numpy as np
from timestral import TimestralForecaster

# Hugging Face revisions recorded on 26/10/2026, so that an update of the weights can't change results.
REVISIONS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}


def load_timestral(size: str, device: str):
    return TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=device,
                                               revision=REVISIONS[size])


def timestral_scores(model, history: np.ndarray, x: np.ndarray, block: int = 64, context: int = 2048,
                     batch_size: int = 64):
    """Returns (score, scaled), one value per point of x. scaled = |x − median| / (q0.9 − q0.1) is
    exploratory. Every block is forecast from observed values only, never from earlier forecasts."""
    series = np.r_[history, x].astype(np.float32)
    # NaN marks a missing value for Timestral: blocks with fewer than `context` points before them
    # see a left-padded context instead of a shorter one, so every block fits in one batch.
    padded = np.r_[np.full(context, np.nan, np.float32), series]
    starts = np.arange(len(history), len(series), block)
    contexts = np.stack([padded[s : s + context] for s in starts])  # = series[s − context : s]
    median, spread = [], []
    for i in range(0, len(contexts), batch_size):
        f = model.forecast(contexts[i : i + batch_size], prediction_length=block)
        median.append(f.median[:, 0].cpu().numpy())
        spread.append((f.quantile(0.9) - f.quantile(0.1))[:, 0].cpu().numpy())
    median = np.concatenate(median).ravel()[: len(x)].astype(np.float64)
    spread = np.concatenate(spread).ravel()[: len(x)].astype(np.float64)
    return (x - median) ** 2, np.abs(x - median) / np.maximum(spread, 1e-8)
