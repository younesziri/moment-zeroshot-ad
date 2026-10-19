"""Why only MOMENT-1-large works zero-shot with the released weights.

For each released size (and an untrained small as reference):
  - loading: does the checkpoint load with strict=True? huggingface-hub 0.24 defaults to
    strict=False, which would silently leave mismatched layers at their random initialisation.
    Checked with the positional embedding on (momentfm's default) and off (the authors' config);
  - reconstruction of a clean sine wave: corr(x, x̂), positional embedding on and off;
  - reconstruction head, read directly from the .safetensors file (no momentfm involved):
    momentfm initialises it with zero bias and orthogonal rows scaled by 1.41, so W·Wᵀ = 1.41²·I.
    Values still at 0 below mean the head matches that initialisation exactly (never trained).

Usage:  uv run python scripts/diagnose_moment.py
Output: results/moment_diagnostics.md
"""

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache
from momentfm import MOMENTPipeline
from safetensors import safe_open

from zsad.moment import load_moment, moment_scores

SIZES = ["small", "base", "large"]
SINE = np.sin(2 * np.pi * np.arange(2048) / 50)  # 4 windows of an easy, clean signal


def loads_strictly(size: str, positional_embedding: bool) -> str:
    try:
        MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{size}", strict=True,
            model_kwargs={"task_name": "reconstruction", "add_positional_embedding": positional_embedding})
        return "yes"
    except RuntimeError as e:
        unexpected = [line.strip() for line in str(e).splitlines() if "Unexpected" in line]
        return "no: " + (unexpected[0] if unexpected else str(e).splitlines()[0])


def untrained_small():
    """MOMENT-1-small built from its config with momentfm's own initialisation, no weights loaded."""
    with open(try_to_load_from_cache("AutonLab/MOMENT-1-small", "config.json")) as f:
        config = json.load(f)
    torch.manual_seed(0)
    model = MOMENTPipeline(config, model_kwargs={"task_name": "reconstruction"})
    model.init()
    return model.eval()


def sine_corr(model) -> str:
    return f"{np.corrcoef(SINE, moment_scores(model, SINE, 'cpu')[1])[0, 1]:.3f}"


def head_from_init(W: torch.Tensor, b: torch.Tensor) -> list[str]:
    W = W.detach().double()  # 8 x d_model
    orth = (W @ W.T - 1.41**2 * torch.eye(W.shape[0], dtype=torch.float64)).abs().max().item()
    return [f"{b.detach().abs().max().item():.4f}", f"{orth:.4f}"]


def checkpoint_head(size: str) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Hugging Face revision (the snapshot folder's name) and the head tensors, read from the file."""
    path = Path(try_to_load_from_cache(f"AutonLab/MOMENT-1-{size}", "model.safetensors"))
    with safe_open(path, "pt") as f:
        return path.parent.name, f.get_tensor("head.linear.weight"), f.get_tensor("head.linear.bias")


def main():
    header = ["Model", "HF revision", "Loads strictly, emb. on", "Loads strictly, emb. off", "corr(sine), emb. on",
              "corr(sine), emb. off", "max abs(head bias), file", "max abs(W·Wᵀ − 1.41²·I), file"]
    rows = []
    for size in SIZES:
        on, off = load_moment(size, "cpu"), load_moment(size, "cpu", positional_embedding=False)
        revision, W, b = checkpoint_head(size)
        rows.append([f"MOMENT-1-{size}", revision[:10], loads_strictly(size, True), loads_strictly(size, False),
                     sine_corr(on), sine_corr(off), *head_from_init(W, b)])
    model = untrained_small()
    rows.append(["small, untrained (no weights)", "–", "–", "–", sine_corr(model), "–",
                 *head_from_init(model.head.linear.weight, model.head.linear.bias)])

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    text = "Produced by `scripts/diagnose_moment.py`.\n\n" + "\n".join(lines) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/moment_diagnostics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
