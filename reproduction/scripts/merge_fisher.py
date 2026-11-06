"""Merge gradient-square sums before sqrt (not average of shard scores)."""
import torch
from common import ROOT, save_json

files = sorted((ROOT / "reproduction/cache").glob("fisher_[0-9]*.pt"))
total = None
count = 0
for file in files:
    state = torch.load(file, map_location="cpu", weights_only=True)
    count += state["count"]
    if total is None:
        total = state["sum_grad_sq"]
    else:
        for name in total:
            total[name].add_(state["sum_grad_sq"][name])
if count != 2048:
    raise ValueError(f"Expected 2048 calibration samples, got {count}")
scores = {n:v.div_(count).sqrt_().mean().item() for n,v in total.items()}
if not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in scores.values()):
    raise ValueError("Invalid Fisher scores")
save_json(ROOT / "reproduction/results/fisher_scores.json", {"count": count, "scores": scores})
print(scores, flush=True)
