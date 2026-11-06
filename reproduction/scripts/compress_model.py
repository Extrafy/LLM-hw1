import argparse
import gc
import json
import math
import time
from types import SimpleNamespace
import torch
from common import ROOT, load, save_json
from palu.model import HeadwiseLowRankModule
from palu.rank_search import rounding_search_result, split_values
from utils import dump_to_huggingface_repos

def allocate(scores, group_size, ratio, uniform=False):
    if uniform:
        ranks = {n:[512 * ratio] * 8 for n in scores} if group_size == 4 else {n:[128*group_size*ratio]*(32//group_size) for n in scores}
        return rounding_search_result(ranks)
    budget = len(scores)*4096*ratio
    denominator = sum(scores.values())
    floats = {n:budget*v/denominator for n,v in scores.items()}
    ranks = {n:min(4096, math.floor(v)) for n,v in floats.items()}
    ordered = sorted(scores, key=lambda n:floats[n]-ranks[n])
    remaining = round(budget-sum(ranks.values()))
    while remaining > 0:
        eligible = [n for n in ordered if ranks[n] < 4096]
        if not eligible:
            raise ValueError("Infeasible rank budget")
        for n in eligible:
            ranks[n] += 1
            remaining -= 1
            if remaining == 0:
                break
    return rounding_search_result(split_values({n:[v] for n,v in ranks.items()}, 32//group_size))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group-size", type=int, default=4, choices=[1,2,4,8,16,32])
    p.add_argument("--ratio", type=float, default=.5, help="retained KV latent fraction before rounding")
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if not 0 < a.ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    model, tokenizer = load()
    scores = ({f"model.layers.{i}.self_attn.{kind}":1. for i in range(32) for kind in ("k_proj","v_proj")}
              if a.uniform else json.loads((ROOT / "reproduction/results/fisher_scores.json").read_text())["scores"])
    ranks = allocate(scores, a.group_size, a.ratio, a.uniform)
    scales = torch.load(ROOT / "reproduction/cache/whiten.pt", map_location="cpu", weights_only=True)
    started = time.perf_counter()
    for i, layer in enumerate(model.model.layers):
        scale = scales[i].cuda()
        for kind in ("k_proj", "v_proj"):
            n = f"model.layers.{i}.self_attn.{kind}"
            raw = getattr(layer.self_attn, kind)
            raw.scaling_diag_matrix = scale
            compressed = HeadwiseLowRankModule.from_linear_whiten(raw, ranks[n])
            setattr(layer.self_attn, kind, compressed)
            del raw
        print(f"compressed layer {i+1}/32, elapsed {time.perf_counter()-started:.1f}s", flush=True)
    actual = sum(map(sum,ranks.values()))/(len(scores)*4096)
    metadata = {**vars(a), "ranks":ranks, "retained_fraction": actual, "decompose_seconds":time.perf_counter()-started,
                "kv_fp16_bytes_128k":sum(map(sum,ranks.values()))*131072*2,
                "note":"Latent cache size is theoretical; HF accuracy path reconstructs full KV cache."}
    save_json(Path(a.output)/"reproduction.json", metadata)
    dump_to_huggingface_repos(model, tokenizer, a.output, SimpleNamespace())
    save_json(ROOT / f"reproduction/results/{Path(a.output).name}_compression.json", metadata)
    print(f"Saved {a.output}, actual KV retained fraction={actual}", flush=True)

if __name__ == "__main__":
    from pathlib import Path
    main()
