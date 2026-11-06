"""Memory-bounded execution of upstream Fisher and SVD-LLM calibration."""
import argparse
import time
import torch
from common import ROOT, load, calib_batches, save_json

p = argparse.ArgumentParser()
p.add_argument("--kind", choices=["fisher", "whiten"], required=True)
p.add_argument("--samples", type=int, default=2048)
p.add_argument("--seqlen", type=int, default=1024)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--shards", type=int, default=1)
a = p.parse_args()
model, tokenizer = load()
start = time.perf_counter()
out = ROOT / "reproduction/cache"
out.mkdir(exist_ok=True)
metadata = vars(a).copy()
metadata.update(seed=3, model=str(ROOT / "model"), loss="upstream double shift" if a.kind == "fisher" else None)
if a.kind == "fisher":
    targets = {n:m for n,m in model.named_modules() if isinstance(m, torch.nn.Linear) and ("k_proj" in n or "v_proj" in n)}
    model.requires_grad_(False)
    for m in targets.values():
        m.weight.requires_grad_(True)
    accum = {n:torch.zeros_like(m.weight, dtype=torch.float32) for n,m in targets.items()}
    count = 0
    for i, ids in enumerate(calib_batches(tokenizer, a.samples, a.seqlen)):
        if i % a.shards != a.shard:
            continue
        ids = ids.cuda()
        # The upstream code shifts labels before HF shifts again. Preserve this
        # behavior for rank-search comparability, and document it explicitly.
        result = model(input_ids=ids[:, :-1], labels=ids[:, 1:], use_cache=False)
        result.loss.backward()
        for n,m in targets.items():
            accum[n].add_(m.weight.grad.detach().float().square())
        model.zero_grad(set_to_none=True)
        count += 1
        if count % 16 == 0:
            print(f"fisher shard {a.shard}: {count}/{a.samples//a.shards}, elapsed {time.perf_counter()-start:.1f}s, loss {result.loss.item():.5f}", flush=True)
    torch.save({"sum_grad_sq": {n:v.cpu() for n,v in accum.items()}, "count": count, "metadata": metadata}, out / f"fisher_{a.shard}.pt")
    metadata["count"] = count
else:
    grams = [torch.zeros((model.config.hidden_size, model.config.hidden_size), device="cuda", dtype=torch.float32) for _ in model.model.layers]
    handles = []
    for index, layer in enumerate(model.model.layers):
        def hook(module, inputs, output, index=index):
            x = inputs[0].detach().reshape(-1, model.config.hidden_size).float()
            grams[index].addmm_(x.T, x)
        handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
    with torch.inference_mode():
        for i, ids in enumerate(calib_batches(tokenizer, a.samples, a.seqlen)):
            model.model(ids.cuda(), use_cache=False)
            if (i+1) % 16 == 0:
                print(f"whiten {i+1}/{a.samples}, elapsed {time.perf_counter()-start:.1f}s", flush=True)
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    scales = []
    regularizers = []
    for i, gram in enumerate(grams):
        raw = gram.double()
        L, info = torch.linalg.cholesky_ex(raw)
        reg = 0.
        if info.item():
            reg = float(-torch.linalg.eigvalsh(raw)[0].item()+1e-3)
            raw.diagonal().add_(reg)
            L = torch.linalg.cholesky(raw)
        scales.append(L.float().cpu())
        regularizers.append(reg)
        print(f"cholesky layer {i}: diagonal correction={reg}", flush=True)
    torch.save(scales, out / "whiten.pt")
    metadata["diagonal_regularizers"] = regularizers
metadata["seconds"] = time.perf_counter()-start
metadata["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
save_json(ROOT / f"reproduction/results/calibration_{a.kind}_{a.shard}.json", metadata)
