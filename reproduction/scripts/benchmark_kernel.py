"""Validate and measure the released fused reconstruction/RoPE/QK kernel."""
import argparse
import torch
import triton.testing
from common import ROOT, seed_all, save_json
from kernel.pytorch_reference import LlamaRotaryEmbedding, apply_rotary_pos_emb_pytorch

p = argparse.ArgumentParser()
p.add_argument("--lengths", type=int, nargs="+", default=[4096,16384,65536])
p.add_argument("--tuned", action="store_true")
a = p.parse_args()
if a.tuned:
    import os
    os.environ["PALU_AUTOTUNE"] = "1"
from kernel.abx_rope import abx, torch_abx
seed_all()
rows = []
for gs in [1,4,32]:
    groups = 32//gs
    rank = 1024//groups
    for length in a.lengths:
        q = torch.randn(32, 1, 128, device="cuda", dtype=torch.float16)*.1
        b = torch.randn(32, rank, 128, device="cuda", dtype=torch.float16)*.1
        x = torch.randn(groups, length, rank, device="cuda", dtype=torch.float16)*.1
        ref = torch_abx(q,b,x)
        actual = abx(q,b,x)
        rel = float(torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float()))
        if rel > .005 or not torch.isfinite(actual).all():
            raise ValueError(f"Kernel correctness failed: gs={gs}, length={length}, relative_l2={rel}")
        # Match upstream WX baseline: KV is already reconstructed and RoPE'd.
        keys = (x[:,None] @ b.reshape(groups,gs,rank,128)).reshape(32,length,128)
        cos,sin = LlamaRotaryEmbedding(128,length)
        keys = apply_rotary_pos_emb_pytorch(keys,cos,sin).half()
        baseline = lambda: q @ keys.transpose(-1,-2)
        fn = lambda: abx(q,b,x)
        base_ms = triton.testing.do_bench(baseline, warmup=100, rep=500)
        fused_ms = triton.testing.do_bench(fn, warmup=100, rep=500)
        row = {"group_size":gs,"length":length,"key_rank":1024,"relative_l2":rel,
               "max_abs_error":float((actual-ref).abs().max()),"baseline_ms":base_ms,
               "palu_ms":fused_ms,"speedup":base_ms/fused_ms}
        rows.append(row)
        print(row, flush=True)
        save_json(ROOT / ("reproduction/results/kernel_tuned.json" if a.tuned else "reproduction/results/kernel.json"), {"gpu":torch.cuda.get_device_name(),
                  "torch":torch.__version__,"baseline":"upstream WX GEMV with materialized post-RoPE K",
                  "tuned":a.tuned,
                  "timing":"triton.testing.do_bench warmup=100ms rep=500ms, default mean",
                  "rows":rows})
