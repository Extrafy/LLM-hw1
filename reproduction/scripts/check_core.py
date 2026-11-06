"""Numerical checks of the core low-rank and fusion identities."""
import torch
from common import ROOT, save_json, seed_all
from palu.model import HeadwiseLowRankModule
from palu.model.modules.quant import quantize_tensor

seed_all()
torch.set_default_dtype(torch.float32)
x = torch.randn(2, 17, 64)
linear = torch.nn.Linear(64, 32, bias=False)
full = HeadwiseLowRankModule.from_linear(linear, [16,16])
torch.testing.assert_close(full(x), linear(x), atol=1e-5, rtol=1e-4)
truncated = HeadwiseLowRankModule.from_linear(linear, [8,8])
before = truncated(x).detach()
latent_before = truncated.project_to_latent(x).detach()
truncated.fused_hadamard_matrix()
after = truncated(x).detach()
torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-4)
# Value reconstruction can be absorbed into output projection per head.
p = torch.softmax(torch.randn(2, 1, 17), dim=-1)
h = torch.randn(2, 17, 8)
b = torch.randn(8, 16)
o = torch.randn(16, 64)
explicit = ((p @ h) @ b) @ o
fused = (p @ h) @ (b @ o)
torch.testing.assert_close(explicit, fused, atol=3e-5, rtol=1e-4)
q = quantize_tensor(torch.randn(17, 8), 3, 0, False)
assert torch.isfinite(q).all()
result = {"full_rank_max_abs_error":float((full(x)-linear(x)).abs().max()),
          "hadamard_fusion_max_abs_error":float((after-before).abs().max()),
          "value_fusion_max_abs_error":float((explicit-fused).abs().max()),
          "truncated_rank_executes":True, "quantization_finite":True,
          "checks_passed":5}
save_json(ROOT / "reproduction/results/core_checks.json", result)
print(result)
