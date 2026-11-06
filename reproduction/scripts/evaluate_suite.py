"""Evaluate one saved model on full PPL sets, then optional quantization ablation."""
import argparse
import torch
from common import ROOT, load, evaluate, save_json
from palu.quant_utils import configure_latent_quantizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--name", required=True)
p.add_argument("--quantization", action="store_true")
a = p.parse_args()
model, tokenizer = load(a.model)
for dataset, seqlen, suffix in [("wikitext2",2048,"2048"),("c4",2048,"c4")]:
    result = evaluate(model, tokenizer, seqlen, dataset=dataset)
    result.update(model=a.model, bits=16, hadamard=False)
    if dataset == "wikitext2":
        prompt = "The capital of France is"
        with torch.inference_mode():
            generated = model.generate(**tokenizer(prompt, return_tensors="pt").to("cuda"), max_new_tokens=48,
                                       do_sample=False, temperature=1., top_p=1., pad_token_id=tokenizer.eos_token_id)
        result["sample"] = {"prompt":prompt,"completion":tokenizer.decode(generated[0], skip_special_tokens=True)}
    save_json(ROOT / f"reproduction/results/{a.name}_{suffix}.json", result)
if a.quantization:
    configs = [(16,False),(3,False),(2,False),(3,True),(2,True)]
    transformed = False
    for bits, hadamard in configs:
        # Fuse WHT once, then reuse the rotated weights for both bit widths.
        configure_latent_quantizer(model, bits, group_size=0, sym=False, clip_ratio=1., hadamard=hadamard and not transformed)
        transformed |= hadamard
        result = evaluate(model, tokenizer, 4096)
        result.update(model=a.model,bits=bits,hadamard=hadamard,quantization="per-token asymmetric fake quantization")
        save_json(ROOT / f"reproduction/results/{a.name}_4096_b{bits}_h{int(hadamard)}.json",result)
        print({k:v for k,v in result.items() if k != "nll_per_block"},flush=True)
