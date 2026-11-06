import argparse
from common import ROOT, load, evaluate, save_json
from palu.quant_utils import configure_latent_quantizer

p = argparse.ArgumentParser()
p.add_argument("--model", default=str(ROOT / "model"))
p.add_argument("--output", required=True)
p.add_argument("--seqlen", type=int, default=2048)
p.add_argument("--max-blocks", type=int, default=0)
p.add_argument("--bits", type=int, default=16)
p.add_argument("--hadamard", action="store_true")
p.add_argument("--sample", action="store_true")
p.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
a = p.parse_args()
model, tokenizer = load(a.model)
if a.bits < 16:
    configure_latent_quantizer(model, a.bits, group_size=0, sym=False, clip_ratio=1., hadamard=a.hadamard)
result = evaluate(model, tokenizer, a.seqlen, a.max_blocks, a.dataset)
result.update(vars(a))
if a.sample:
    import torch
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    result["sample"] = {"prompt": prompt, "completion": tokenizer.decode(generated[0], skip_special_tokens=True)}
save_json(a.output, result)
print({k:v for k,v in result.items() if k != "nll_per_block"}, flush=True)
