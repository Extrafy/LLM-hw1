import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)

def load(model_path=None):
    seed_all()
    path = str(model_path or ROOT / "model")
    import palu.model  # register custom HF models
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16,
                                              attn_implementation="sdpa", low_cpu_mem_usage=True).cuda().eval()
    return model, tokenizer

def text(split):
    return "\n\n".join(pd.read_parquet(ROOT / f"reproduction/data/{split}-00000-of-00001.parquet")["text"])

def calib_batches(tokenizer, nsamples, seqlen, seed=3):
    # Preserve upstream's character-window sampling, not random token offsets.
    rng = random.Random(seed)
    corpus = text("train")
    for _ in range(nsamples):
        i = rng.randint(0, len(corpus) - seqlen - 1)
        ids = tokenizer(corpus[i:i + seqlen * 10], return_tensors="pt").input_ids[:, :seqlen]
        if ids.shape[1] != seqlen:
            raise ValueError("Calibration sample shorter than requested sequence length")
        yield ids

def save_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    temp.replace(path)

@torch.inference_mode()
def evaluate(model, tokenizer, seqlen=2048, max_blocks=0, dataset="wikitext2"):
    if dataset == "wikitext2":
        corpus = text("test")
    elif dataset == "c4":
        docs = [json.loads(line) for line in (ROOT / "reproduction/data/c4_validation_first1100.jsonl").read_text().splitlines()]
        corpus = " ".join(d["text"] for d in docs)
    else:
        raise ValueError(dataset)
    ids = tokenizer(corpus, return_tensors="pt").input_ids
    if dataset == "c4":
        ids = ids[:, :256*seqlen]
    blocks = ids.numel() // seqlen
    if max_blocks:
        blocks = min(blocks, max_blocks)
    losses = []
    start = time.perf_counter()
    peak_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for i in range(blocks):
        batch = ids[:, i * seqlen:(i + 1) * seqlen].cuda()
        hidden = model.model(batch, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden)
        # FP32 CE avoids fp16 loss-rounding; same nonoverlapping upstream windows.
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        losses.append(loss.item())
        if (i+1) % 10 == 0 or i+1 == blocks:
            print(f"PPL blocks {i+1}/{blocks}: {np.exp(np.mean(losses)):.6f}", flush=True)
    torch.cuda.synchronize()
    return {"ppl": float(np.exp(np.mean(losses))), "dataset":dataset, "seqlen": seqlen, "blocks": blocks,
            "predicted_tokens": blocks * (seqlen-1), "total_tokenized_tokens": ids.numel(),
            "dropped_tokens": ids.numel() - blocks * seqlen, "nll_per_block": losses,
            "seconds": time.perf_counter()-start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "model_allocated_bytes": peak_before,
            "token_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
