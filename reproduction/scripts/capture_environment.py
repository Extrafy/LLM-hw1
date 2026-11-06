import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import torch
from common import ROOT, save_json

packages = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}
result = {"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
          "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
          "gpus":[{"name":torch.cuda.get_device_name(i),"memory_bytes":torch.cuda.get_device_properties(i).total_memory}
                  for i in range(torch.cuda.device_count())],"packages":packages,
          "repo_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
          "device_details":subprocess.check_output(["nvidia-smi"],text=True),
          "cpu":subprocess.check_output(["lscpu"],text=True),
          "uv":"/home/wangfy26/miniconda3/envs/LLM-hw1/bin/uv",
          "environment":"conda environment LLM-hw1; commands use its global uv and Python"}
save_json(ROOT / "reproduction/results/environment.json",result)
manifest = {}
for path in sorted((ROOT / "model").iterdir()):
    if path.suffix in (".json", ".safetensors", ".model"):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):
                h.update(block)
        manifest[path.name]={"bytes":path.stat().st_size,"sha256":h.hexdigest()}
save_json(ROOT / "reproduction/results/model_manifest.json",manifest)
print("Environment and model manifest saved.")
