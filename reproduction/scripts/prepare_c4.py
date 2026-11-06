"""Stream only the upstream evaluation prefix from the fixed C4 shard."""
import gzip
import hashlib
import json
import requests
from common import ROOT, save_json

revision = "607bd4c8450a42878aa9ddc051a65a055450ef87"
url = f"https://huggingface.co/datasets/allenai/c4/resolve/{revision}/en/c4-validation.00000-of-00008.json.gz"
out = ROOT / "reproduction/data/c4_validation_first1100.jsonl"
if not out.exists():
    response = requests.get(url, stream=True, timeout=180)
    response.raise_for_status()
    response.raw.decode_content = False
    with gzip.GzipFile(fileobj=response.raw) as stream, out.with_suffix(".tmp").open("wb") as target:
        for i, line in enumerate(stream):
            target.write(line)
            if i+1 == 1100:
                break
    out.with_suffix(".tmp").replace(out)
    response.close()
save_json(ROOT / "reproduction/data/c4_manifest.json", {"url":url,"revision":revision,"rows":1100,
          "selection":"first 1100 documents of fixed validation shard, matching upstream",
          "sha256":hashlib.sha256(out.read_bytes()).hexdigest()})
print(out, out.stat().st_size)
