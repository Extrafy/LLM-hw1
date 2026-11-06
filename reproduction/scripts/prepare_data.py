"""Download pinned WikiText-2 raw parquet files and record provenance."""
import hashlib
import json
from pathlib import Path
import requests
import pandas as pd

root = Path(__file__).resolve().parents[2]
out = root / "reproduction/data"
out.mkdir(exist_ok=True, parents=True)
repo = "Salesforce/wikitext"
revision = "b08601e04326c79dfdd32d625aee71d232d685c3"
manifest = {"repo": repo, "revision": revision, "config": "wikitext-2-raw-v1", "files": {}}
for split in ("train", "validation", "test"):
    filename = f"{split}-00000-of-00001.parquet"
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/wikitext-2-raw-v1/{filename}"
    target = out / filename
    if not target.exists():
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        target.write_bytes(response.content)
    frame = pd.read_parquet(target)
    manifest["files"][split] = {"url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "rows": len(frame)}
    print(split, len(frame), flush=True)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
