# src/data/download.py
"""Download BIRD dev (+ mini-dev) into data/bird_raw/.
Usage: python -m src.data.download
BIRD links change occasionally; URLs are read from configs/bird_urls.yaml so they
can be updated without code changes.
"""
from __future__ import annotations
import os
import sys
import urllib.request
import zipfile
import yaml

CONFIG = "configs/bird_urls.yaml"
DEST = "data/bird_raw"

def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    with open(CONFIG) as f:
        urls = yaml.safe_load(f)
    for name, url in urls.items():
        out = os.path.join(DEST, f"{name}.zip")
        if os.path.exists(out):
            print(f"skip {name} (exists)")
            continue
        print(f"downloading {name} <- {url}")
        urllib.request.urlretrieve(url, out)
        with zipfile.ZipFile(out) as z:
            z.extractall(DEST)
    print("done. Inspect data/bird_raw/ and confirm dev.json + database folders exist.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
