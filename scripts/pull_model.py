"""Download a GGUF model published as an OCI artifact on Docker Hub's `ai/` namespace
(the Docker Model Runner catalog), without needing Docker itself.

usage: python scripts/pull_model.py ai/qwen3-reranker 0.6B models/qwen3-reranker-0.6b.gguf [sha256]
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REG = "https://registry-1.docker.io/v2"
ACCEPT = (
    "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)


def main(repo: str, tag: str, out: str, expect: str | None = None) -> None:
    out_p = Path(out)
    if out_p.exists() and expect and sha256(out_p) == expect:
        print(f"{out} already present")
        return
    tok = json.load(
        urllib.request.urlopen(
            f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        )
    )["token"]

    def get(url: str, accept: str | None = None):
        h = {"Authorization": f"Bearer {tok}"}
        if accept:
            h["Accept"] = accept
        return urllib.request.urlopen(urllib.request.Request(url, headers=h))

    m = json.load(get(f"{REG}/{repo}/manifests/{tag}", ACCEPT))
    if "manifests" in m:  # index -> pick the manifest that carries a gguf layer
        subs = [json.load(get(f"{REG}/{repo}/manifests/{x['digest']}", ACCEPT)) for x in m["manifests"]]
        m = next(s for s in subs if any("gguf" in json.dumps(layer) for layer in s.get("layers", [])))
    layer = next(
        x
        for x in m["layers"]
        if "gguf" in x["mediaType"]
        or x.get("annotations", {}).get("org.cncf.model.filepath", "").endswith(".gguf")
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    r = get(f"{REG}/{repo}/blobs/{layer['digest']}")
    h = hashlib.sha256()
    with open(out_p, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
    digest = h.hexdigest()
    if f"sha256:{digest}" != layer["digest"] or (expect and digest != expect):
        out_p.unlink()
        raise SystemExit(f"checksum mismatch for {repo}:{tag}")
    print(f"wrote {out} ({out_p.stat().st_size / 1e9:.2f} GB) sha256={digest}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    main(*sys.argv[1:])
