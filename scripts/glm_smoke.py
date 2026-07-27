#!/usr/bin/env python3
"""Smoke-test the self-hosted GLM-5.2 vLLM endpoint. Run on the LUMI LOGIN node (it can reach the
compute node serving the model over the internal network).

  python scripts/glm_smoke.py                       # reads logs/glm_endpoint.txt
  python scripts/glm_smoke.py --api-base http://nidXXXX:8000/v1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--prompt", default="Vad är meningen med livet? Svara kort på svenska.")
    ap.add_argument("--wait", type=int, default=0, help="seconds to poll /health before giving up")
    a = ap.parse_args()

    base = a.api_base
    if not base:
        f = pathlib.Path(__file__).resolve().parent.parent / "logs" / "glm_endpoint.txt"
        base = f.read_text().strip()
    base = base.rstrip("/")
    print(f"[glm] endpoint: {base}")

    # 1) health (vLLM exposes /health at the server root, not under /v1)
    root = base[:-3] if base.endswith("/v1") else base
    deadline = time.time() + a.wait
    while True:
        try:
            code, _ = _get(root + "/health")
            print(f"[glm] /health -> {code}")
            break
        except Exception as e:
            if time.time() >= deadline:
                raise SystemExit(f"[glm] server not healthy: {e}")
            print(f"[glm] waiting for server... ({type(e).__name__})"); time.sleep(15)

    # 2) list models
    code, body = _get(base + "/models")
    print("[glm] /models ->", json.loads(body).get("data", [{}])[0].get("id"))

    # 3) a real chat completion
    payload = json.dumps({"model": a.model,
                          "messages": [{"role": "user", "content": a.prompt}],
                          "max_tokens": 200, "temperature": 0.3}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    ans = out["choices"][0]["message"]["content"]
    print(f"[glm] completion ({time.time()-t:.1f}s):\n{ans}")


if __name__ == "__main__":
    main()
