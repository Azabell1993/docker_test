from __future__ import annotations

import json
import time
from pathlib import Path

import requests

PROMPTS = [
    "Summarize the network risks in an underwater acoustic relay mission with intermittent backhaul.",
    "Explain why edge-first inference is useful for marine DX physical AI on a shipboard node.",
    "Give a 5-step incident response for packet loss spikes in a polar telemetry gateway.",
]

TARGETS = {
    "llama": "http://llama32-server:8000/generate",
    "deepseek": "http://deepseek-server:8000/generate",
}

OUT_DIR = Path("/workspace/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(name: str, url: str) -> dict:
    records = []
    for prompt in PROMPTS:
        started = time.time()
        r = requests.post(url, json={"prompt": prompt, "max_new_tokens": 128, "temperature": 0.1}, timeout=300)
        r.raise_for_status()
        payload = r.json()
        records.append(
            {
                "prompt": prompt,
                "latency_sec": round(time.time() - started, 3),
                "response": payload["text"],
            }
        )
    result = {"target": name, "records": records}
    (OUT_DIR / f"benchmark_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    summary = {name: run_one(name, url) for name, url in TARGETS.items()}
    (OUT_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
