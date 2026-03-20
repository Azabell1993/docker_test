from __future__ import annotations

import json
import os
import random
from pathlib import Path

random.seed(42)

ROOT = Path(os.getenv("DATASET_OUT_DIR", "/workspace/dataset/generated"))
ROOT.mkdir(parents=True, exist_ok=True)

AREAS = [
    "underwater acoustic relay network",
    "polar remote sensing gateway",
    "shipboard edge inference node",
    "autonomous surface vehicle telemetry mesh",
    "offshore platform anomaly monitoring",
    "marine sensor fusion station",
]

TASKS = [
    "link adaptation",
    "fault detection",
    "edge inference scheduling",
    "radio resource prioritization",
    "packet loss root-cause analysis",
    "mission latency control",
]

SENSORS = [
    "IMU", "sonar", "hydrophone", "AIS", "multibeam echo sounder", "temperature-depth probe"
]

CONSTRAINTS = [
    "power budget is limited",
    "backhaul is intermittent",
    "salt fog raises hardware failure risk",
    "temperature drifts alter sensor calibration",
    "bandwidth is capped below 5 Mbps",
    "mission requires 95% on-time telemetry delivery",
]

ACTIONS = [
    "switch to a smaller distilled model",
    "raise packet priority for navigation frames",
    "buffer low-value telemetry at the edge",
    "reduce generation length to control latency",
    "trigger fallback rule-based control",
    "pin inference to the local GPU and defer upload",
]


def sample_record(idx: int) -> dict:
    area = random.choice(AREAS)
    task = random.choice(TASKS)
    sensor = random.choice(SENSORS)
    constraint = random.choice(CONSTRAINTS)
    action = random.choice(ACTIONS)
    instruction = (
        f"You are assisting a marine DX Physical AI + Network engineer. "
        f"Given a {area} scenario, explain how to handle {task} when the primary sensor is {sensor} "
        f"and the operating constraint is that {constraint}. "
        f"Return a compact action plan with system reasoning, network consideration, and edge-AI consideration."
    )
    response = (
        f"1) Confirm whether the {sensor} feed and transport path are healthy. "
        f"2) Apply {task} policy at the edge node first to avoid unnecessary backhaul usage. "
        f"3) Because {constraint}, keep the SLM context window small and prefer deterministic prompts. "
        f"4) Operationally, {action}. "
        f"5) Emit structured logs for latency, drop rate, and inference confidence so the policy can be migrated to a larger DGX-class platform later."
    )
    return {
        "id": f"marine-dx-{idx:05d}",
        "domain": "marine_dx_physical_ai_network",
        "instruction": instruction,
        "input": "",
        "output": response,
        "metadata": {
            "area": area,
            "task": task,
            "sensor": sensor,
        },
    }


def write_jsonl(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps(sample_record(i), ensure_ascii=False) + "\n")


def main() -> None:
    write_jsonl(ROOT / "train.jsonl", 240)
    write_jsonl(ROOT / "val.jsonl", 40)
    write_jsonl(ROOT / "test.jsonl", 40)
    manifest = {
        "name": "marine-dx-physical-ai-network-synthetic",
        "format": "instruction jsonl",
        "splits": {"train": 240, "val": 40, "test": 40},
        "note": "Synthetic bootstrap dataset for edge SLM prototyping on Jetson and later DGX-class expansion.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
