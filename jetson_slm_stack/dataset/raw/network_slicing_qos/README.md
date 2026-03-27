# 6G Network Slicing QoS dataset prep

- Source: https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset
- Status: prep-only
- Expected filename: 6G_network_slicing_qos_dataset_2345.csv
- Expected data type: CSV
- Core features: throughput, latency, packet loss rate
- Intended tasks: QoS prediction, slice optimization
- Intended LLM queries: latency-sensitive slice analysis, QoS anomaly detection, slice policy explanation
- Output files: train.jsonl, val.jsonl, test.jsonl

## What to place here

- Download the Kaggle dataset manually.
- Put `6G_network_slicing_qos_dataset_2345.csv` in this directory.
- If the exported filename differs, the prep script will inspect the first CSV it finds.

## Conversion rule

- One CSV row becomes one QoS analysis sample.
- The script writes `train.jsonl`, `val.jsonl`, and `test.jsonl` to the prepared directory.
- Split rule is deterministic `8:1:1` by row index.

## Planned next step

- Inspect source columns from `manifest.prep.json`.
- Start the model server.
- Run `test_dataset.sh` on the generated split files.
