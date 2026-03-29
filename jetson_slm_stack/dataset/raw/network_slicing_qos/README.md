# 6G Network Slicing QoS dataset prep

- Source: https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset
- Expected filename: 6G_network_slicing_qos_dataset_2345.csv
- Output files: train.jsonl, val.jsonl, test.jsonl
- Split rule: deterministic 8:1:1 by row index
- Evaluation policy: only 3GPP-aligned QoS KPI fields are retained for LLM evaluation
- Prompt policy: compressed format for Jetson Orin Nano + Llama 3.2 1B

## Allowed QoS KPI fields

- latency_ms
- packet_loss_rate_percent
- qos_metric_throughput
- network_utilization_percent
- bandwidth_utilization_percent
- signal_strength_dbm
- overload_status

## Auxiliary field

- traffic_type

## Metadata retained for traceability only

- network_slice_id
- timestamp
- device_id

## Excluded non-standard fields

- network_slice_failure
- device_type
- region
- network_failure_count
- time_of_day
- weather_conditions

## What to do

1. Place the source CSV in the raw directory.
2. Run this prep script.
3. Review manifest.prep.json, csv_schema.json, and csv_preview.json.
4. Use generated JSONL files for deterministic QoS evaluation experiments.
