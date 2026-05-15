#!/usr/bin/env python3
"""Extract compact recommender sweep metrics from an evaluation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract aggregate Spearman@3/5/8 and per-client Spearman values "
            "from a recommender evaluation summary JSON."
        )
    )
    parser.add_argument("--input", required=True, help="Path to evaluation summary JSON.")
    parser.add_argument("--output", required=True, help="Path to compact metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    clients = payload.get("clients", [])

    summary = {
        "status": payload.get("status"),
        "run_id": payload.get("run_id"),
        "selection_id": payload.get("selection_id"),
        "persona": payload.get("persona"),
        "aggregation_mode": payload.get("aggregation_mode"),
        "training_variant": payload.get("training_variant"),
        "recommender_type": payload.get("recommender_type"),
        "generated_at": payload.get("generated_at"),
        "evaluation_summary_path": str(input_path),
        "aggregate": {
            "spearman_at_3": aggregate.get("spearman_at_3"),
            "spearman_at_5": aggregate.get("spearman_at_5"),
            "spearman_at_8": aggregate.get("spearman_at_8"),
        },
        "clients": {
            client.get("client_id"): {
                "spearman": client.get("spearman"),
            }
            for client in clients
            if client.get("client_id") is not None
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
