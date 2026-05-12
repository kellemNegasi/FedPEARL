from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "evaluation_results"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_ROOT / "recommender_evaluation_consolidated.json"

TARGET_DATASETS = {
    "loan_default": "loan_default",
    "bank_marketing": "bank_marketing",
    "cencus_income": "census_income",
}
AGGREGATION_MODES = ("plain", "secure", "clustered")
AGGREGATION_MODE_ALIASES = {
    "plain": ("non_clustered",),
}

RUN_ID_PATTERN = re.compile(
    r"^federated-training-"
    r"(?P<dataset>.+)-"
    r"(?P<timestamp>\d{8}t\d+\+\d{4})-"
    r"(?P<model>.+)-"
    r"(?P<clients>\d+)clients-"
    r"alpha(?P<alpha>[^-]+)-"
    r"seed(?P<seed>\d+)-"
    r"(?P<suffix>[0-9a-f]+)$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate recommender evaluation summaries stored under "
            "evaluation_results/<run_id>/<mode>/evaluation_summary.json."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the consolidated output file if it already exists.",
    )
    return parser.parse_args()


def _discover_summary_paths(results_root: Path) -> list[Path]:
    paths: list[Path] = []
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if not RUN_ID_PATTERN.match(run_dir.name):
            continue
        for aggregation_mode in AGGREGATION_MODES:
            summary_path = run_dir / aggregation_mode / "evaluation_summary.json"
            if summary_path.exists():
                paths.append(summary_path)
    return paths


def _parse_run_id(run_id: str) -> dict[str, object]:
    match = RUN_ID_PATTERN.match(run_id)
    if not match:
        raise ValueError(f"Unexpected run_id format: {run_id}")
    groups = match.groupdict()
    dataset_key = groups["dataset"]
    alpha_text = groups["alpha"]
    return {
        "dataset_key": dataset_key,
        "dataset_name": TARGET_DATASETS.get(dataset_key, dataset_key),
        "model_name": groups["model"],
        "num_clients": int(groups["clients"]),
        "alpha": float(alpha_text),
        "alpha_text": alpha_text,
        "seed": int(groups["seed"]),
    }


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _client_metrics(summary: dict[str, object]) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    for client in summary.get("clients", []):
        client_id = str(client["client_id"])
        metrics[client_id] = {
            "spearman_at_3": client.get("spearman_at_3"),
            "spearman_at_5": client.get("spearman_at_5"),
            "spearman_at_8": client.get("spearman_at_8"),
        }
    return dict(sorted(metrics.items()))


def _aggregate_metrics(summary: dict[str, object]) -> dict[str, float | None]:
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return {}
    return {
        str(key): float(value) if isinstance(value, (int, float)) else None
        for key, value in aggregate.items()
    }


def build_payload(results_root: Path, output_path: Path) -> dict[str, object]:
    dataset_entries: dict[str, dict[str, dict[str, object]]] = {
        canonical_name: {} for canonical_name in sorted(set(TARGET_DATASETS.values()))
    }

    summary_paths = _discover_summary_paths(results_root)
    for path in summary_paths:
        summary = _load_summary(path)
        run_id = str(summary["run_id"])
        parsed = _parse_run_id(run_id)
        dataset_name = str(parsed["dataset_name"])
        federated_config = f"{parsed['num_clients']}clients-alpha{parsed['alpha_text']}"
        config_key = f"{run_id}::{federated_config}"
        aggregation_mode = str(summary.get("aggregation_mode") or path.parent.name)

        config_entry = dataset_entries[dataset_name].setdefault(
            config_key,
            {
                "runid": run_id,
                "federated_config": federated_config,
                "num_clients": parsed["num_clients"],
                "alpha": parsed["alpha"],
                "alpha_text": parsed["alpha_text"],
                "seed": parsed["seed"],
                "model_name": parsed["model_name"],
                "selection_id": summary.get("selection_id"),
                "persona": summary.get("persona"),
                "recommender_type": summary.get("recommender_type"),
                "training_variant": summary.get("training_variant"),
                "aggregations": {},
            },
        )

        aggregation_payload = {
            "client_count": summary.get("client_count"),
            "source_path": str(path.relative_to(PROJECT_ROOT)),
            "aggregate": _aggregate_metrics(summary),
            "clients": _client_metrics(summary),
        }
        config_entry["aggregations"][aggregation_mode] = aggregation_payload
        for alias in AGGREGATION_MODE_ALIASES.get(aggregation_mode, ()):
            config_entry["aggregations"][alias] = aggregation_payload

    finalized_datasets: dict[str, list[dict[str, object]]] = {}
    for dataset_name, configs in dataset_entries.items():
        entries = list(configs.values())
        entries.sort(key=lambda item: (item["num_clients"], item["alpha"], item["runid"]))
        finalized_datasets[dataset_name] = entries

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "dataset_count": len(finalized_datasets),
        "config_count": sum(len(entries) for entries in finalized_datasets.values()),
        "aggregation_modes": list(AGGREGATION_MODES),
        "datasets": finalized_datasets,
    }


def main() -> None:
    args = _parse_args()
    results_root = args.results_root.resolve()
    output_path = args.output.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing consolidated output without --overwrite: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(results_root=results_root, output_path=output_path)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {payload['config_count']} configs to {output_path}")


if __name__ == "__main__":
    main()
