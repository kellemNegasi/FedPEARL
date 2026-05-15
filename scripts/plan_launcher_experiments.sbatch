#!/bin/bash
#
# Slurm job for expanding a launcher config into predictive-training experiments.
# Submit with:
#   sbatch scripts/plan_launcher_experiments.sbatch CONFIG_PATH [OUTPUT_DIR]
#

#SBATCH --job-name=plan-launcher
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/helios/home/kelemnegasi/fed-xai/fed-perso-xai}"
cd "$PROJECT_ROOT"

module load $(module -t avail python 2>&1 | grep '^python/3\.14' | head -n 1)
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Check module/venv setup." >&2
  exit 1
fi

usage() {
  echo "Usage: sbatch scripts/plan_launcher_experiments.sbatch CONFIG_PATH [OUTPUT_DIR]" >&2
  echo "  CONFIG_PATH: launcher YAML such as configs/job_launcher.yml" >&2
  echo "  OUTPUT_DIR:  optional manifest directory. Defaults to job_launcher/plans/<config-stem>__launcher__<timestamp>" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config path does not exist: $CONFIG_PATH" >&2
  exit 2
fi

if [[ $# -ge 2 ]]; then
  OUTPUT_DIR="$2"
else
  CONFIG_STEM="$(basename "${CONFIG_PATH%.*}")"
  OUTPUT_DIR="job_launcher/plans/${CONFIG_STEM}__launcher__$(date +%Y%m%dT%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"

echo "[$(date --iso-8601=seconds)] Starting launcher planning"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"
echo "  config_path=${CONFIG_PATH}"
echo "  output_dir=${OUTPUT_DIR}"

python3 - "$CONFIG_PATH" "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import json
from pathlib import Path

from fed_perso_xai.orchestration.job_launcher import (
    _build_paths,
    _expand_experiments,
    _load_launcher_yaml,
    _render_run_id,
    _validate_unsupported_list_fields,
)

config_path = Path(__import__("sys").argv[1]).resolve()
output_dir = Path(__import__("sys").argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

raw_config = _load_launcher_yaml(config_path)
_validate_unsupported_list_fields(raw_config)
paths = _build_paths(raw_config.get("paths") or {})
experiments = _expand_experiments(raw_config)
pre_cfg = raw_config.get("preprocessing") or {}
part_cfg = raw_config.get("partition") or {}
train_cfg = raw_config.get("training") or {}
explain_cfg = raw_config.get("explain_eval") or {}
secure_cfg = train_cfg.get("secure_aggregation") or {}
selection_id = (
    f"{str(explain_cfg.get('split', 'test'))}"
    f"__max-{int(explain_cfg.get('max_instances', 50))}"
    f"__seed-{int(explain_cfg.get('random_state', 42))}"
)

experiments_path = output_dir / "launcher_experiments.jsonl"
partitions_path = output_dir / "launcher_partitions.jsonl"
summary_path = output_dir / "planning_summary.json"

partition_rows: dict[tuple[str, int, int, float], dict[str, object]] = {}
with experiments_path.open("w", encoding="utf-8") as handle:
    for experiment_index, experiment in enumerate(experiments):
        record = {
            "experiment_index": experiment_index,
            "dataset_name": experiment.dataset_name,
            "seed": experiment.seed,
            "num_clients": experiment.num_clients,
            "alpha": experiment.alpha,
            "model_label": experiment.model_label,
            "model_name": experiment.model_name,
            "model": {
                "epochs": experiment.model_config.epochs,
                "batch_size": experiment.model_config.batch_size,
                "learning_rate": experiment.model_config.learning_rate,
                "l2_regularization": experiment.model_config.l2_regularization,
                "hidden_dim": getattr(experiment.model_config, "hidden_dim", None),
                "activation": getattr(experiment.model_config, "activation", None),
                "optimizer": getattr(experiment.model_config, "optimizer", None),
                "device": getattr(experiment.model_config, "device", None),
            },
            "training": {
                "rounds": experiment.rounds,
                "strategy_name": experiment.strategy_name,
                "simulation_backend": experiment.simulation_backend,
                "fit_fraction": float(train_cfg.get("fit_fraction", 1.0)),
                "evaluate_fraction": float(train_cfg.get("evaluate_fraction", 1.0)),
                "prediction_threshold": float(train_cfg.get("prediction_threshold", 0.5)),
                "min_available_clients": int(train_cfg.get("min_available_clients", 2)),
                "debug_fallback_on_error": bool(train_cfg.get("debug_fallback_on_error", False)),
                "force": bool(train_cfg.get("force", False)),
                "secure_aggregation": {
                    "enabled": bool(secure_cfg) if isinstance(secure_cfg, dict) else bool(secure_cfg),
                    "num_helpers": int(secure_cfg.get("num_helpers", 5)) if isinstance(secure_cfg, dict) else 5,
                    "privacy_threshold": int(secure_cfg.get("privacy_threshold", 2)) if isinstance(secure_cfg, dict) else 2,
                    "reconstruction_threshold": secure_cfg.get("reconstruction_threshold") if isinstance(secure_cfg, dict) else None,
                    "field_modulus": int(secure_cfg.get("field_modulus", 2_147_483_647)) if isinstance(secure_cfg, dict) else 2_147_483_647,
                    "quantization_scale": int(secure_cfg.get("quantization_scale", 1 << 16)) if isinstance(secure_cfg, dict) else (1 << 16),
                    "seed": int(secure_cfg.get("seed", 0)) if isinstance(secure_cfg, dict) else 0,
                },
            },
            "preprocessing": {
                "global_eval_size": float(pre_cfg.get("global_eval_size", 0.2)),
                "client_test_size": float(pre_cfg.get("client_test_size", 0.2)),
                "fitting_mode": str(pre_cfg.get("fitting_mode", "global_shared")),
                "numeric_imputation_strategy": str(
                    pre_cfg.get("numeric_imputation_strategy", "median")
                ),
                "categorical_imputation_strategy": str(
                    pre_cfg.get("categorical_imputation_strategy", "most_frequent")
                ),
                "min_client_samples": int(part_cfg.get("min_client_samples", 10)),
                "max_retries": int(part_cfg.get("max_retries", 50)),
            },
            "paths": {
                "prepared_root": str(paths.prepared_root),
                "partition_root": str(paths.partition_root),
                "centralized_root": str(paths.centralized_root),
                "federated_root": str(paths.federated_root),
                "comparison_root": str(paths.comparison_root),
                "cache_dir": str(paths.cache_dir),
            },
            "explain_eval": {
                "enabled": bool(explain_cfg.get("enabled", True)),
                "clients": str(explain_cfg.get("clients", "all")),
                "split": str(explain_cfg.get("split", "test")),
                "explainers": str(explain_cfg.get("explainers", "all")),
                "configs": str(explain_cfg.get("configs", "all")),
                "max_instances": int(explain_cfg.get("max_instances", 50)),
                "rows_per_shard": int(explain_cfg.get("rows_per_shard", 1024)),
                "random_state": int(explain_cfg.get("random_state", 42)),
                "skip_existing": bool(explain_cfg.get("skip_existing", False)),
                "force": bool(explain_cfg.get("force", False)),
                "selection_id": selection_id,
            },
            "planned_run_id": _render_run_id(raw_config.get("run_id_template"), experiment),
        }
        handle.write(json.dumps(record) + "\n")

        partition_key = (
            experiment.dataset_name,
            experiment.seed,
            experiment.num_clients,
            experiment.alpha,
        )
        if partition_key not in partition_rows:
            partition_rows[partition_key] = {
                "dataset_name": experiment.dataset_name,
                "seed": experiment.seed,
                "num_clients": experiment.num_clients,
                "alpha": experiment.alpha,
                "preprocessing": record["preprocessing"],
                "paths": record["paths"],
            }

with partitions_path.open("w", encoding="utf-8") as handle:
    for record in partition_rows.values():
        handle.write(json.dumps(record) + "\n")

summary = {
    "status": "planned",
    "config_path": str(config_path),
    "output_dir": str(output_dir),
    "experiment_count": len(experiments),
    "partition_count": len(partition_rows),
    "array_range": f"0-{len(experiments) - 1}" if experiments else None,
    "experiments_path": str(experiments_path),
    "partitions_path": str(partitions_path),
    "selection_id": selection_id,
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[$(date --iso-8601=seconds)] Launcher planning completed"
