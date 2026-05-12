#!/bin/bash
#
# Slurm job for generating explain/evaluate plans and array sbatch scripts from completed predictive training results.
# Submit with:
#   sbatch scripts/plan_explain_eval_from_training.sbatch CONFIG_PATH EXPERIMENTS_JSONL [RESULTS_DIR]
#

#SBATCH --job-name=plan-exp-eval
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

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
  echo "Usage: sbatch scripts/plan_explain_eval_from_training.sbatch CONFIG_PATH EXPERIMENTS_JSONL [RESULTS_DIR]" >&2
  echo "  CONFIG_PATH:       launcher YAML such as configs/job_launcher.yml" >&2
  echo "  EXPERIMENTS_JSONL: launcher_experiments.jsonl produced by plan_launcher_experiments.sbatch" >&2
  echo "  RESULTS_DIR:       defaults to <manifest-dir>/training_results" >&2
  echo "Environment:" >&2
  echo "  SUBMIT_SLURM=0     Submit generated explain-eval sbatch scripts when set to 1" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

CONFIG_PATH="$1"
EXPERIMENTS_PATH="$2"
RESULTS_DIR="${3:-$(dirname "$EXPERIMENTS_PATH")/training_results}"
SUBMIT_SLURM="${SUBMIT_SLURM:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config path does not exist: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -f "$EXPERIMENTS_PATH" ]]; then
  echo "ERROR: experiments manifest does not exist: $EXPERIMENTS_PATH" >&2
  exit 2
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "ERROR: results directory does not exist: $RESULTS_DIR" >&2
  exit 2
fi
if [[ ! "$SUBMIT_SLURM" =~ ^(0|1)$ ]]; then
  echo "ERROR: SUBMIT_SLURM must be 0 or 1." >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] Starting post-training explain/eval planning"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"
echo "  config_path=${CONFIG_PATH}"
echo "  experiments_path=${EXPERIMENTS_PATH}"
echo "  results_dir=${RESULTS_DIR}"
echo "  submit_slurm=${SUBMIT_SLURM}"

python3 - "$CONFIG_PATH" "$EXPERIMENTS_PATH" "$RESULTS_DIR" "$SUBMIT_SLURM" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fed_perso_xai.orchestration.explain_eval import plan_explain_eval_jobs
from fed_perso_xai.orchestration.job_launcher import (
    _build_paths,
    _expand_experiments,
    _load_launcher_yaml,
    _plan_path,
    _selector_csv,
    _submit_slurm_job,
    _write_slurm_array_script,
)

config_path = Path(sys.argv[1]).resolve()
experiments_path = Path(sys.argv[2]).resolve()
results_dir = Path(sys.argv[3]).resolve()
submit_slurm = bool(int(sys.argv[4]))

raw_config = _load_launcher_yaml(config_path)
paths = _build_paths(raw_config.get("paths") or {})
explain_cfg = raw_config.get("explain_eval") or {}
slurm_cfg = explain_cfg.get("slurm") or {}
experiments = _expand_experiments(raw_config)

experiment_rows = [
    json.loads(line)
    for line in experiments_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(experiments) != len(experiment_rows):
    raise ValueError(
        f"Experiment manifest row count {len(experiment_rows)} does not match "
        f"expanded config count {len(experiments)}."
    )


def _assert_equal(label: str, expected: object, actual: object, *, index: int) -> None:
    if expected != actual:
        raise ValueError(
            f"Mismatch for experiment index {index}: {label} expected {expected!r}, got {actual!r}."
        )


def _validate_manifest_row(*, row: dict[str, object], experiment: object, index: int) -> None:
    _assert_equal("experiment_index", index, int(row["experiment_index"]), index=index)
    _assert_equal("dataset_name", experiment.dataset_name, row["dataset_name"], index=index)
    _assert_equal("seed", experiment.seed, int(row["seed"]), index=index)
    _assert_equal("num_clients", experiment.num_clients, int(row["num_clients"]), index=index)
    _assert_equal("alpha", experiment.alpha, float(row["alpha"]), index=index)
    _assert_equal("model_label", experiment.model_label, row["model_label"], index=index)
    _assert_equal("model_name", experiment.model_name, row["model_name"], index=index)


def _validate_training_result(
    *,
    row: dict[str, object],
    result_experiment: dict[str, object],
    training_summary: dict[str, object],
    index: int,
) -> None:
    _assert_equal(
        "result experiment_index",
        index,
        int(result_experiment["experiment_index"]),
        index=index,
    )
    _assert_equal(
        "result dataset_name",
        row["dataset_name"],
        result_experiment["dataset_name"],
        index=index,
    )
    _assert_equal("result seed", int(row["seed"]), int(result_experiment["seed"]), index=index)
    _assert_equal(
        "result num_clients",
        int(row["num_clients"]),
        int(result_experiment["num_clients"]),
        index=index,
    )
    _assert_equal(
        "result alpha",
        float(row["alpha"]),
        float(result_experiment["alpha"]),
        index=index,
    )
    _assert_equal(
        "result model_label",
        row["model_label"],
        result_experiment["model_label"],
        index=index,
    )
    _assert_equal(
        "result model_name",
        row["model_name"],
        result_experiment["model_name"],
        index=index,
    )

    planned_run_id = str(row.get("planned_run_id") or "").strip()
    actual_run_id = str(training_summary["run_id"]).strip()
    if planned_run_id and planned_run_id != actual_run_id:
        raise ValueError(
            f"Mismatch for experiment index {index}: planned_run_id {planned_run_id!r} "
            f"does not match training_summary.run_id {actual_run_id!r}. "
            "This usually means the launcher_experiments.jsonl or training_results directory "
            "belongs to a different dataset/config run."
        )


result_paths = sorted(results_dir.glob("training_result_*.json"))
results_by_index: dict[int, dict[str, object]] = {}
for path in result_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment = payload.get("experiment") or {}
    index = int(experiment["experiment_index"])
    results_by_index[index] = payload

missing = [
    row["experiment_index"]
    for row in experiment_rows
    if int(row["experiment_index"]) not in results_by_index
]
if missing:
    raise FileNotFoundError(
        "Missing training result files for experiment indices: "
        + ", ".join(str(index) for index in missing)
    )

runs: list[dict[str, object]] = []
for row in experiment_rows:
    index = int(row["experiment_index"])
    experiment = experiments[index]
    _validate_manifest_row(row=row, experiment=experiment, index=index)

    result = results_by_index[index]
    result_experiment = result.get("experiment") or {}
    training_summary = result["training_summary"]
    _validate_training_result(
        row=row,
        result_experiment=result_experiment,
        training_summary=training_summary,
        index=index,
    )

    run_id = str(training_summary["run_id"])
    plan_path = _plan_path(explain_cfg=explain_cfg, experiment=experiment, run_id=run_id)
    plan_summary = plan_explain_eval_jobs(
        run_id=run_id,
        output_path=plan_path,
        clients=str(explain_cfg.get("clients", "all")),
        split=str(explain_cfg.get("split", "test")),
        explainers=_selector_csv(explain_cfg.get("explainers"), default="all"),
        config_ids=_selector_csv(explain_cfg.get("configs"), default="all"),
        max_instances=int(explain_cfg.get("max_instances", 50)),
        rows_per_shard=int(explain_cfg.get("rows_per_shard", 1024)),
        random_state=int(explain_cfg.get("random_state", 42)),
        skip_existing=bool(explain_cfg.get("skip_existing", False)),
        force=bool(explain_cfg.get("force", False)),
        paths=paths,
    )
    run_record: dict[str, object] = {
        "experiment_index": index,
        "run_id": run_id,
        "plan_summary": plan_summary,
    }
    if plan_summary["job_count"] > 0 and bool(slurm_cfg.get("enabled", True)):
        script_path = _write_slurm_array_script(
            slurm_cfg=slurm_cfg,
            plan_path=plan_path,
            array_range=str(plan_summary["array_range"]),
            run_id=run_id,
            experiment=experiment,
        )
        run_record["slurm_script_path"] = str(script_path)
        if submit_slurm:
            run_record["slurm_submission"] = _submit_slurm_job(script_path, plan_path)
    runs.append(run_record)

summary = {
    "status": "planned",
    "config_path": str(config_path),
    "experiments_path": str(experiments_path),
    "results_dir": str(results_dir),
    "run_count": len(runs),
    "runs": runs,
}
summary_path = experiments_path.parent / "post_training_plans_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[$(date --iso-8601=seconds)] Post-training explain/eval planning completed"
