#!/bin/bash
#
# Slurm array job for running prepare-data + predictive train-federated for one launcher experiment row.
# Submit with:
#   sbatch --array=0-N scripts/train_predictive_array.sbatch EXPERIMENTS_JSONL [RESULTS_DIR]
#

#SBATCH --job-name=train-predictive
#SBATCH --output=logs/slurm-%x.%A_%a.out
#SBATCH --error=logs/slurm-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=06:00:00

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
  echo "Usage: sbatch --array=0-N scripts/train_predictive_array.sbatch EXPERIMENTS_JSONL [RESULTS_DIR]" >&2
  echo "  EXPERIMENTS_JSONL: launcher_experiments.jsonl produced by plan_launcher_experiments.sbatch" >&2
  echo "  RESULTS_DIR:       defaults to <manifest-dir>/training_results" >&2
  echo "Environment:" >&2
  echo "  PYTHON_BIN=python3   Python executable to use" >&2
  echo "  FORCE_TRAINING=0     Override manifest default and pass --force to train-federated when set to 1" >&2
  echo "  RAY_NUM_CPUS=4       Override train-federated --ray-num-cpus" >&2
  echo "  CLIENT_NUM_CPUS=1.0  Override train-federated --client-num-cpus" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

EXPERIMENTS_PATH="$1"
RESULTS_DIR="${2:-$(dirname "$EXPERIMENTS_PATH")/training_results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_TRAINING="${FORCE_TRAINING:-0}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-1.0}"

if [[ ! -f "$EXPERIMENTS_PATH" ]]; then
  echo "ERROR: experiments manifest does not exist: $EXPERIMENTS_PATH" >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set; submit with sbatch --array." >&2
  exit 2
fi
if [[ ! "$FORCE_TRAINING" =~ ^(0|1)$ ]]; then
  echo "ERROR: FORCE_TRAINING must be 0 or 1." >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
PLAN_ROWS="$(wc -l < "$EXPERIMENTS_PATH")"
if (( TASK_ID < 0 || TASK_ID >= PLAN_ROWS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range for experiment rows=${PLAN_ROWS}." >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR"
TASK_OUTPUT_JSON="${RESULTS_DIR}/train_output_$(printf '%04d' "$TASK_ID").json"
TASK_RESULT_JSON="${RESULTS_DIR}/training_result_$(printf '%04d' "$TASK_ID").json"

echo "[$(date --iso-8601=seconds)] Starting predictive launcher task"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"
echo "  task_id=${TASK_ID}"
echo "  experiments_path=${EXPERIMENTS_PATH}"
echo "  results_dir=${RESULTS_DIR}"

eval "$("$PYTHON_BIN" - "$EXPERIMENTS_PATH" "$TASK_ID" <<'PY'
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
task_id = int(sys.argv[2])
row = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[task_id])

mapping = {
    "EXPERIMENT_INDEX": row["experiment_index"],
    "DATASET_NAME": row["dataset_name"],
    "SEED": row["seed"],
    "NUM_CLIENTS": row["num_clients"],
    "ALPHA": row["alpha"],
    "MODEL_NAME": row["model_name"],
    "MODEL_LABEL": row["model_label"],
    "EPOCHS": row["model"]["epochs"],
    "BATCH_SIZE": row["model"]["batch_size"],
    "LEARNING_RATE": row["model"]["learning_rate"],
    "L2_REGULARIZATION": row["model"]["l2_regularization"],
    "ROUNDS": row["training"]["rounds"],
    "STRATEGY_NAME": row["training"]["strategy_name"],
    "SIMULATION_BACKEND": row["training"]["simulation_backend"],
    "FIT_FRACTION": row["training"]["fit_fraction"],
    "EVALUATE_FRACTION": row["training"]["evaluate_fraction"],
    "PREDICTION_THRESHOLD": row["training"]["prediction_threshold"],
    "MIN_AVAILABLE_CLIENTS": row["training"]["min_available_clients"],
    "DEBUG_FALLBACK_ON_ERROR": int(bool(row["training"]["debug_fallback_on_error"])),
    "MANIFEST_FORCE": int(bool(row["training"]["force"])),
    "SECURE_ENABLED": int(bool(row["training"]["secure_aggregation"]["enabled"])),
    "SECURE_NUM_HELPERS": row["training"]["secure_aggregation"]["num_helpers"],
    "SECURE_PRIVACY_THRESHOLD": row["training"]["secure_aggregation"]["privacy_threshold"],
    "SECURE_RECONSTRUCTION_THRESHOLD": row["training"]["secure_aggregation"]["reconstruction_threshold"] or "",
    "SECURE_FIELD_MODULUS": row["training"]["secure_aggregation"]["field_modulus"],
    "SECURE_QUANTIZATION_SCALE": row["training"]["secure_aggregation"]["quantization_scale"],
    "SECURE_SEED": row["training"]["secure_aggregation"]["seed"],
    "GLOBAL_EVAL_SIZE": row["preprocessing"]["global_eval_size"],
    "CLIENT_TEST_SIZE": row["preprocessing"]["client_test_size"],
    "MIN_CLIENT_SAMPLES": row["preprocessing"]["min_client_samples"],
    "MAX_RETRIES": row["preprocessing"]["max_retries"],
    "PLANNED_RUN_ID": row.get("planned_run_id") or "",
    "PREPARED_ROOT": row["paths"]["prepared_root"],
    "PARTITION_ROOT": row["paths"]["partition_root"],
    "CENTRALIZED_ROOT": row["paths"]["centralized_root"],
    "FEDERATED_ROOT": row["paths"]["federated_root"],
    "COMPARISON_ROOT": row["paths"]["comparison_root"],
    "CACHE_DIR": row["paths"]["cache_dir"],
}
for key, value in mapping.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

echo "  experiment_index=${EXPERIMENT_INDEX}"
echo "  dataset_name=${DATASET_NAME}"
echo "  num_clients=${NUM_CLIENTS}"
echo "  alpha=${ALPHA}"
echo "  model_name=${MODEL_NAME}"
echo "  model_label=${MODEL_LABEL}"
echo "  rounds=${ROUNDS}"
echo "  strategy=${STRATEGY_NAME}"
echo "  simulation_backend=${SIMULATION_BACKEND}"
echo "  planned_run_id=${PLANNED_RUN_ID:-<generated>}"

echo "==> Preparing partitions and training predictive federated model"
"$PYTHON_BIN" - "$EXPERIMENTS_PATH" "$TASK_ID" "$TASK_OUTPUT_JSON" "$TASK_RESULT_JSON" "$FORCE_TRAINING" "$RAY_NUM_CPUS" "$CLIENT_NUM_CPUS" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.federated_training import train_federated_from_partitions
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
    PreprocessingConfig,
)
from fed_perso_xai.utils.paths import partition_root

manifest_path = Path(sys.argv[1])
task_id = int(sys.argv[2])
task_output_path = Path(sys.argv[3])
task_result_path = Path(sys.argv[4])
force_training = bool(int(sys.argv[5]))
ray_num_cpus = int(float(sys.argv[6]))
client_num_cpus = float(sys.argv[7])

experiment = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[task_id])

paths = ArtifactPaths(
    prepared_root=Path(experiment["paths"]["prepared_root"]),
    partition_root=Path(experiment["paths"]["partition_root"]),
    centralized_root=Path(experiment["paths"]["centralized_root"]),
    federated_root=Path(experiment["paths"]["federated_root"]),
    comparison_root=Path(experiment["paths"]["comparison_root"]),
    cache_dir=Path(experiment["paths"]["cache_dir"]),
)
preprocessing = experiment["preprocessing"]
prepare_config = DataPreparationConfig(
    dataset_name=str(experiment["dataset_name"]),
    seed=int(experiment["seed"]),
    paths=paths,
    preprocessing=PreprocessingConfig(
        global_eval_size=float(preprocessing["global_eval_size"]),
        client_test_size=float(preprocessing["client_test_size"]),
        fitting_mode=str(preprocessing["fitting_mode"]),
        numeric_imputation_strategy=str(preprocessing["numeric_imputation_strategy"]),
        categorical_imputation_strategy=str(preprocessing["categorical_imputation_strategy"]),
    ),
    partition=PartitionConfig(
        num_clients=int(experiment["num_clients"]),
        alpha=float(experiment["alpha"]),
        min_client_samples=int(preprocessing["min_client_samples"]),
        max_retries=int(preprocessing["max_retries"]),
    ),
)
prepare_result = prepare_federated_dataset(prepare_config)

training = experiment["training"]
secure = training["secure_aggregation"]
training_config = FederatedTrainingConfig(
    dataset_name=str(experiment["dataset_name"]),
    seed=int(experiment["seed"]),
    model_name=str(experiment["model_name"]),
    paths=paths,
    model=LogisticRegressionConfig(
        epochs=int(experiment["model"]["epochs"]),
        batch_size=int(experiment["model"]["batch_size"]),
        learning_rate=float(experiment["model"]["learning_rate"]),
        l2_regularization=float(experiment["model"]["l2_regularization"]),
    ),
    num_clients=int(experiment["num_clients"]),
    alpha=float(experiment["alpha"]),
    strategy_name=str(training["strategy_name"]),
    rounds=int(training["rounds"]),
    fit_fraction=float(training["fit_fraction"]),
    evaluate_fraction=float(training["evaluate_fraction"]),
    prediction_threshold=float(training["prediction_threshold"]),
    min_available_clients=int(training["min_available_clients"]),
    simulation_backend=str(training["simulation_backend"]),
    debug_fallback_on_error=bool(training["debug_fallback_on_error"]),
    ray_num_cpus=ray_num_cpus,
    simulation_resources={"num_cpus": client_num_cpus, "num_gpus": 0.0},
    secure_aggregation=bool(secure["enabled"]),
    secure_num_helpers=int(secure["num_helpers"]),
    secure_privacy_threshold=int(secure["privacy_threshold"]),
    secure_reconstruction_threshold=(
        int(secure["reconstruction_threshold"])
        if secure["reconstruction_threshold"] not in ("", None)
        else None
    ),
    secure_field_modulus=int(secure["field_modulus"]),
    secure_quantization_scale=int(secure["quantization_scale"]),
    secure_seed=int(secure["seed"]),
)
partition_data_root = partition_root(
    paths.partition_root,
    str(experiment["dataset_name"]),
    int(experiment["num_clients"]),
    float(experiment["alpha"]),
    int(experiment["seed"]),
)
artifacts, training_summary = train_federated_from_partitions(
    training_config,
    run_id=(str(experiment.get("planned_run_id") or "") or None),
    partition_data_root=partition_data_root,
    force=force_training or bool(training["force"]),
)
task_output_payload = {
    "status": training_summary.get("status"),
    "run_id": training_summary.get("run_id"),
    "run_dir": str(artifacts.run_dir),
    "model_path": str(artifacts.model_artifact_path),
    "training_metadata_path": str(artifacts.training_metadata_path),
    "rounds_completed": training_summary.get("rounds_completed"),
    "training_summary": training_summary,
    "prepared_partition_root": str(prepare_result.federated_artifacts.root_dir),
}
task_output_path.write_text(json.dumps(task_output_payload, indent=2), encoding="utf-8")

payload = {
    "status": "completed",
    "experiment": experiment,
    "training_summary": training_summary,
    "run_id": training_summary.get("run_id"),
    "training_output_path": str(task_output_path),
    "run_dir": str(artifacts.run_dir),
    "model_path": str(artifacts.model_artifact_path),
    "training_metadata_path": str(artifacts.training_metadata_path),
}
task_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(task_output_payload, indent=2))
PY

echo "[$(date --iso-8601=seconds)] Predictive launcher task completed"
