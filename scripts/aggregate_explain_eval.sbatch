#!/bin/bash
#
# Slurm job for aggregating all explain/eval explainer-config pairs for one run.
# Submit with:
#   sbatch scripts/aggregate_explain_eval.sbatch RUN_ID [SELECTION_ID]
#

#SBATCH --job-name=agg-exp-eval
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

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
  echo "Usage: sbatch scripts/aggregate_explain_eval.sbatch RUN_ID [SELECTION_ID]" >&2
  echo "  RUN_ID: predictive training run id whose shard outputs should be aggregated" >&2
  echo "  SELECTION_ID defaults to test__max-40__seed-42" >&2
  echo "Environment:" >&2
  echo "  PYTHON_BIN=python3      Python executable to use" >&2
  echo "  ALLOW_PARTIAL=0         Pass --allow-partial when set to 1" >&2
  echo "  EXPLAINERS=all          Optional comma-separated explainer filter" >&2
  echo "  CONFIGS=all             Optional comma-separated config-id filter" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

RUN_ID="$1"
SELECTION_ID="${2:-${SELECTION_ID:-test__max-40__seed-42}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
EXPLAINERS="${EXPLAINERS:-all}"
CONFIGS="${CONFIGS:-all}"
RUN_DIR="$PROJECT_ROOT/federated/runs/$RUN_ID"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run directory does not exist: $RUN_DIR" >&2
  exit 2
fi

if [[ ! "$ALLOW_PARTIAL" =~ ^(0|1)$ ]]; then
  echo "ERROR: ALLOW_PARTIAL must be 0 or 1." >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] Starting explain/eval aggregation"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"
echo "  run_id=${RUN_ID}"
echo "  selection_id=${SELECTION_ID}"
echo "  explainers=${EXPLAINERS}"
echo "  configs=${CONFIGS}"
echo "  allow_partial=${ALLOW_PARTIAL}"

mapfile -t PAIRS < <(
  "$PYTHON_BIN" - "$RUN_DIR" "$SELECTION_ID" "$EXPLAINERS" "$CONFIGS" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
selection_id = sys.argv[2]
explainers = sys.argv[3]
configs = sys.argv[4]

explainer_filter = None if explainers == "all" else {
    item.strip() for item in explainers.split(",") if item.strip()
}
config_filter = None if configs == "all" else {
    item.strip() for item in configs.split(",") if item.strip()
}

pairs: set[tuple[str, str]] = set()
pattern = f"client_*/selections/{selection_id}/shards/*/metrics_results/*/*.json"
for metrics_path in run_dir.joinpath("clients").glob(pattern):
    explainer_name = metrics_path.parent.name
    config_id = metrics_path.stem
    if explainer_filter is not None and explainer_name not in explainer_filter:
        continue
    if config_filter is not None and config_id not in config_filter:
        continue
    pairs.add((explainer_name, config_id))

for explainer_name, config_id in sorted(pairs):
    print(f"{explainer_name}\t{config_id}")
PY
)

if [[ "${#PAIRS[@]}" -eq 0 ]]; then
  echo "ERROR: no explainer/config pairs were discovered for run_id=${RUN_ID} selection=${SELECTION_ID}." >&2
  echo "Check that shard metrics exist under clients/*/selections/${SELECTION_ID}/shards/*/metrics_results/." >&2
  exit 2
fi

echo "  discovered_pairs=${#PAIRS[@]}"

AGGREGATE_EXTRA=()
if [[ "$ALLOW_PARTIAL" == "1" ]]; then
  AGGREGATE_EXTRA+=(--allow-partial)
fi

for pair in "${PAIRS[@]}"; do
  IFS=$'\t' read -r EXPLAINER CONFIG_ID <<< "$pair"
  echo "==> Aggregating ${EXPLAINER} / ${CONFIG_ID}"
  "$PYTHON_BIN" -m fed_perso_xai aggregate-explain-eval \
    --run-id "$RUN_ID" \
    --selection "$SELECTION_ID" \
    --explainer "$EXPLAINER" \
    --config "$CONFIG_ID" \
    "${AGGREGATE_EXTRA[@]}"
done

echo "[$(date --iso-8601=seconds)] Explain/eval aggregation completed"
