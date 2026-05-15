#!/bin/bash
#SBATCH --job-name=recommender-eval-logreg
#SBATCH --output=slurm-recommender-eval-%j.out
#SBATCH --error=slurm-recommender-eval-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_ROOT"

RESULTS_ROOT="$PROJECT_ROOT/evaluation_results"
RUNS_ROOT="$PROJECT_ROOT/federated/runs"
SELECTION_ID="test__max-40__seed-42"
PERSONA="dirichlet_sampled"
RECOMMENDER_DIR="svm_rank_fedavg"
TOP_K="1,3,5,8"
FORCE="${FORCE:-0}"

mkdir -p "$RESULTS_ROOT"

RUN_IDS=(
  "federated-training-bank_marketing-20260506t100054920236+0000-logreg-5clients-alpha0.1-seed42-7c35429dff4e"
  "federated-training-bank_marketing-20260506t100055467077+0000-logreg-5clients-alpha0.3-seed42-9a4bff9f34a1"
  "federated-training-bank_marketing-20260506t100055574191+0000-logreg-10clients-alpha10.0-seed42-9199ddd82897"
  "federated-training-bank_marketing-20260506t100058395708+0000-logreg-10clients-alpha1.0-seed42-ba2e71fd4cdf"
  "federated-training-bank_marketing-20260506t100059638427+0000-logreg-5clients-alpha10.0-seed42-feddbbb362d9"
  "federated-training-bank_marketing-20260506t100059638440+0000-logreg-5clients-alpha1.0-seed42-6716ba670f5f"
  "federated-training-bank_marketing-20260506t100059645944+0000-logreg-10clients-alpha0.3-seed42-4005ec074133"
  "federated-training-bank_marketing-20260506t100059932037+0000-logreg-10clients-alpha0.1-seed42-675255e23062"
  "federated-training-cencus_income-20260505t201144701891+0000-logreg-15clients-alpha0.3-seed42-cbf1f302be03"
  "federated-training-cencus_income-20260505t201144898551+0000-logreg-5clients-alpha10.0-seed42-11e02098db5b"
  "federated-training-cencus_income-20260505t201144927494+0000-logreg-10clients-alpha1.0-seed42-43d059ce5300"
  "federated-training-cencus_income-20260505t201144965410+0000-logreg-10clients-alpha10.0-seed42-e2a118c54dd6"
  "federated-training-cencus_income-20260505t201145083329+0000-logreg-15clients-alpha10.0-seed42-a5e99ffe6412"
  "federated-training-cencus_income-20260505t201145183850+0000-logreg-10clients-alpha0.3-seed42-8e5937a1fefe"
  "federated-training-cencus_income-20260505t201145721165+0000-logreg-10clients-alpha0.1-seed42-fa46be85d0ab"
  "federated-training-cencus_income-20260505t201145890543+0000-logreg-5clients-alpha0.3-seed42-0dbcaf7ddce5"
  "federated-training-cencus_income-20260505t201146187154+0000-logreg-5clients-alpha0.1-seed42-fbc771aca376"
  "federated-training-cencus_income-20260505t201146298557+0000-logreg-15clients-alpha0.1-seed42-3c965b4d5942"
  "federated-training-cencus_income-20260505t201146414429+0000-logreg-5clients-alpha1.0-seed42-286f05f2aa11"
  "federated-training-cencus_income-20260505t201146706924+0000-logreg-15clients-alpha1.0-seed42-12483fcc7aa8"
  "federated-training-loan_default-20260506t150506940283+0000-logreg-5clients-alpha10.0-seed42-4b46b580b199"
  "federated-training-loan_default-20260506t150513688715+0000-logreg-10clients-alpha0.3-seed42-ffc078cdbd9a"
  "federated-training-loan_default-20260506t150514510628+0000-logreg-5clients-alpha0.3-seed42-af8a57f89bda"
  "federated-training-loan_default-20260506t150515534664+0000-logreg-10clients-alpha10.0-seed42-eadd6c717a89"
  "federated-training-loan_default-20260506t150515535354+0000-logreg-15clients-alpha0.3-seed42-c30b4a3c12ed"
  "federated-training-loan_default-20260506t150516029127+0000-logreg-5clients-alpha1.0-seed42-b6d74e400440"
  "federated-training-loan_default-20260506t150516264799+0000-logreg-15clients-alpha10.0-seed42-99087037bae1"
  "federated-training-loan_default-20260506t150516265176+0000-logreg-15clients-alpha1.0-seed42-0a79319724d3"
  "federated-training-loan_default-20260506t150516449846+0000-logreg-15clients-alpha0.1-seed42-9c95533169ea"
  "federated-training-loan_default-20260506t150518932069+0000-logreg-5clients-alpha0.1-seed42-4e3b9873a6df"
  "federated-training-loan_default-20260506t150526008995+0000-logreg-10clients-alpha1.0-seed42-71744e771867"
  "federated-training-loan_default-20260506t150534943473+0000-logreg-10clients-alpha0.1-seed42-b01a52d5dd06"
)

echo "Project root: $PROJECT_ROOT"
echo "Results root: $RESULTS_ROOT"
echo "Run count: ${#RUN_IDS[@]}"

for run_id in "${RUN_IDS[@]}"; do
  echo "============================================================"
  echo "Evaluating run_id=$run_id"
  for mode in plain secure clustered; do
    train_dir="$RUNS_ROOT/$run_id/recommender_training/$SELECTION_ID/$PERSONA/$RECOMMENDER_DIR/$mode"
    output_dir="$RESULTS_ROOT/$run_id/$mode"
    output_path="$output_dir/evaluation_summary.json"

    if [[ ! -d "$train_dir" ]]; then
      echo "Skipping mode=$mode for run_id=$run_id because $train_dir does not exist"
      continue
    fi

    mkdir -p "$output_dir"

    if [[ -f "$output_path" && "$FORCE" != "1" ]]; then
      echo "Skipping mode=$mode for run_id=$run_id because output already exists: $output_path"
      continue
    fi

    extra_args=()
    case "$mode" in
      plain)
        extra_args+=(--plain-aggregation)
        ;;
      secure)
        extra_args+=(--secure-aggregation)
        ;;
      clustered)
        extra_args+=(--clustered)
        ;;
    esac

    echo "Running mode=$mode -> $output_path"
    uv run python -m fed_perso_xai evaluate-recommender \
      --run-id "$run_id" \
      --selection "$SELECTION_ID" \
      --persona "$PERSONA" \
      --clients all \
      --recommender svm_rank \
      --top-k "$TOP_K" \
      --output "$output_path" \
      "${extra_args[@]}"
  done
done

echo "============================================================"
echo "Consolidating evaluation results"
consolidate_args=(
  --results-root "$RESULTS_ROOT"
  --output "$RESULTS_ROOT/recommender_evaluation_consolidated.json"
)
if [[ "$FORCE" == "1" ]]; then
  consolidate_args+=(--overwrite)
elif [[ -f "$RESULTS_ROOT/recommender_evaluation_consolidated.json" ]]; then
  echo "Skipping consolidation because output already exists and FORCE is not 1:"
  echo "  $RESULTS_ROOT/recommender_evaluation_consolidated.json"
  exit 0
fi

uv run python scripts/consolidate_recommender_evaluation_results.py \
  "${consolidate_args[@]}"

echo "Finished. Consolidated output:"
echo "  $RESULTS_ROOT/recommender_evaluation_consolidated.json"
