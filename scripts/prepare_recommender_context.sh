#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/prepare_recommender_context.sh [SELECTION_ID]

Environment variables:
  EXPLAINERS=all                       Comma-separated explainer names or all.
  CONFIGS=all                          Comma-separated config ids or all.
  CLIENTS=all                          Comma-separated client ids or all.
  RUN_ID_FILE=job_launcher/plans/prepare_recommender_context_run_ids.txt
                                        File written for the sbatch array, one run_id per line.

Behavior:
  - Writes the configured run ids to RUN_ID_FILE.
  - Submits one Slurm array task per run id via scripts/prepare_recommender_context.sbatch.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

SELECTION_ID="${1:-${SELECTION_ID:-test__max-40__seed-42}}"
EXPLAINERS="${EXPLAINERS:-all}"
CONFIGS="${CONFIGS:-all}"
CLIENTS="${CLIENTS:-all}"
RUN_ID_FILE="${RUN_ID_FILE:-job_launcher/plans/prepare_recommender_context_run_ids.txt}"
SBATCH_SCRIPT="scripts/prepare_recommender_context.sbatch"

# RUN_IDS=(
# "federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b"
# "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
# "federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2"
# "federated-training-adult_income-20260503t135153628015+0000-logistic_regression-5clients-alpha0.1-seed42-c4ce0720be13"
# "federated-training-adult_income-20260503t135226242826+0000-logistic_regression-5clients-alpha0.3-seed42-987d584d73df"
# "federated-training-adult_income-20260503t135259866809+0000-logistic_regression-5clients-alpha1.0-seed42-2c46959e584f"
# "federated-training-adult_income-20260503t135335672402+0000-logistic_regression-5clients-alpha10.0-seed42-95917a12ac76"
# "federated-training-adult_income-20260503t135410435425+0000-logistic_regression-10clients-alpha0.1-seed42-5ab9379ae09e"
# "federated-training-adult_income-20260503t135448936944+0000-logistic_regression-10clients-alpha10.0-seed42-ff564875d681"
# "federated-training-adult_income-20260503t135521322560+0000-logistic_regression-15clients-alpha0.1-seed42-f17b987d1290"
# "federated-training-adult_income-20260503t135603131352+0000-logistic_regression-15clients-alpha1.0-seed42-a75fb8d8fee4"
# "federated-training-adult_income-20260503t135645220328+0000-logistic_regression-15clients-alpha10.0-seed42-dfa15b7a087a"
# )

# census_income runs

# RUN_IDS=(
# "federated-training-cencus_income-20260505t201144701891+0000-logreg-15clients-alpha0.3-seed42-cbf1f302be03"
# "federated-training-cencus_income-20260505t201144898551+0000-logreg-5clients-alpha10.0-seed42-11e02098db5b"
# "federated-training-cencus_income-20260505t201144927494+0000-logreg-10clients-alpha1.0-seed42-43d059ce5300"
# "federated-training-cencus_income-20260505t201144965410+0000-logreg-10clients-alpha10.0-seed42-e2a118c54dd6"
# "federated-training-cencus_income-20260505t201145083329+0000-logreg-15clients-alpha10.0-seed42-a5e99ffe6412"
# "federated-training-cencus_income-20260505t201145183850+0000-logreg-10clients-alpha0.3-seed42-8e5937a1fefe"
# "federated-training-cencus_income-20260505t201145721165+0000-logreg-10clients-alpha0.1-seed42-fa46be85d0ab"
# "federated-training-cencus_income-20260505t201145890543+0000-logreg-5clients-alpha0.3-seed42-0dbcaf7ddce5"
# "federated-training-cencus_income-20260505t201146187154+0000-logreg-5clients-alpha0.1-seed42-fbc771aca376"
# "federated-training-cencus_income-20260505t201146298557+0000-logreg-15clients-alpha0.1-seed42-3c965b4d5942"
# "federated-training-cencus_income-20260505t201146414429+0000-logreg-5clients-alpha1.0-seed42-286f05f2aa11"
# "federated-training-cencus_income-20260505t201146706924+0000-logreg-15clients-alpha1.0-seed42-12483fcc7aa8"
# )

# bank_marketing runs
RUN_IDS=(
# "federated-training-bank_marketing-20260506t100051005369+0000-logreg-15clients-alpha0.3-seed42-89223f1af61a"
# "federated-training-bank_marketing-20260506t100051025065+0000-logreg-15clients-alpha1.0-seed42-8f824f9930fc"
# "federated-training-bank_marketing-20260506t100054281100+0000-logreg-15clients-alpha10.0-seed42-d9fd3546c904"
"federated-training-bank_marketing-20260506t100054920236+0000-logreg-5clients-alpha0.1-seed42-7c35429dff4e"
"federated-training-bank_marketing-20260506t100055467077+0000-logreg-5clients-alpha0.3-seed42-9a4bff9f34a1"
"federated-training-bank_marketing-20260506t100055574191+0000-logreg-10clients-alpha10.0-seed42-9199ddd82897"
"federated-training-bank_marketing-20260506t100058395708+0000-logreg-10clients-alpha1.0-seed42-ba2e71fd4cdf"
"federated-training-bank_marketing-20260506t100059638427+0000-logreg-5clients-alpha10.0-seed42-feddbbb362d9"
"federated-training-bank_marketing-20260506t100059638440+0000-logreg-5clients-alpha1.0-seed42-6716ba670f5f"
"federated-training-bank_marketing-20260506t100059645944+0000-logreg-10clients-alpha0.3-seed42-4005ec074133"
"federated-training-bank_marketing-20260506t100059932037+0000-logreg-10clients-alpha0.1-seed42-675255e23062"
)

mkdir -p "$(dirname "$RUN_ID_FILE")"
printf '%s
' "${RUN_IDS[@]}" > "$RUN_ID_FILE"

ARRAY_SPEC="0-$((${#RUN_IDS[@]} - 1))"

echo "Submitting recommender-context preparation array"
echo "  sbatch_script=${SBATCH_SCRIPT}"
echo "  run_id_file=${RUN_ID_FILE}"
echo "  selection_id=${SELECTION_ID}"
echo "  explainers=${EXPLAINERS}"
echo "  configs=${CONFIGS}"
echo "  clients=${CLIENTS}"
echo "  array=${ARRAY_SPEC}"

sbatch   --array="${ARRAY_SPEC}"   --export=ALL,EXPLAINERS="${EXPLAINERS}",CONFIGS="${CONFIGS}",CLIENTS="${CLIENTS}"   "$SBATCH_SCRIPT"   "$RUN_ID_FILE"   "$SELECTION_ID"
