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
# RUN_IDS=(
# "federated-training-bank_marketing-20260506t100051005369+0000-logreg-15clients-alpha0.3-seed42-89223f1af61a"
# "federated-training-bank_marketing-20260506t100051025065+0000-logreg-15clients-alpha1.0-seed42-8f824f9930fc"
# "federated-training-bank_marketing-20260506t100054281100+0000-logreg-15clients-alpha10.0-seed42-d9fd3546c904"
# "federated-training-bank_marketing-20260506t100054920236+0000-logreg-5clients-alpha0.1-seed42-7c35429dff4e"
# "federated-training-bank_marketing-20260506t100055467077+0000-logreg-5clients-alpha0.3-seed42-9a4bff9f34a1"
# "federated-training-bank_marketing-20260506t100055574191+0000-logreg-10clients-alpha10.0-seed42-9199ddd82897"
# "federated-training-bank_marketing-20260506t100058395708+0000-logreg-10clients-alpha1.0-seed42-ba2e71fd4cdf"
# "federated-training-bank_marketing-20260506t100059638427+0000-logreg-5clients-alpha10.0-seed42-feddbbb362d9"
# "federated-training-bank_marketing-20260506t100059638440+0000-logreg-5clients-alpha1.0-seed42-6716ba670f5f"
# "federated-training-bank_marketing-20260506t100059645944+0000-logreg-10clients-alpha0.3-seed42-4005ec074133"
# "federated-training-bank_marketing-20260506t100059932037+0000-logreg-10clients-alpha0.1-seed42-675255e23062"
# )

# RUN_IDS=(
# "federated-training-loan_default-20260506t150506940283+0000-logreg-5clients-alpha10.0-seed42-4b46b580b199"
# "federated-training-loan_default-20260506t150513688715+0000-logreg-10clients-alpha0.3-seed42-ffc078cdbd9a"
# "federated-training-loan_default-20260506t150514510628+0000-logreg-5clients-alpha0.3-seed42-af8a57f89bda"
# "federated-training-loan_default-20260506t150515534664+0000-logreg-10clients-alpha10.0-seed42-eadd6c717a89"
# "federated-training-loan_default-20260506t150515535354+0000-logreg-15clients-alpha0.3-seed42-c30b4a3c12ed"
# "federated-training-loan_default-20260506t150516029127+0000-logreg-5clients-alpha1.0-seed42-b6d74e400440"
# "federated-training-loan_default-20260506t150516264799+0000-logreg-15clients-alpha10.0-seed42-99087037bae1"
# "federated-training-loan_default-20260506t150516265176+0000-logreg-15clients-alpha1.0-seed42-0a79319724d3"
# "federated-training-loan_default-20260506t150516449846+0000-logreg-15clients-alpha0.1-seed42-9c95533169ea"
# "federated-training-loan_default-20260506t150518932069+0000-logreg-5clients-alpha0.1-seed42-4e3b9873a6df"
# "federated-training-loan_default-20260506t150526008995+0000-logreg-10clients-alpha1.0-seed42-71744e771867"
# "federated-training-loan_default-20260506t150534943473+0000-logreg-10clients-alpha0.1-seed42-b01a52d5dd06"
# )

RUN_IDS=(
"federated-training-bank_marketing-20260507t012319019498+0000-mlp_classifier-5clients-alpha0.1-seed42-5cec84b0765b"
"federated-training-bank_marketing-20260507t012357803690+0000-mlp_classifier-5clients-alpha0.3-seed42-aac13ebeacb3"
"federated-training-bank_marketing-20260507t012658335610+0000-mlp_classifier-5clients-alpha1.0-seed42-0843eb7808bc"
"federated-training-bank_marketing-20260507t012738686581+0000-mlp_classifier-5clients-alpha10.0-seed42-f0c260353f98"
"federated-training-bank_marketing-20260507t013036907867+0000-mlp_classifier-10clients-alpha0.1-seed42-0e8a21d8880e"
"federated-training-bank_marketing-20260507t013116545176+0000-mlp_classifier-10clients-alpha0.3-seed42-e17ac799f9b7"
"federated-training-bank_marketing-20260507t013438611246+0000-mlp_classifier-10clients-alpha1.0-seed42-f6ac25b91ef5"
"federated-training-bank_marketing-20260507t013517989821+0000-mlp_classifier-10clients-alpha10.0-seed42-e4cf5b24a1c2"
"federated-training-cencus_income-20260506t222902686786+0000-mlp_classifier-5clients-alpha0.1-seed42-249ff01e8075"
"federated-training-cencus_income-20260506t222902686725+0000-mlp_classifier-5clients-alpha0.3-seed42-ef898b9a73ab"
"federated-training-cencus_income-20260506t224346667160+0000-mlp_classifier-5clients-alpha1.0-seed42-9fa18cd5e96f"
"federated-training-cencus_income-20260506t224346667151+0000-mlp_classifier-5clients-alpha10.0-seed42-eb09212a1574"
"federated-training-cencus_income-20260506t225837614057+0000-mlp_classifier-10clients-alpha0.1-seed42-9db23665e6c0"
"federated-training-cencus_income-20260506t225837612250+0000-mlp_classifier-10clients-alpha0.3-seed42-980852e07e3d"
"federated-training-cencus_income-20260506t231552481693+0000-mlp_classifier-10clients-alpha1.0-seed42-9712ebbf1103"
"federated-training-cencus_income-20260506t231552480561+0000-mlp_classifier-10clients-alpha10.0-seed42-718fb140b9eb"
"federated-training-loan_default-20260507t001300230109+0000-mlp_classifier-5clients-alpha0.1-seed42-e39ea8d4c1b5"
"federated-training-loan_default-20260507t001312352686+0000-mlp_classifier-5clients-alpha0.3-seed42-7c63ce563937"
"federated-training-loan_default-20260507t002402058908+0000-mlp_classifier-5clients-alpha1.0-seed42-60e995ccc770"
"federated-training-loan_default-20260507t002402058902+0000-mlp_classifier-5clients-alpha10.0-seed42-d17114a46236"
"federated-training-loan_default-20260507t003505441475+0000-mlp_classifier-10clients-alpha0.1-seed42-2876b20d020d"
"federated-training-loan_default-20260507t003505440029+0000-mlp_classifier-10clients-alpha0.3-seed42-a7fe32381792"
"federated-training-loan_default-20260507t004647650199+0000-mlp_classifier-10clients-alpha1.0-seed42-daf9d965fb9b"
"federated-training-loan_default-20260507t004650006944+0000-mlp_classifier-10clients-alpha10.0-seed42-5604c6cbb710"
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
