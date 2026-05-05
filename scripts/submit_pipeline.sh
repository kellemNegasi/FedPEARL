#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_pipeline.sh [clustered] [CLUSTERING_K] [CLUSTERING_WARMUP_ROUNDS] [CLUSTERING_FREEZE_PCA_AFTER_WARMUP] [TOP_K] [CLUSTERING_ENABLE_PCA]

Behavior:
  Default: submit both plain and secure runs for every RUN_ID.
  clustered: submit one clustered run for every RUN_ID.

Environment variables:
  LABEL_NAMESPACE=
                                    Shared label namespace used by label/train/eval.
                                    Defaults to FIXED_PERSONA for fixed policy or dirichlet_sampled otherwise.
  PERSONA=
                                    Deprecated alias for LABEL_NAMESPACE.
  FIXED_PERSONA=lay
                                    Bundled persona config used only when PERSONA_ASSIGNMENT_POLICY=fixed.
  PERSONA_ASSIGNMENT_POLICY=dirichlet_sampled
  PERSONA_ASSIGNMENT_ALPHA=
  INSTANCE_TEST_SIZE=0.2
  INSTANCE_VALIDATION_SIZE=0.1
  TRAIN_ROUNDS=200
  TRAIN_EPOCHS=10
  TRAIN_BATCH_SIZE=2048
  TRAIN_LEARNING_RATE=0.01
  TRAIN_SVM_C=0.1
  TRAIN_SVM_INTERCEPT_SCALING=1.0
  SKIP_LABELING=0
  CLUSTERING_K=2
  CLUSTERING_REPRESENTATION=delta
  CLUSTERING_NORMALIZE_VECTOR=1
  CLUSTERING_NORMALIZATION_MODE=l2
  CLUSTERING_DELTA_OVER_BASE_NORM=1
  CLUSTERING_ASSIGNMENT_MARGIN=0.05
  CLUSTERING_NUM_RESTARTS=5
  CLUSTERING_WARMUP_ROUNDS=15
  CLUSTERING_FREEZE_PCA_AFTER_WARMUP=1
  CLUSTERING_ENABLE_PCA=0
  TOP_K=1,3,5,8
USAGE
}

# Previous RUN_IDS kept temporarily for reference:
RUN_IDS=(
  "federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b"
  "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
  "federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2"
  "federated-training-adult_income-20260503t135153628015+0000-logistic_regression-5clients-alpha0.1-seed42-c4ce0720be13"
  "federated-training-adult_income-20260503t135226242826+0000-logistic_regression-5clients-alpha0.3-seed42-987d584d73df"
  "federated-training-adult_income-20260503t135259866809+0000-logistic_regression-5clients-alpha1.0-seed42-2c46959e584f"
  "federated-training-adult_income-20260503t135335672402+0000-logistic_regression-5clients-alpha10.0-seed42-95917a12ac76"
  "federated-training-adult_income-20260503t135410435425+0000-logistic_regression-10clients-alpha0.1-seed42-5ab9379ae09e"
  "federated-training-adult_income-20260503t135448936944+0000-logistic_regression-10clients-alpha10.0-seed42-ff564875d681"
  "federated-training-adult_income-20260503t135521322560+0000-logistic_regression-15clients-alpha0.1-seed42-f17b987d1290"
  "federated-training-adult_income-20260503t135603131352+0000-logistic_regression-15clients-alpha1.0-seed42-a75fb8d8fee4"
  "federated-training-adult_income-20260503t135645220328+0000-logistic_regression-15clients-alpha10.0-seed42-dfa15b7a087a"
)

MODE_ARG="${1:-}"
CLUSTERING_K_ARG="${2:-}"
CLUSTERING_WARMUP_ROUNDS_ARG="${3:-}"
CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG="${4:-}"
TOP_K_ARG="${5:-}"
CLUSTERING_ENABLE_PCA_ARG="${6:-}"

case "${MODE_ARG,,}" in
  "")
    SUBMISSION_MODES=(
      # "plain"
      "secure"
    )
    ;;
  clustered|--clustered)
    SUBMISSION_MODES=("clustered")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unsupported mode '$MODE_ARG'." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -n "$CLUSTERING_K_ARG" && ! "$CLUSTERING_K_ARG" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLUSTERING_K must be a positive integer." >&2
  exit 2
fi
if [[ -n "$CLUSTERING_WARMUP_ROUNDS_ARG" && ! "$CLUSTERING_WARMUP_ROUNDS_ARG" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLUSTERING_WARMUP_ROUNDS must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG" && ! "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_FREEZE_PCA_AFTER_WARMUP must be 0 or 1." >&2
  exit 2
fi
if [[ -n "$CLUSTERING_ENABLE_PCA_ARG" && ! "$CLUSTERING_ENABLE_PCA_ARG" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_ENABLE_PCA must be 0 or 1." >&2
  exit 2
fi

SELECTION_ID="${SELECTION_ID:-test__max-40__seed-42}"
PERSONA_ASSIGNMENT_POLICY="${PERSONA_ASSIGNMENT_POLICY:-dirichlet_sampled}"
FIXED_PERSONA="${FIXED_PERSONA:-lay}"
LABEL_NAMESPACE_ENV="${LABEL_NAMESPACE:-}"
LEGACY_PERSONA_NAMESPACE="${PERSONA:-}"
if [[ -n "$LABEL_NAMESPACE_ENV" ]]; then
  LABEL_NAMESPACE="$LABEL_NAMESPACE_ENV"
elif [[ -n "$LEGACY_PERSONA_NAMESPACE" ]]; then
  LABEL_NAMESPACE="$LEGACY_PERSONA_NAMESPACE"
elif [[ "$PERSONA_ASSIGNMENT_POLICY" == "fixed" ]]; then
  LABEL_NAMESPACE="$FIXED_PERSONA"
else
  LABEL_NAMESPACE="dirichlet_sampled"
fi
PERSONA_ASSIGNMENT_ALPHA="${PERSONA_ASSIGNMENT_ALPHA:-10.0}"
INSTANCE_TEST_SIZE="${INSTANCE_TEST_SIZE:-0.2}"
INSTANCE_VALIDATION_SIZE="${INSTANCE_VALIDATION_SIZE:-0.1}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-200}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.02}"
TRAIN_SVM_C="${TRAIN_SVM_C:-0.5}"
TRAIN_SVM_INTERCEPT_SCALING="${TRAIN_SVM_INTERCEPT_SCALING:-1.0}"
SKIP_LABELING="${SKIP_LABELING:-0}"
CLUSTERING_K="${CLUSTERING_K_ARG:-${CLUSTERING_K:-3}}"
CLUSTERING_REPRESENTATION="${CLUSTERING_REPRESENTATION:-delta}"
CLUSTERING_NORMALIZE_VECTOR="${CLUSTERING_NORMALIZE_VECTOR:-1}"
CLUSTERING_NORMALIZATION_MODE="${CLUSTERING_NORMALIZATION_MODE:-l2}"
CLUSTERING_DELTA_OVER_BASE_NORM="${CLUSTERING_DELTA_OVER_BASE_NORM:-1}"
CLUSTERING_ASSIGNMENT_MARGIN="${CLUSTERING_ASSIGNMENT_MARGIN:-0.05}"
CLUSTERING_NUM_RESTARTS="${CLUSTERING_NUM_RESTARTS:-5}"
CLUSTERING_WARMUP_ROUNDS="${CLUSTERING_WARMUP_ROUNDS_ARG:-${CLUSTERING_WARMUP_ROUNDS:-15}}"
CLUSTERING_FREEZE_PCA_AFTER_WARMUP="${CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG:-${CLUSTERING_FREEZE_PCA_AFTER_WARMUP:-1}}"
TOP_K="${TOP_K_ARG:-${TOP_K:-1,3,5,8}}"
CLUSTERING_ENABLE_PCA="${CLUSTERING_ENABLE_PCA_ARG:-${CLUSTERING_ENABLE_PCA:-0}}"
if [[ ! "$CLUSTERING_K" =~ ^[0-9]+$ ]] || [[ "$CLUSTERING_K" -lt 1 ]]; then
  echo "ERROR: CLUSTERING_K must be a positive integer." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_REPRESENTATION" =~ ^(model|delta)$ ]]; then
  echo "ERROR: CLUSTERING_REPRESENTATION must be model or delta." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_NORMALIZE_VECTOR" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_NORMALIZE_VECTOR must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_NORMALIZATION_MODE" =~ ^(l2)$ ]]; then
  echo "ERROR: CLUSTERING_NORMALIZATION_MODE must be l2." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_DELTA_OVER_BASE_NORM" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_DELTA_OVER_BASE_NORM must be 0 or 1." >&2
  exit 2
fi
if ! [[ "$CLUSTERING_ASSIGNMENT_MARGIN" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ "$(awk "BEGIN {print ($CLUSTERING_ASSIGNMENT_MARGIN >= 0 && $CLUSTERING_ASSIGNMENT_MARGIN < 1)}")" != "1" ]]; then
  echo "ERROR: CLUSTERING_ASSIGNMENT_MARGIN must be a number in [0, 1)." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_NUM_RESTARTS" =~ ^[0-9]+$ ]] || [[ "$CLUSTERING_NUM_RESTARTS" -lt 1 ]]; then
  echo "ERROR: CLUSTERING_NUM_RESTARTS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_WARMUP_ROUNDS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLUSTERING_WARMUP_ROUNDS must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_FREEZE_PCA_AFTER_WARMUP must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_ENABLE_PCA" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_ENABLE_PCA must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$SKIP_LABELING" =~ ^(0|1)$ ]]; then
  echo "ERROR: SKIP_LABELING must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$PERSONA_ASSIGNMENT_POLICY" =~ ^(fixed|dirichlet_sampled)$ ]]; then
  echo "ERROR: PERSONA_ASSIGNMENT_POLICY must be fixed or dirichlet_sampled." >&2
  exit 2
fi
if [[ -n "$PERSONA_ASSIGNMENT_ALPHA" ]] && { ! [[ "$PERSONA_ASSIGNMENT_ALPHA" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ "$(awk "BEGIN {print ($PERSONA_ASSIGNMENT_ALPHA > 0)}")" != "1" ]]; }; then
  echo "ERROR: PERSONA_ASSIGNMENT_ALPHA must be a positive number when provided." >&2
  exit 2
fi
if ! [[ "$TRAIN_LEARNING_RATE" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ "$(awk "BEGIN {print ($TRAIN_LEARNING_RATE > 0)}")" != "1" ]]; then
  echo "ERROR: TRAIN_LEARNING_RATE must be a positive number." >&2
  exit 2
fi
if ! [[ "$TRAIN_SVM_C" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ "$(awk "BEGIN {print ($TRAIN_SVM_C > 0)}")" != "1" ]]; then
  echo "ERROR: TRAIN_SVM_C must be a positive number." >&2
  exit 2
fi
if ! [[ "$TRAIN_SVM_INTERCEPT_SCALING" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ "$(awk "BEGIN {print ($TRAIN_SVM_INTERCEPT_SCALING > 0)}")" != "1" ]]; then
  echo "ERROR: TRAIN_SVM_INTERCEPT_SCALING must be a positive number." >&2
  exit 2
fi
if [[ -z "$TOP_K" ]]; then
  echo "ERROR: TOP_K must be a non-empty comma-separated list." >&2
  exit 2
fi
SBATCH_SCRIPT="scripts/recommender_pipeline.sbatch"

for run_id in "${RUN_IDS[@]}"; do
  for submission_mode in "${SUBMISSION_MODES[@]}"; do
    FIXED_PERSONA="$FIXED_PERSONA" \
    CLUSTERING_REPRESENTATION="$CLUSTERING_REPRESENTATION" \
    CLUSTERING_NORMALIZE_VECTOR="$CLUSTERING_NORMALIZE_VECTOR" \
    CLUSTERING_NORMALIZATION_MODE="$CLUSTERING_NORMALIZATION_MODE" \
    CLUSTERING_DELTA_OVER_BASE_NORM="$CLUSTERING_DELTA_OVER_BASE_NORM" \
    CLUSTERING_ASSIGNMENT_MARGIN="$CLUSTERING_ASSIGNMENT_MARGIN" \
    CLUSTERING_NUM_RESTARTS="$CLUSTERING_NUM_RESTARTS" \
    INSTANCE_TEST_SIZE="$INSTANCE_TEST_SIZE" \
    INSTANCE_VALIDATION_SIZE="$INSTANCE_VALIDATION_SIZE" \
    sbatch \
      "$SBATCH_SCRIPT" \
      "$run_id" \
      "$submission_mode" \
      "$SELECTION_ID" \
      "$LABEL_NAMESPACE" \
      "$TRAIN_ROUNDS" \
      "$TRAIN_EPOCHS" \
      "$TRAIN_BATCH_SIZE" \
      "$TRAIN_LEARNING_RATE" \
      "$TRAIN_SVM_C" \
      "$TRAIN_SVM_INTERCEPT_SCALING" \
      "$SKIP_LABELING" \
      "$CLUSTERING_K" \
      "$CLUSTERING_WARMUP_ROUNDS" \
      "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP" \
      "$TOP_K" \
      "$CLUSTERING_ENABLE_PCA" \
      "$PERSONA_ASSIGNMENT_POLICY" \
      "$PERSONA_ASSIGNMENT_ALPHA"
  done
done
