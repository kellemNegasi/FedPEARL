#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_recommender_sweep.sh RUN_ID [MODE]

Arguments:
  RUN_ID        Predictive training run id to sweep.
  MODE          plain, secure, or clustered. Defaults to plain.

Environment variables:
  SWEEP_KIND=recommender              recommender or clustering.
  SWEEP_RESULTS_ROOT=sweep_results/recommender
  SWEEP_BATCH_ID=auto-generated-unique-id
  SELECTION_ID=test__max-40__seed-42
  PERSONA_ASSIGNMENT_POLICY=fixed
  FIXED_PERSONA=lay
  PERSONA_ASSIGNMENT_ALPHA=
  LABEL_NAMESPACE_PREFIX=<FIXED_PERSONA>
  SKIP_LABELING=0
  TOP_K=1,3,5,8
  TRAIN_BATCH_SIZE=2048
  TRAIN_SVM_INTERCEPT_SCALING=1.0
  TRAIN_LEARNING_RATE=0.005           Fixed default for clustering sweeps.
  TRAIN_SVM_C=15.0                    Fixed default for clustering sweeps.
  TRAIN_ROUNDS=150                    Fixed default for clustering sweeps.
  TRAIN_EPOCHS=3                      Fixed default for clustering sweeps.

  Recommender sweep:
    SWEEP_LEARNING_RATES=0.001,0.005,0.01,0.02,0.05
    SWEEP_SVM_C_VALUES=0.1,0.5,1.0,2.0,5.0
    SWEEP_TRAIN_ROUNDS_VALUES=50,100,200
    SWEEP_TRAIN_EPOCHS_VALUES=1,3,5,10

  Clustering sweep:
    MODE should be clustered.
    SWEEP_CLUSTERING_K_VALUES=2,3,4
    SWEEP_CLUSTERING_WARMUP_ROUNDS_VALUES=5,15,30
    SWEEP_CLUSTERING_FREEZE_PCA_VALUES=0,1
    SWEEP_CLUSTERING_ENABLE_PCA_VALUES=0
    SWEEP_CLUSTERING_REPRESENTATION_VALUES=delta,model
    SWEEP_CLUSTERING_ASSIGNMENT_MARGIN_VALUES=0.0,0.05,0.1
    SWEEP_CLUSTERING_NUM_RESTARTS_VALUES=5,10

Outputs:
  For each trial, writes:
    sweep_results/recommender/<RUN_ID>/<SWEEP_BATCH_ID>/<TRIAL_ID>/config.json
    sweep_results/recommender/<RUN_ID>/<SWEEP_BATCH_ID>/<TRIAL_ID>/evaluation_summary.json
    sweep_results/recommender/<RUN_ID>/<SWEEP_BATCH_ID>/<TRIAL_ID>/metrics_summary.json
    sweep_results/recommender/<RUN_ID>/<SWEEP_BATCH_ID>/<TRIAL_ID>/submission.txt
USAGE
}

sanitize_component() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//,/__}"
  echo "$value"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

RUN_ID="$1"
MODE="${2:-plain}"
case "${MODE,,}" in
  plain|secure|clustered)
    ;;
  *)
    echo "ERROR: MODE must be one of plain, secure, or clustered." >&2
    exit 2
    ;;
esac

if [[ -n "${SWEEP_KIND:-}" ]]; then
  RESOLVED_SWEEP_KIND="$SWEEP_KIND"
elif [[ "${MODE,,}" == "clustered" ]]; then
  RESOLVED_SWEEP_KIND="clustering"
else
  RESOLVED_SWEEP_KIND="recommender"
fi
SWEEP_KIND="$RESOLVED_SWEEP_KIND"
if [[ ! "$SWEEP_KIND" =~ ^(recommender|clustering)$ ]]; then
  echo "ERROR: SWEEP_KIND must be recommender or clustering." >&2
  exit 2
fi

SWEEP_RESULTS_ROOT="${SWEEP_RESULTS_ROOT:-sweep_results/recommender}"
SELECTION_ID="${SELECTION_ID:-test__max-40__seed-42}"
PERSONA_ASSIGNMENT_POLICY="${PERSONA_ASSIGNMENT_POLICY:-fixed}"
FIXED_PERSONA="${FIXED_PERSONA:-lay}"
PERSONA_ASSIGNMENT_ALPHA="${PERSONA_ASSIGNMENT_ALPHA:-}"
LABEL_NAMESPACE_PREFIX="${LABEL_NAMESPACE_PREFIX:-$FIXED_PERSONA}"
SKIP_LABELING="${SKIP_LABELING:-0}"
TOP_K="${TOP_K:-1,3,5,8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
TRAIN_SVM_INTERCEPT_SCALING="${TRAIN_SVM_INTERCEPT_SCALING:-1.0}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.005}"
TRAIN_SVM_C="${TRAIN_SVM_C:-15.0}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-150}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
SWEEP_LEARNING_RATES="${SWEEP_LEARNING_RATES:-0.005}"
SWEEP_SVM_C_VALUES="${SWEEP_SVM_C_VALUES:-15.0}"
SWEEP_TRAIN_ROUNDS_VALUES="${SWEEP_TRAIN_ROUNDS_VALUES:-150}"
SWEEP_TRAIN_EPOCHS_VALUES="${SWEEP_TRAIN_EPOCHS_VALUES:-3}"
SWEEP_CLUSTERING_K_VALUES="${SWEEP_CLUSTERING_K_VALUES:-2,3,4}"
SWEEP_CLUSTERING_WARMUP_ROUNDS_VALUES="${SWEEP_CLUSTERING_WARMUP_ROUNDS_VALUES:-5,15,30}"
SWEEP_CLUSTERING_FREEZE_PCA_VALUES="${SWEEP_CLUSTERING_FREEZE_PCA_VALUES:-0}"
SWEEP_CLUSTERING_ENABLE_PCA_VALUES="${SWEEP_CLUSTERING_ENABLE_PCA_VALUES:-0}"
SWEEP_CLUSTERING_REPRESENTATION_VALUES="${SWEEP_CLUSTERING_REPRESENTATION_VALUES:-delta,model}"
SWEEP_CLUSTERING_ASSIGNMENT_MARGIN_VALUES="${SWEEP_CLUSTERING_ASSIGNMENT_MARGIN_VALUES:-0.0,0.02,0.05,0.1}"
SWEEP_CLUSTERING_NUM_RESTARTS_VALUES="${SWEEP_CLUSTERING_NUM_RESTARTS_VALUES:-5,10}"
SWEEP_BATCH_ID="${SWEEP_BATCH_ID:-$(date -u +%Y%m%dt%H%M%S)-$$}"

if [[ "$SWEEP_KIND" == "clustering" && "${MODE,,}" != "clustered" ]]; then
  echo "ERROR: clustering sweeps must use MODE=clustered." >&2
  exit 2
fi

RUN_SWEEP_ROOT="${SWEEP_RESULTS_ROOT}/${RUN_ID}/${SWEEP_BATCH_ID}"
mkdir -p "$RUN_SWEEP_ROOT"

echo "Submitting recommender sweep"
echo "  run_id=${RUN_ID}"
echo "  mode=${MODE}"
echo "  sweep_kind=${SWEEP_KIND}"
echo "  sweep_batch_id=${SWEEP_BATCH_ID}"
echo "  output_root=${RUN_SWEEP_ROOT}"
echo "  persona_assignment_policy=${PERSONA_ASSIGNMENT_POLICY}"
echo "  fixed_persona=${FIXED_PERSONA}"
echo "  fixed_train_learning_rate=${TRAIN_LEARNING_RATE}"
echo "  fixed_train_svm_c=${TRAIN_SVM_C}"
echo "  fixed_train_rounds=${TRAIN_ROUNDS}"
echo "  fixed_train_epochs=${TRAIN_EPOCHS}"

submit_trial() {
  local trial_id="$1"
  local train_learning_rate="$2"
  local train_svm_c="$3"
  local train_rounds="$4"
  local train_epochs="$5"
  local clustering_k="$6"
  local clustering_warmup_rounds="$7"
  local clustering_freeze_pca="$8"
  local clustering_enable_pca="$9"
  local clustering_representation="${10}"
  local clustering_assignment_margin="${11}"
  local clustering_num_restarts="${12}"

  local trial_dir="${RUN_SWEEP_ROOT}/${trial_id}"
  local eval_output="${trial_dir}/evaluation_summary.json"
  local summary_output="${trial_dir}/metrics_summary.json"
  local submission_output_path="${trial_dir}/submission.txt"
  local config_path="${trial_dir}/config.json"
  local label_namespace="${LABEL_NAMESPACE_PREFIX}__${SWEEP_BATCH_ID}__${trial_id}"

  mkdir -p "$trial_dir"

  RUN_ID_VALUE="$RUN_ID" \
  MODE_VALUE="$MODE" \
  SWEEP_KIND_VALUE="$SWEEP_KIND" \
  SWEEP_BATCH_ID_VALUE="$SWEEP_BATCH_ID" \
  SELECTION_ID_VALUE="$SELECTION_ID" \
  PERSONA_ASSIGNMENT_POLICY_VALUE="$PERSONA_ASSIGNMENT_POLICY" \
  FIXED_PERSONA_VALUE="$FIXED_PERSONA" \
  PERSONA_ASSIGNMENT_ALPHA_VALUE="$PERSONA_ASSIGNMENT_ALPHA" \
  LABEL_NAMESPACE_VALUE="$label_namespace" \
  TOP_K_VALUE="$TOP_K" \
  TRAIN_BATCH_SIZE_VALUE="$TRAIN_BATCH_SIZE" \
  TRAIN_SVM_INTERCEPT_SCALING_VALUE="$TRAIN_SVM_INTERCEPT_SCALING" \
  TRAIN_LEARNING_RATE_VALUE="$train_learning_rate" \
  TRAIN_SVM_C_VALUE="$train_svm_c" \
  TRAIN_ROUNDS_VALUE="$train_rounds" \
  TRAIN_EPOCHS_VALUE="$train_epochs" \
  CLUSTERING_K_VALUE="$clustering_k" \
  CLUSTERING_WARMUP_ROUNDS_VALUE="$clustering_warmup_rounds" \
  CLUSTERING_FREEZE_PCA_VALUE="$clustering_freeze_pca" \
  CLUSTERING_ENABLE_PCA_VALUE="$clustering_enable_pca" \
  CLUSTERING_REPRESENTATION_VALUE="$clustering_representation" \
  CLUSTERING_ASSIGNMENT_MARGIN_VALUE="$clustering_assignment_margin" \
  CLUSTERING_NUM_RESTARTS_VALUE="$clustering_num_restarts" \
  SKIP_LABELING_VALUE="$SKIP_LABELING" \
  TRIAL_ID_VALUE="$trial_id" \
  TRIAL_DIR_VALUE="$trial_dir" \
  EVAL_OUTPUT_VALUE="$eval_output" \
  SUMMARY_OUTPUT_VALUE="$summary_output" \
  CONFIG_PATH_VALUE="$config_path" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID_VALUE"],
    "mode": os.environ["MODE_VALUE"],
    "sweep_kind": os.environ["SWEEP_KIND_VALUE"],
    "sweep_batch_id": os.environ["SWEEP_BATCH_ID_VALUE"],
    "trial_id": os.environ["TRIAL_ID_VALUE"],
    "selection_id": os.environ["SELECTION_ID_VALUE"],
    "persona_assignment_policy": os.environ["PERSONA_ASSIGNMENT_POLICY_VALUE"],
    "fixed_persona": os.environ["FIXED_PERSONA_VALUE"],
    "persona_assignment_alpha": os.environ["PERSONA_ASSIGNMENT_ALPHA_VALUE"] or None,
    "label_namespace": os.environ["LABEL_NAMESPACE_VALUE"],
    "top_k": [int(value) for value in os.environ["TOP_K_VALUE"].split(",") if value],
    "hyperparameters": {
        "train_rounds": int(os.environ["TRAIN_ROUNDS_VALUE"]),
        "train_epochs": int(os.environ["TRAIN_EPOCHS_VALUE"]),
        "train_batch_size": int(os.environ["TRAIN_BATCH_SIZE_VALUE"]),
        "train_learning_rate": float(os.environ["TRAIN_LEARNING_RATE_VALUE"]),
        "train_svm_c": float(os.environ["TRAIN_SVM_C_VALUE"]),
        "train_svm_intercept_scaling": float(os.environ["TRAIN_SVM_INTERCEPT_SCALING_VALUE"]),
    },
    "clustering": {
        "clustering_k": int(os.environ["CLUSTERING_K_VALUE"]),
        "clustering_warmup_rounds": int(os.environ["CLUSTERING_WARMUP_ROUNDS_VALUE"]),
        "clustering_freeze_pca_after_warmup": int(os.environ["CLUSTERING_FREEZE_PCA_VALUE"]),
        "clustering_enable_pca": int(os.environ["CLUSTERING_ENABLE_PCA_VALUE"]),
        "clustering_representation": os.environ["CLUSTERING_REPRESENTATION_VALUE"],
        "clustering_assignment_margin": float(os.environ["CLUSTERING_ASSIGNMENT_MARGIN_VALUE"]),
        "clustering_num_restarts": int(os.environ["CLUSTERING_NUM_RESTARTS_VALUE"]),
    },
    "skip_labeling": int(os.environ["SKIP_LABELING_VALUE"]),
    "paths": {
        "trial_dir": os.environ["TRIAL_DIR_VALUE"],
        "evaluation_summary": os.environ["EVAL_OUTPUT_VALUE"],
        "metrics_summary": os.environ["SUMMARY_OUTPUT_VALUE"],
    },
}

config_path = Path(os.environ["CONFIG_PATH_VALUE"])
config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  echo "  submitting ${trial_id}"
  submission_output="$(
    env \
      RUN_IDS_CSV="$RUN_ID" \
      SELECTION_ID="$SELECTION_ID" \
      PERSONA_ASSIGNMENT_POLICY="$PERSONA_ASSIGNMENT_POLICY" \
      FIXED_PERSONA="$FIXED_PERSONA" \
      PERSONA_ASSIGNMENT_ALPHA="$PERSONA_ASSIGNMENT_ALPHA" \
      LABEL_NAMESPACE="$label_namespace" \
      TRAIN_ROUNDS="$train_rounds" \
      TRAIN_EPOCHS="$train_epochs" \
      TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
      TRAIN_LEARNING_RATE="$train_learning_rate" \
      TRAIN_SVM_C="$train_svm_c" \
      TRAIN_SVM_INTERCEPT_SCALING="$TRAIN_SVM_INTERCEPT_SCALING" \
      CLUSTERING_K="$clustering_k" \
      CLUSTERING_WARMUP_ROUNDS="$clustering_warmup_rounds" \
      CLUSTERING_FREEZE_PCA_AFTER_WARMUP="$clustering_freeze_pca" \
      CLUSTERING_ENABLE_PCA="$clustering_enable_pca" \
      CLUSTERING_REPRESENTATION="$clustering_representation" \
      CLUSTERING_ASSIGNMENT_MARGIN="$clustering_assignment_margin" \
      CLUSTERING_NUM_RESTARTS="$clustering_num_restarts" \
      SKIP_LABELING="$SKIP_LABELING" \
      TOP_K="$TOP_K" \
      EVAL_OUTPUT="$eval_output" \
      SWEEP_SUMMARY_OUTPUT="$summary_output" \
      scripts/submit_pipeline.sh "$MODE"
  )"

  printf '%s\n' "$submission_output" > "$submission_output_path"

  job_id="$(printf '%s\n' "$submission_output" | awk '/Submitted batch job/ {print $4; exit}')"
  if [[ -n "$job_id" ]]; then
    printf '%s\n' "$job_id" > "${trial_dir}/job_id.txt"
    echo "    job_id=${job_id}"
  else
    echo "    warning: could not parse job id from submission output" >&2
  fi
}

if [[ "$SWEEP_KIND" == "recommender" ]]; then
  IFS=',' read -r -a LEARNING_RATES <<< "$SWEEP_LEARNING_RATES"
  IFS=',' read -r -a SVM_C_VALUES <<< "$SWEEP_SVM_C_VALUES"
  IFS=',' read -r -a TRAIN_ROUNDS_VALUES <<< "$SWEEP_TRAIN_ROUNDS_VALUES"
  IFS=',' read -r -a TRAIN_EPOCHS_VALUES <<< "$SWEEP_TRAIN_EPOCHS_VALUES"

  if [[ "${#LEARNING_RATES[@]}" -eq 0 || "${#SVM_C_VALUES[@]}" -eq 0 || "${#TRAIN_ROUNDS_VALUES[@]}" -eq 0 || "${#TRAIN_EPOCHS_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: recommender sweep lists must not be empty." >&2
    exit 2
  fi

  for learning_rate in "${LEARNING_RATES[@]}"; do
    for svm_c in "${SVM_C_VALUES[@]}"; do
      for train_rounds in "${TRAIN_ROUNDS_VALUES[@]}"; do
        for train_epochs in "${TRAIN_EPOCHS_VALUES[@]}"; do
          trial_id="lr-$(sanitize_component "$learning_rate")__c-$(sanitize_component "$svm_c")__rounds-${train_rounds}__epochs-${train_epochs}"
          submit_trial \
            "$trial_id" \
            "$learning_rate" \
            "$svm_c" \
            "$train_rounds" \
            "$train_epochs" \
            "2" \
            "15" \
            "1" \
            "0" \
            "delta" \
            "0.05" \
            "10"
        done
      done
    done
  done
else
  IFS=',' read -r -a CLUSTERING_K_VALUES <<< "$SWEEP_CLUSTERING_K_VALUES"
  IFS=',' read -r -a CLUSTERING_WARMUP_ROUNDS_VALUES <<< "$SWEEP_CLUSTERING_WARMUP_ROUNDS_VALUES"
  IFS=',' read -r -a CLUSTERING_FREEZE_PCA_VALUES <<< "$SWEEP_CLUSTERING_FREEZE_PCA_VALUES"
  IFS=',' read -r -a CLUSTERING_ENABLE_PCA_VALUES <<< "$SWEEP_CLUSTERING_ENABLE_PCA_VALUES"
  IFS=',' read -r -a CLUSTERING_REPRESENTATION_VALUES <<< "$SWEEP_CLUSTERING_REPRESENTATION_VALUES"
  IFS=',' read -r -a CLUSTERING_ASSIGNMENT_MARGIN_VALUES <<< "$SWEEP_CLUSTERING_ASSIGNMENT_MARGIN_VALUES"
  IFS=',' read -r -a CLUSTERING_NUM_RESTARTS_VALUES <<< "$SWEEP_CLUSTERING_NUM_RESTARTS_VALUES"

  if [[ "${#CLUSTERING_K_VALUES[@]}" -eq 0 || "${#CLUSTERING_WARMUP_ROUNDS_VALUES[@]}" -eq 0 || "${#CLUSTERING_FREEZE_PCA_VALUES[@]}" -eq 0 || "${#CLUSTERING_ENABLE_PCA_VALUES[@]}" -eq 0 || "${#CLUSTERING_REPRESENTATION_VALUES[@]}" -eq 0 || "${#CLUSTERING_ASSIGNMENT_MARGIN_VALUES[@]}" -eq 0 || "${#CLUSTERING_NUM_RESTARTS_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: clustering sweep lists must not be empty." >&2
    exit 2
  fi

  for clustering_k in "${CLUSTERING_K_VALUES[@]}"; do
    for clustering_warmup_rounds in "${CLUSTERING_WARMUP_ROUNDS_VALUES[@]}"; do
      for clustering_freeze_pca in "${CLUSTERING_FREEZE_PCA_VALUES[@]}"; do
        for clustering_enable_pca in "${CLUSTERING_ENABLE_PCA_VALUES[@]}"; do
          for clustering_representation in "${CLUSTERING_REPRESENTATION_VALUES[@]}"; do
            for clustering_assignment_margin in "${CLUSTERING_ASSIGNMENT_MARGIN_VALUES[@]}"; do
              for clustering_num_restarts in "${CLUSTERING_NUM_RESTARTS_VALUES[@]}"; do
                trial_id="k-${clustering_k}__warmup-${clustering_warmup_rounds}__freeze-${clustering_freeze_pca}__pca-${clustering_enable_pca}__repr-$(sanitize_component "$clustering_representation")__margin-$(sanitize_component "$clustering_assignment_margin")__restarts-${clustering_num_restarts}"
                submit_trial \
                  "$trial_id" \
                  "$TRAIN_LEARNING_RATE" \
                  "$TRAIN_SVM_C" \
                  "$TRAIN_ROUNDS" \
                  "$TRAIN_EPOCHS" \
                  "$clustering_k" \
                  "$clustering_warmup_rounds" \
                  "$clustering_freeze_pca" \
                  "$clustering_enable_pca" \
                  "$clustering_representation" \
                  "$clustering_assignment_margin" \
                  "$clustering_num_restarts"
              done
            done
          done
        done
      done
    done
  done
fi

echo "Sweep submission complete"
echo "  output_root=${RUN_SWEEP_ROOT}"
