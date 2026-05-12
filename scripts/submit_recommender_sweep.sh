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
  SWEEP_LEARNING_RATES=0.001,0.005,0.01,0.02,0.05
  SWEEP_SVM_C_VALUES=0.1,0.5,1.0,2.0,5.0
  SWEEP_TRAIN_ROUNDS_VALUES=50,100,200
  SWEEP_TRAIN_EPOCHS_VALUES=1,3,5,10

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
SWEEP_LEARNING_RATES="${SWEEP_LEARNING_RATES:-0.001,0.005,0.01,0.02}"
SWEEP_SVM_C_VALUES="${SWEEP_SVM_C_VALUES:-0.1,1.0,5.0}"
SWEEP_TRAIN_ROUNDS_VALUES="${SWEEP_TRAIN_ROUNDS_VALUES:-100,200}"
SWEEP_TRAIN_EPOCHS_VALUES="${SWEEP_TRAIN_EPOCHS_VALUES:-3,5}"
SWEEP_BATCH_ID="${SWEEP_BATCH_ID:-$(date -u +%Y%m%dt%H%M%S)-$$}"

IFS=',' read -r -a LEARNING_RATES <<< "$SWEEP_LEARNING_RATES"
IFS=',' read -r -a SVM_C_VALUES <<< "$SWEEP_SVM_C_VALUES"
IFS=',' read -r -a TRAIN_ROUNDS_VALUES <<< "$SWEEP_TRAIN_ROUNDS_VALUES"
IFS=',' read -r -a TRAIN_EPOCHS_VALUES <<< "$SWEEP_TRAIN_EPOCHS_VALUES"

if [[ "${#LEARNING_RATES[@]}" -eq 0 || "${#SVM_C_VALUES[@]}" -eq 0 || "${#TRAIN_ROUNDS_VALUES[@]}" -eq 0 || "${#TRAIN_EPOCHS_VALUES[@]}" -eq 0 ]]; then
  echo "ERROR: sweep lists must not be empty." >&2
  exit 2
fi

RUN_SWEEP_ROOT="${SWEEP_RESULTS_ROOT}/${RUN_ID}/${SWEEP_BATCH_ID}"
mkdir -p "$RUN_SWEEP_ROOT"

echo "Submitting recommender sweep"
echo "  run_id=${RUN_ID}"
echo "  mode=${MODE}"
echo "  sweep_batch_id=${SWEEP_BATCH_ID}"
echo "  output_root=${RUN_SWEEP_ROOT}"
echo "  persona_assignment_policy=${PERSONA_ASSIGNMENT_POLICY}"
echo "  fixed_persona=${FIXED_PERSONA}"

for learning_rate in "${LEARNING_RATES[@]}"; do
  for svm_c in "${SVM_C_VALUES[@]}"; do
    for train_rounds in "${TRAIN_ROUNDS_VALUES[@]}"; do
      for train_epochs in "${TRAIN_EPOCHS_VALUES[@]}"; do
        trial_id="lr-$(sanitize_component "$learning_rate")__c-$(sanitize_component "$svm_c")__rounds-${train_rounds}__epochs-${train_epochs}"
        trial_dir="${RUN_SWEEP_ROOT}/${trial_id}"
        eval_output="${trial_dir}/evaluation_summary.json"
        summary_output="${trial_dir}/metrics_summary.json"
        submission_output_path="${trial_dir}/submission.txt"
        config_path="${trial_dir}/config.json"
        label_namespace="${LABEL_NAMESPACE_PREFIX}__${SWEEP_BATCH_ID}__${trial_id}"

        mkdir -p "$trial_dir"

        RUN_ID_VALUE="$RUN_ID" \
        MODE_VALUE="$MODE" \
        SWEEP_BATCH_ID_VALUE="$SWEEP_BATCH_ID" \
        SELECTION_ID_VALUE="$SELECTION_ID" \
        PERSONA_ASSIGNMENT_POLICY_VALUE="$PERSONA_ASSIGNMENT_POLICY" \
        FIXED_PERSONA_VALUE="$FIXED_PERSONA" \
        PERSONA_ASSIGNMENT_ALPHA_VALUE="$PERSONA_ASSIGNMENT_ALPHA" \
        LABEL_NAMESPACE_VALUE="$label_namespace" \
        TOP_K_VALUE="$TOP_K" \
        TRAIN_BATCH_SIZE_VALUE="$TRAIN_BATCH_SIZE" \
        TRAIN_SVM_INTERCEPT_SCALING_VALUE="$TRAIN_SVM_INTERCEPT_SCALING" \
        TRAIN_LEARNING_RATE_VALUE="$learning_rate" \
        TRAIN_SVM_C_VALUE="$svm_c" \
        TRAIN_ROUNDS_VALUE="$train_rounds" \
        TRAIN_EPOCHS_VALUE="$train_epochs" \
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
            TRAIN_LEARNING_RATE="$learning_rate" \
            TRAIN_SVM_C="$svm_c" \
            TRAIN_SVM_INTERCEPT_SCALING="$TRAIN_SVM_INTERCEPT_SCALING" \
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
      done
    done
  done
done

echo "Sweep submission complete"
echo "  output_root=${RUN_SWEEP_ROOT}"
