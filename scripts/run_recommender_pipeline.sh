#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_recommender_pipeline.sh RUN_ID SELECTION_ID [LABEL_NAMESPACE]

Environment variables:
  PYTHON=python                         Python executable to use.
  CLIENTS=all                           Comma-separated client ids or all.
  CONTEXT_FILENAME=candidate_context.parquet
                                        Context file to label/train on.
  LABEL_FILENAME=pairwise_labels.parquet
                                        Label artifact name expected by training/eval.
  SIMULATOR=dirichlet_persona           Labeling simulator name.
  LABEL_SEED=1729                       RNG seed for simulated pairwise labels.
  PERSONA_SEED=42                       RNG seed for persona assignment / metric sampling.
  INSTANCE_TEST_SIZE=0.2               Held-out recommender test split size over selected instances.
  INSTANCE_VALIDATION_SIZE=0.1         Validation split size over the remaining post-test instances.
  PERSONA_ASSIGNMENT_POLICY=dirichlet_sampled
                                        Labeling persona assignment policy.
  PERSONA_ASSIGNMENT_ALPHA=             Optional Dirichlet concentration for client-level persona assignment.
  FIXED_PERSONA=lay                     Bundled persona config used only when PERSONA_ASSIGNMENT_POLICY=fixed.
  LABEL_NAMESPACE=                      Shared label namespace used by label/train/eval.
                                        Defaults to FIXED_PERSONA for fixed policy or dirichlet_sampled otherwise.
  PERSONA=                              Deprecated alias for LABEL_NAMESPACE.
  TRAIN_ROUNDS=10                       Federated recommender rounds.
  TRAIN_EPOCHS=5                        Local recommender epochs.
  TRAIN_BATCH_SIZE=64                   Local recommender batch size.
  TRAIN_LEARNING_RATE=0.05              Local recommender learning rate.
  TRAIN_L2_REGULARIZATION=0.0           Local recommender L2 regularization.
  TRAIN_SVM_C=0.5                       LinearSVC-style SVM C parameter.
  TRAIN_SVM_INTERCEPT_SCALING=1.0       LinearSVC-style intercept scaling.
  TRAIN_SEED=42                         Federated recommender training seed.
  RECOMMENDER_TYPE=svm_rank             Recommender backend: svm_rank or pairwise_logistic.
  FIT_FRACTION=1.0                      Fraction of clients sampled for fit.
  EVALUATE_FRACTION=1.0                 Fraction of clients sampled for eval.
  MIN_AVAILABLE_CLIENTS=2               Minimum clients for fit/eval.
  SIMULATION_BACKEND=auto               auto, ray, debug-sequential, sequential_fallback.
  RAY_NUM_CPUS=4                        Ray runtime CPU cap passed to Flower init_args.
  CLIENT_NUM_CPUS=1                     CPU reservation per Flower client actor.
  DEBUG_FALLBACK_ON_ERROR=0             Pass --debug-fallback-on-error to training.
  SKIP_LABELING=0                       Skip label-recommender-context when set to 1.
  SECURE_AGGREGATION=0                  Pass --secure-aggregation to training when set to 1.
  SECURE_NUM_HELPERS=5                  Secure aggregation helper count.
  SECURE_PRIVACY_THRESHOLD=2            Secure aggregation privacy threshold.
  SECURE_RECONSTRUCTION_THRESHOLD=      Optional secure aggregation reconstruction threshold.
  SECURE_FIELD_MODULUS=2147483647       Secure aggregation field modulus.
  SECURE_QUANTIZATION_SCALE=65536       Secure aggregation quantization scale.
  SECURE_SEED=0                         Secure aggregation RNG seed.
  CLUSTERED=0                           Pass --clustered to training when set to 1.
  CLUSTERING_METHOD=secure_kmeans       Clustered training method.
  CLUSTERING_REPRESENTATION=model       Cluster either full local models or per-round deltas: model or delta.
  CLUSTERING_NORMALIZE_VECTOR=1         Pass --no-clustering-normalize-vector when set to 0.
  CLUSTERING_NORMALIZATION_MODE=l2      Client-side clustering vector normalization mode.
  CLUSTERING_DELTA_OVER_BASE_NORM=1     When clustering deltas, normalize by the starting model norm.
  CLUSTERING_K=3                        Number of recommender clusters when clustering is enabled.
  CLUSTERING_NUM_RESTARTS=5             Number of K-means restarts per clustered round.
  CLUSTERING_ENABLE_PCA=1              Pass --no-clustering-enable-pca when set to 0.
  CLUSTERING_PCA_COMPONENTS=8           PCA components for clustered training.
  CLUSTERING_WARMUP_ROUNDS=0            Initial global-only rounds before clustering starts.
  CLUSTERING_FREEZE_PCA_AFTER_WARMUP=0  Pass --clustering-freeze-pca-after-warmup when set to 1.
  TOP_K=1,3,5                           Comma-separated precision@k cutoffs.
  FORCE_TRAINING=0                      Pass --force to train-recommender-federated.
  EVAL_OUTPUT=                          Optional path for evaluate-recommender JSON output.

Pipeline:
  1. label-recommender-context
  2. train-recommender-federated
  3. evaluate-recommender
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

RUN_ID="$1"
SELECTION_ID="$2"
LABEL_NAMESPACE_ARG="${3:-}"
PERSONA_ASSIGNMENT_POLICY="${PERSONA_ASSIGNMENT_POLICY:-dirichlet_sampled}"
FIXED_PERSONA="${FIXED_PERSONA:-lay}"
LABEL_NAMESPACE_ENV="${LABEL_NAMESPACE:-}"
LEGACY_PERSONA_NAMESPACE="${PERSONA:-}"

if [[ -n "$LABEL_NAMESPACE_ARG" ]]; then
  LABEL_NAMESPACE="$LABEL_NAMESPACE_ARG"
elif [[ -n "$LABEL_NAMESPACE_ENV" ]]; then
  LABEL_NAMESPACE="$LABEL_NAMESPACE_ENV"
elif [[ -n "$LEGACY_PERSONA_NAMESPACE" ]]; then
  LABEL_NAMESPACE="$LEGACY_PERSONA_NAMESPACE"
elif [[ "$PERSONA_ASSIGNMENT_POLICY" == "fixed" ]]; then
  LABEL_NAMESPACE="$FIXED_PERSONA"
else
  LABEL_NAMESPACE="dirichlet_sampled"
fi

CLIENTS="${CLIENTS:-all}"
CONTEXT_FILENAME="${CONTEXT_FILENAME:-candidate_context.parquet}"
LABEL_FILENAME="${LABEL_FILENAME:-pairwise_labels.parquet}"
SIMULATOR="${SIMULATOR:-dirichlet_persona}"
LABEL_SEED="${LABEL_SEED:-1729}"
PERSONA_SEED="${PERSONA_SEED:-42}"
INSTANCE_TEST_SIZE="${INSTANCE_TEST_SIZE:-0.2}"
INSTANCE_VALIDATION_SIZE="${INSTANCE_VALIDATION_SIZE:-0.1}"
PERSONA_ASSIGNMENT_ALPHA="${PERSONA_ASSIGNMENT_ALPHA:-}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.05}"
TRAIN_L2_REGULARIZATION="${TRAIN_L2_REGULARIZATION:-0.0}"
TRAIN_SVM_C="${TRAIN_SVM_C:-0.5}"
TRAIN_SVM_INTERCEPT_SCALING="${TRAIN_SVM_INTERCEPT_SCALING:-1.0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
RECOMMENDER_TYPE="${RECOMMENDER_TYPE:-svm_rank}"
FIT_FRACTION="${FIT_FRACTION:-1.0}"
EVALUATE_FRACTION="${EVALUATE_FRACTION:-1.0}"
MIN_AVAILABLE_CLIENTS="${MIN_AVAILABLE_CLIENTS:-2}"
SIMULATION_BACKEND="${SIMULATION_BACKEND:-auto}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-1}"
DEBUG_FALLBACK_ON_ERROR="${DEBUG_FALLBACK_ON_ERROR:-0}"
SKIP_LABELING="${SKIP_LABELING:-1}"
SECURE_AGGREGATION="${SECURE_AGGREGATION:-1}"
SECURE_NUM_HELPERS="${SECURE_NUM_HELPERS:-5}"
SECURE_PRIVACY_THRESHOLD="${SECURE_PRIVACY_THRESHOLD:-2}"
SECURE_RECONSTRUCTION_THRESHOLD="${SECURE_RECONSTRUCTION_THRESHOLD:-}"
SECURE_FIELD_MODULUS="${SECURE_FIELD_MODULUS:-2147483647}"
SECURE_QUANTIZATION_SCALE="${SECURE_QUANTIZATION_SCALE:-65536}"
SECURE_SEED="${SECURE_SEED:-0}"
CLUSTERED="${CLUSTERED:-0}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-secure_kmeans}"
CLUSTERING_REPRESENTATION="${CLUSTERING_REPRESENTATION:-model}"
CLUSTERING_NORMALIZE_VECTOR="${CLUSTERING_NORMALIZE_VECTOR:-1}"
CLUSTERING_NORMALIZATION_MODE="${CLUSTERING_NORMALIZATION_MODE:-l2}"
CLUSTERING_DELTA_OVER_BASE_NORM="${CLUSTERING_DELTA_OVER_BASE_NORM:-1}"
CLUSTERING_K="${CLUSTERING_K:-3}"
CLUSTERING_NUM_RESTARTS="${CLUSTERING_NUM_RESTARTS:-5}"
CLUSTERING_ENABLE_PCA="${CLUSTERING_ENABLE_PCA:-1}"
CLUSTERING_PCA_COMPONENTS="${CLUSTERING_PCA_COMPONENTS:-8}"
CLUSTERING_WARMUP_ROUNDS="${CLUSTERING_WARMUP_ROUNDS:-0}"
CLUSTERING_FREEZE_PCA_AFTER_WARMUP="${CLUSTERING_FREEZE_PCA_AFTER_WARMUP:-0}"
TOP_K="${TOP_K:-1,3,5}"
FORCE_TRAINING="${FORCE_TRAINING:-1}"
EVAL_OUTPUT="${EVAL_OUTPUT:-}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    echo "ERROR: no Python executable found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

if [[ ! "$PERSONA_ASSIGNMENT_POLICY" =~ ^(fixed|dirichlet_sampled)$ ]]; then
  echo "ERROR: PERSONA_ASSIGNMENT_POLICY must be fixed or dirichlet_sampled." >&2
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
if [[ ! "$CLUSTERING_NUM_RESTARTS" =~ ^[0-9]+$ ]] || [[ "$CLUSTERING_NUM_RESTARTS" -lt 1 ]]; then
  echo "ERROR: CLUSTERING_NUM_RESTARTS must be a positive integer." >&2
  exit 2
fi

if [[ "$PERSONA_ASSIGNMENT_POLICY" == "fixed" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  FIXED_PERSONA_CONFIG_PATH="$PROJECT_ROOT/src/fed_perso_xai/recommender/configs/${FIXED_PERSONA}.yaml"
  if [[ ! -f "$FIXED_PERSONA_CONFIG_PATH" ]]; then
    echo "ERROR: FIXED_PERSONA='$FIXED_PERSONA' does not resolve to a bundled persona config at $FIXED_PERSONA_CONFIG_PATH." >&2
    exit 2
  fi
fi

TRAIN_EXTRA=()
if [[ "$DEBUG_FALLBACK_ON_ERROR" == "1" ]]; then
  TRAIN_EXTRA+=(--debug-fallback-on-error)
fi
if [[ "$FORCE_TRAINING" == "1" ]]; then
  TRAIN_EXTRA+=(--force)
fi
if [[ "$SECURE_AGGREGATION" == "1" ]]; then
  TRAIN_EXTRA+=(--secure-aggregation)
fi
if [[ "$CLUSTERED" == "1" ]]; then
  TRAIN_EXTRA+=(--clustered)
fi
TRAIN_EXTRA+=(--secure-num-helpers "$SECURE_NUM_HELPERS")
TRAIN_EXTRA+=(--secure-privacy-threshold "$SECURE_PRIVACY_THRESHOLD")
if [[ -n "$SECURE_RECONSTRUCTION_THRESHOLD" ]]; then
  TRAIN_EXTRA+=(--secure-reconstruction-threshold "$SECURE_RECONSTRUCTION_THRESHOLD")
fi
TRAIN_EXTRA+=(--secure-field-modulus "$SECURE_FIELD_MODULUS")
TRAIN_EXTRA+=(--secure-quantization-scale "$SECURE_QUANTIZATION_SCALE")
TRAIN_EXTRA+=(--secure-seed "$SECURE_SEED")
TRAIN_EXTRA+=(--clustering-method "$CLUSTERING_METHOD")
TRAIN_EXTRA+=(--clustering-representation "$CLUSTERING_REPRESENTATION")
if [[ "$CLUSTERING_NORMALIZE_VECTOR" == "0" ]]; then
  TRAIN_EXTRA+=(--no-clustering-normalize-vector)
else
  TRAIN_EXTRA+=(--clustering-normalize-vector)
fi
TRAIN_EXTRA+=(--clustering-normalization-mode "$CLUSTERING_NORMALIZATION_MODE")
if [[ "$CLUSTERING_DELTA_OVER_BASE_NORM" == "0" ]]; then
  TRAIN_EXTRA+=(--no-clustering-delta-over-base-norm)
else
  TRAIN_EXTRA+=(--clustering-delta-over-base-norm)
fi
TRAIN_EXTRA+=(--clustering-k "$CLUSTERING_K")
TRAIN_EXTRA+=(--clustering-num-restarts "$CLUSTERING_NUM_RESTARTS")
if [[ "$CLUSTERING_ENABLE_PCA" == "0" ]]; then
  TRAIN_EXTRA+=(--no-clustering-enable-pca)
else
  TRAIN_EXTRA+=(--clustering-enable-pca)
fi
TRAIN_EXTRA+=(--clustering-pca-components "$CLUSTERING_PCA_COMPONENTS")
TRAIN_EXTRA+=(--clustering-warmup-rounds "$CLUSTERING_WARMUP_ROUNDS")
if [[ "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP" == "1" ]]; then
  TRAIN_EXTRA+=(--clustering-freeze-pca-after-warmup)
fi

EVAL_EXTRA=()
if [[ "$CLUSTERED" == "1" ]]; then
  EVAL_EXTRA+=(--clustered)
elif [[ "$SECURE_AGGREGATION" == "1" ]]; then
  EVAL_EXTRA+=(--secure-aggregation)
else
  EVAL_EXTRA+=(--plain-aggregation)
fi
if [[ -n "$EVAL_OUTPUT" ]]; then
  EVAL_EXTRA+=(--output "$EVAL_OUTPUT")
fi

LABEL_EXTRA=(
  --persona-assignment-policy "$PERSONA_ASSIGNMENT_POLICY"
  --output-persona "$LABEL_NAMESPACE"
)
if [[ -n "$PERSONA_ASSIGNMENT_ALPHA" ]]; then
  LABEL_EXTRA+=(--persona-assignment-alpha "$PERSONA_ASSIGNMENT_ALPHA")
fi
LABEL_PERSONA_ARGS=()
if [[ "$PERSONA_ASSIGNMENT_POLICY" == "fixed" ]]; then
  LABEL_PERSONA_ARGS+=(--persona "$FIXED_PERSONA")
fi

if [[ "$SKIP_LABELING" == "1" ]]; then
  echo "==> Skipping recommender labeling"
else
  echo "==> Labeling recommender context"
  "$PYTHON" -m fed_perso_xai label-recommender-context \
    --run-id "$RUN_ID" \
    --selection "$SELECTION_ID" \
    "${LABEL_PERSONA_ARGS[@]}" \
    --simulator "$SIMULATOR" \
    --clients "$CLIENTS" \
    --context-filename "$CONTEXT_FILENAME" \
    --label-filename "$LABEL_FILENAME" \
    --seed "$PERSONA_SEED" \
    --label-seed "$LABEL_SEED" \
    --instance-test-size "$INSTANCE_TEST_SIZE" \
    --instance-validation-size "$INSTANCE_VALIDATION_SIZE" \
    "${LABEL_EXTRA[@]}"
fi

echo "==> Training federated recommender"
"$PYTHON" -m fed_perso_xai train-recommender-federated \
  --run-id "$RUN_ID" \
  --selection "$SELECTION_ID" \
  --persona "$LABEL_NAMESPACE" \
  --clients "$CLIENTS" \
  --context-filename "$CONTEXT_FILENAME" \
  --label-filename "$LABEL_FILENAME" \
  --recommender "$RECOMMENDER_TYPE" \
  --rounds "$TRAIN_ROUNDS" \
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --learning-rate "$TRAIN_LEARNING_RATE" \
  --l2-regularization "$TRAIN_L2_REGULARIZATION" \
  --svm-c "$TRAIN_SVM_C" \
  --svm-intercept-scaling "$TRAIN_SVM_INTERCEPT_SCALING" \
  --seed "$TRAIN_SEED" \
  --fit-fraction "$FIT_FRACTION" \
  --evaluate-fraction "$EVALUATE_FRACTION" \
  --min-available-clients "$MIN_AVAILABLE_CLIENTS" \
  --simulation-backend "$SIMULATION_BACKEND" \
  --ray-num-cpus "$RAY_NUM_CPUS" \
  --client-num-cpus "$CLIENT_NUM_CPUS" \
  --top-k "$TOP_K" \
  "${TRAIN_EXTRA[@]}"

echo "==> Evaluating federated recommender"
"$PYTHON" -m fed_perso_xai evaluate-recommender \
  --run-id "$RUN_ID" \
  --selection "$SELECTION_ID" \
  --persona "$LABEL_NAMESPACE" \
  --clients "$CLIENTS" \
  --context-filename "$CONTEXT_FILENAME" \
  --label-filename "$LABEL_FILENAME" \
  --recommender "$RECOMMENDER_TYPE" \
  --top-k "$TOP_K" \
  "${EVAL_EXTRA[@]}"

echo "==> Recommender pipeline complete"
echo "Run ID: $RUN_ID"
echo "Selection: $SELECTION_ID"
echo "Label Namespace: $LABEL_NAMESPACE"
