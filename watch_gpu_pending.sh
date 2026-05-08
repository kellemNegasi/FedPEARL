#!/usr/bin/env bash
# watch_gpu_pending.sh h200-141g 15
set -euo pipefail

GPU_TYPE="${1:-h200-141g}"
INTERVAL="${2:-10}"
PARTITION="${PARTITION:-gpu}"

while true; do
  clear
  echo "Time: $(date '+%F %T')"
  echo "Partition: ${PARTITION}"
  echo "GPU type: ${GPU_TYPE}"
  echo

  TMP_FILE="$(mktemp)"
  squeue -p "${PARTITION}" -t PD \
    -o "%.18i %.8u %.10T %.12M %.10l %.6D %.20R %.24b" > "${TMP_FILE}"

  echo "Pending jobs requesting ${GPU_TYPE}:"
  awk -v gpu="${GPU_TYPE}" '
    NR==1 { print; next }
    $0 ~ ("gres/gpu:" gpu) { print; count++ }
    END { printf("\nCount: %d\n", count+0) }
  ' "${TMP_FILE}"

  rm -f "${TMP_FILE}"
  sleep "${INTERVAL}"
done
