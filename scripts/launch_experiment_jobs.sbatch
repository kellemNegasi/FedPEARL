#!/bin/bash
#
# Slurm job for running the launcher with the configured experiment matrix.
# Submit with:
#   sbatch scripts/launch_experiment_jobs.sbatch [CONFIG_PATH]
#

#SBATCH --job-name=launch-jobs
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=81G
#SBATCH --time=00:30:00

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
  echo "Usage: sbatch scripts/launch_experiment_jobs.sbatch [CONFIG_PATH]" >&2
  echo "  CONFIG_PATH: optional launcher YAML. Defaults to configs/job_launcher.yml" >&2
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

CONFIG_PATH="${1:-configs/job_launcher.yml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config path does not exist: $CONFIG_PATH" >&2
  exit 2
fi

mkdir -p logs

echo "[$(date --iso-8601=seconds)] Starting launcher run"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"
echo "  config_path=${CONFIG_PATH}"

python3 -m fed_perso_xai launch-experiment-jobs --config "$CONFIG_PATH" --force

echo "[$(date --iso-8601=seconds)] Launcher run completed"
