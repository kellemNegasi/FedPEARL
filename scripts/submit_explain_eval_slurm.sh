#!/usr/bin/env bash

set -euo pipefail

# Submit every explain/eval Slurm launcher script generated under job_launcher/slurm.
# Each matching .sbatch file is passed directly to sbatch.
shopt -s nullglob
for sbatch_file in job_launcher/slurm/explain_eval*.sbatch; do
  sbatch "$sbatch_file"
done
