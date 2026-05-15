#!/usr/bin/env bash
set -euo pipefail

SLURM_FILES=(
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-5__alpha-0.1__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t222902686786-249ff01e8075.sbatch"
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-5__alpha-0.3__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t222902686725-ef898b9a73ab.sbatch"
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-5__alpha-1.0__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t224346667160-9fa18cd5e96f.sbatch"
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-5__alpha-10.0__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t224346667151-eb09212a1574.sbatch"
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-10__alpha-0.1__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t225837614057-9db23665e6c0.sbatch"
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-10__alpha-0.3__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t225837612250-980852e07e3d.sbatch"
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-10__alpha-1.0__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t231552481693-9712ebbf1103.sbatch"
  "plans/cencus_income/mlp_classifier/slurm/cencus_income__clients-10__alpha-10.0__seed-42__mlp_classifier-epochs5-batch256-lr0.005-l20.01-hidden100-actrelu-optadam-devcpu__plan-11649661bed3__run-20260506t231552480561-718fb140b9eb.sbatch"
)

for sbatch_file in "${SLURM_FILES[@]}"; do
  echo "Submitting $sbatch_file"
  sbatch "$sbatch_file"
done
