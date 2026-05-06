```bash
sbatch scripts/plan_launcher_experiments.sbatch configs/job_launcher.yml
```
Prepares the dataset for all Client(K) X alpha configuration, in our case 12 configs.
if you rerun planning without specifying OUTPUT_DIR, it creates a new timestamped plan directory
if you explicitly reuse the same OUTPUT_DIR, it will overwrite the manifest files inside that plan directory

This dumps jsonl launcher experiments like this 
```bash
job_launcher/plans/job_launcher__launcher__20260505T230705/launcher_experiments.jsonl
```

Then submit the training array like so
```bash
sbatch --array=0-11 scripts/train_predictive_array.sbatch job_launcher/plans/job_launcher__launcher__20260505T230705/launcher_experiments.jsonl
```

Then after that use this to plan the run and submit the slurms

```bash
SUBMIT_SLURM=1 sbatch scripts/plan_explain_eval_from_training.sbatch \
  configs/job_launcher.yml \
  job_launcher/plans/job_launcher__launcher__20260505T230705/launcher_experiments.jsonl
```

Once This is done, check if all jobs as finished and are marked as done using the noteobok in `notebooks/explain_eval_completion_report.ipynb`

After that run the `scripts/submit_aggregate_explain_eval.sh` script by updating the list of run ids manually.

```bash
bash ./scripts/submit_aggregate_explain_eval.sh
```

Using the same run IDs used in the aggregator, run the followiong context 

```bash
 bash scripts/prepare_recommender_context.sh
```

Finally train recommendation using:
```bash
scripts/submit_pipeline.sh
```
There are a log of modle training and resource management configuration to chose here. The configuration for LCC can also be set here meaning the modulus prime and the quantization scale. 