#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"^federated-training-"
    r"(?P<dataset>.+)-"
    r"(?P<timestamp>\d{8}t\d+\+\d{4})-"
    r"(?P<model>[^-]+)-"
    r"(?P<num_clients>\d+)clients-"
    r"alpha(?P<alpha>[0-9.]+)-"
    r"seed(?P<seed>\d+)-"
    r"(?P<suffix>[A-Za-z0-9]+)$"
)

CANONICAL_FILES = [
    "config_snapshot.json",
    "dataset_metadata.json",
    "feature_metadata.json",
    "partition_metadata.json",
    "reproducibility_metadata.json",
    "run_manifest.json",
    "split_metadata.json",
    "training/runtime_report.json",
    "training/training.done",
    "training/training_history.csv",
    "training/training_metadata.json",
]

RUN_FILES = [
    "config_snapshot.json",
    "dataset_metadata.json",
    "feature_metadata.json",
    "partition_metadata.json",
    "reproducibility_metadata.json",
    "run_manifest.json",
    "run_metadata.json",
    "split_metadata.json",
    "model/global_model.npz",
    "model/model_metadata.json",
    "preprocessor.joblib",
    "training/runtime_report.json",
    "training/training_history.csv",
    "training/training_metadata.json",
]


@dataclass
class ConfigInfo:
    dataset: str
    num_clients: int
    alpha: str
    seed: int
    canonical_dir: Path
    run_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a repo-local snapshot of current federated model training "
            "artifacts under tracker/model-training."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tracker/model-training"),
        help="Snapshot output directory, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name to include. Repeat to include multiple datasets.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Seed to include. Repeat to include multiple seeds.",
    )
    return parser.parse_args()


def run_cmd(repo_root: Path, args: list[str]) -> str:
    return subprocess.check_output(args, cwd=repo_root, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_output_dir(repo_root: Path, output_dir: Path) -> Path:
    return output_dir if output_dir.is_absolute() else repo_root / output_dir


def iter_canonical_configs(
    repo_root: Path,
    datasets: set[str] | None,
    seeds: set[int] | None,
) -> list[ConfigInfo]:
    federated_root = repo_root / "federated"
    configs: list[ConfigInfo] = []
    for dataset_dir in sorted(federated_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "runs":
            continue
        if datasets and dataset_dir.name not in datasets:
            continue
        for clients_dir in sorted(dataset_dir.glob("*_clients")):
            try:
                num_clients = int(clients_dir.name.split("_", 1)[0])
            except ValueError:
                continue
            for alpha_dir in sorted(clients_dir.glob("alpha_*")):
                alpha = alpha_dir.name.split("_", 1)[1]
                for seed_dir in sorted(alpha_dir.glob("seed_*")):
                    try:
                        seed = int(seed_dir.name.split("_", 1)[1])
                    except ValueError:
                        continue
                    if seeds and seed not in seeds:
                        continue
                    if seed_dir.is_dir():
                        configs.append(
                            ConfigInfo(
                                dataset=dataset_dir.name,
                                num_clients=num_clients,
                                alpha=alpha,
                                seed=seed,
                                canonical_dir=seed_dir,
                            )
                        )
    return sorted(configs, key=lambda cfg: (cfg.dataset, cfg.num_clients, float(cfg.alpha), cfg.seed))


def parse_run_name(run_dir: Path) -> dict[str, Any] | None:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return None
    data = match.groupdict()
    data["num_clients"] = int(data["num_clients"])
    data["seed"] = int(data["seed"])
    return data


def build_run_map(
    repo_root: Path,
    datasets: set[str] | None,
    seeds: set[int] | None,
) -> dict[tuple[str, int, str, int], Path]:
    runs_root = repo_root / "federated" / "runs"
    candidates: dict[tuple[str, int, str, int], list[Path]] = {}
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir)
        if not parsed:
            continue
        dataset = parsed["dataset"]
        seed = parsed["seed"]
        if datasets and dataset not in datasets:
            continue
        if seeds and seed not in seeds:
            continue
        key = (dataset, parsed["num_clients"], parsed["alpha"], seed)
        candidates.setdefault(key, []).append(run_dir)

    selected: dict[tuple[str, int, str, int], Path] = {}
    for key, run_dirs in candidates.items():
        selected[key] = max(run_dirs, key=run_sort_key)
    return selected


def run_sort_key(run_dir: Path) -> tuple[str, str]:
    metadata_path = run_dir / "run_metadata.json"
    created_at = ""
    if metadata_path.exists():
        try:
            created_at = load_json(metadata_path).get("created_at", "")
        except json.JSONDecodeError:
            created_at = ""
    parsed = parse_run_name(run_dir)
    timestamp = parsed["timestamp"] if parsed else run_dir.name
    return (created_at, timestamp)


def copy_if_exists(
    repo_root: Path,
    tracker_root: Path,
    src: Path,
    destination_root: Path,
    manifest: list[dict[str, Any]],
    category: str,
) -> None:
    if not src.exists():
        return
    rel = src.relative_to(repo_root)
    dest = destination_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    manifest.append(
        {
            "category": category,
            "source": str(rel),
            "destination": str(dest.relative_to(tracker_root)),
            "size_bytes": src.stat().st_size,
            "sha256": sha256_file(src),
            "modified_at": datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
    )


def parse_training_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fit_metrics = json.loads(row["fit_metrics_json"]) if row.get("fit_metrics_json") else {}
            eval_metrics = json.loads(row["evaluate_metrics_json"]) if row.get("evaluate_metrics_json") else {}
            rows.append(
                {
                    "round": int(row["round"]),
                    "evaluate_loss": float(row["evaluate_loss"]),
                    "aggregation_mode": row.get("aggregation_mode") or None,
                    "aggregation_num_contributors": (
                        int(float(row["aggregation_num_contributors"]))
                        if row.get("aggregation_num_contributors")
                        else None
                    ),
                    "aggregation_helper_count": (
                        int(float(row["aggregation_helper_count"]))
                        if row.get("aggregation_helper_count")
                        else None
                    ),
                    "train_loss": fit_metrics.get("train_loss"),
                    "fit_num_examples": fit_metrics.get("num_examples"),
                    "shared_parameter_count": fit_metrics.get("shared_parameter_count"),
                    "accuracy": eval_metrics.get("accuracy"),
                    "f1": eval_metrics.get("f1"),
                    "precision": eval_metrics.get("precision"),
                    "recall": eval_metrics.get("recall"),
                    "roc_auc": eval_metrics.get("roc_auc"),
                    "threshold": eval_metrics.get("threshold"),
                    "eval_num_examples": eval_metrics.get("num_examples"),
                    "loss": eval_metrics.get("loss"),
                }
            )
    return rows


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary_readme(
    tracker_root: Path,
    snapshot_created_at_utc: str,
    head_commit: str,
    branch: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Model Training Snapshot",
        "",
        f"- Snapshot created at UTC: `{snapshot_created_at_utc}`",
        f"- Repository commit: `{head_commit}`",
        f"- Branch: `{branch}`",
        f"- Captured configs: `{len(summary_rows)}`",
        "",
        "## Included Configs",
        "",
        "| dataset | clients | alpha | seed | final_accuracy | final_f1 | final_roc_auc | final_eval_loss | run_id |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    str(row["num_clients"]),
                    row["alpha"],
                    str(row["seed"]),
                    format_metric(row["final_accuracy"]),
                    format_metric(row["final_f1"]),
                    format_metric(row["final_roc_auc"]),
                    format_metric(row["final_eval_loss"]),
                    f"`{row['run_id']}`" if row["run_id"] else "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Contents",
            "",
            "- `summary/performance_summary.csv`: final and best metrics per config.",
            "- `summary/performance_summary.json`: same summary in JSON form.",
            "- `summary/copied_files_manifest.json`: copied file inventory with SHA256 and sizes.",
            "- `raw/.../canonical`: copied current canonical training outputs from `federated/<dataset>/...`.",
            "- `raw/.../run`: copied timestamped run artifacts from `federated/runs/...`, including model binaries and run metadata.",
            "- `repo-config/`: copied repo-level configuration files relevant to training and launch setup.",
            "",
            "## Notes",
            "",
            "- This tracker snapshot is intentionally repo-local and can remain untracked.",
            "- Re-running the generator overwrites this snapshot directory.",
        ]
    )
    (tracker_root / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve_output_dir(repo_root, args.output_dir)
    datasets = set(args.datasets) if args.datasets else None
    seeds = set(args.seeds) if args.seeds else None

    canonical_configs = iter_canonical_configs(repo_root, datasets, seeds)
    if not canonical_configs:
        raise SystemExit("No canonical federated configs found for the requested filters.")

    run_map = build_run_map(repo_root, datasets, seeds)
    for cfg in canonical_configs:
        cfg.run_dir = run_map.get((cfg.dataset, cfg.num_clients, cfg.alpha, cfg.seed))

    if output_dir.exists():
        shutil.rmtree(output_dir)

    summary_dir = output_dir / "summary"
    raw_dir = output_dir / "raw"
    repo_config_dir = output_dir / "repo-config"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo_config_dir.mkdir(parents=True, exist_ok=True)

    copied_manifest: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    source_paths: list[dict[str, Any]] = []

    for cfg in canonical_configs:
        config_key = f"{cfg.dataset}/{cfg.num_clients}_clients/alpha_{cfg.alpha}/seed_{cfg.seed}"
        raw_dest_root = raw_dir / config_key

        for rel_path in CANONICAL_FILES:
            copy_if_exists(
                repo_root,
                output_dir,
                cfg.canonical_dir / rel_path,
                raw_dest_root / "canonical",
                copied_manifest,
                "canonical",
            )

        run_metadata: dict[str, Any] = {}
        run_manifest: dict[str, Any] = {}
        if cfg.run_dir is not None:
            for rel_path in RUN_FILES:
                copy_if_exists(
                    repo_root,
                    output_dir,
                    cfg.run_dir / rel_path,
                    raw_dest_root / "run",
                    copied_manifest,
                    "run",
                )
            for client_meta in sorted((cfg.run_dir / "clients").glob("client_*/client_metadata.json")):
                copy_if_exists(
                    repo_root,
                    output_dir,
                    client_meta,
                    raw_dest_root / "run",
                    copied_manifest,
                    "run_client_metadata",
                )
            run_metadata_path = cfg.run_dir / "run_metadata.json"
            run_manifest_path = cfg.run_dir / "run_manifest.json"
            if run_metadata_path.exists():
                run_metadata = load_json(run_metadata_path)
            if run_manifest_path.exists():
                run_manifest = load_json(run_manifest_path)

        history_path = cfg.canonical_dir / "training" / "training_history.csv"
        if not history_path.exists():
            continue
        history_rows = parse_training_history(history_path)
        if not history_rows:
            continue

        final_row = max(history_rows, key=lambda row: row["round"])
        best_accuracy_row = max(history_rows, key=lambda row: ((row["accuracy"] or float("-inf")), -row["round"]))
        best_f1_row = max(history_rows, key=lambda row: ((row["f1"] or float("-inf")), -row["round"]))
        best_roc_auc_row = max(history_rows, key=lambda row: ((row["roc_auc"] or float("-inf")), -row["round"]))

        training_config = run_metadata.get("training_config", {})
        summary_rows.append(
            {
                "dataset": cfg.dataset,
                "num_clients": cfg.num_clients,
                "alpha": cfg.alpha,
                "seed": cfg.seed,
                "canonical_dir": str(cfg.canonical_dir.relative_to(repo_root)),
                "run_dir": str(cfg.run_dir.relative_to(repo_root)) if cfg.run_dir else "",
                "run_id": run_metadata.get("run_id", cfg.run_dir.name if cfg.run_dir else ""),
                "run_created_at": run_metadata.get("created_at", ""),
                "run_completed_at": run_metadata.get("completed_at", ""),
                "run_status": run_metadata.get("status", ""),
                "run_git_commit_hash": run_manifest.get("git_commit_hash", ""),
                "secure_aggregation": training_config.get("secure_aggregation"),
                "secure_num_helpers": training_config.get("secure_num_helpers"),
                "secure_privacy_threshold": training_config.get("secure_privacy_threshold"),
                "rounds_recorded": len(history_rows),
                "final_round": final_row["round"],
                "final_accuracy": final_row["accuracy"],
                "final_f1": final_row["f1"],
                "final_precision": final_row["precision"],
                "final_recall": final_row["recall"],
                "final_roc_auc": final_row["roc_auc"],
                "final_eval_loss": final_row["loss"],
                "final_train_loss": final_row["train_loss"],
                "final_threshold": final_row["threshold"],
                "final_aggregation_mode": final_row["aggregation_mode"],
                "final_aggregation_num_contributors": final_row["aggregation_num_contributors"],
                "final_aggregation_helper_count": final_row["aggregation_helper_count"],
                "best_accuracy": best_accuracy_row["accuracy"],
                "best_accuracy_round": best_accuracy_row["round"],
                "best_f1": best_f1_row["f1"],
                "best_f1_round": best_f1_row["round"],
                "best_roc_auc": best_roc_auc_row["roc_auc"],
                "best_roc_auc_round": best_roc_auc_row["round"],
            }
        )
        source_paths.append(
            {
                "config": {
                    "dataset": cfg.dataset,
                    "num_clients": cfg.num_clients,
                    "alpha": cfg.alpha,
                    "seed": cfg.seed,
                },
                "canonical_dir": str(cfg.canonical_dir.relative_to(repo_root)),
                "run_dir": str(cfg.run_dir.relative_to(repo_root)) if cfg.run_dir else None,
            }
        )

    if not summary_rows:
        raise SystemExit("No training histories were found to summarize.")

    repo_config_paths = [repo_root / "pyproject.toml", repo_root / "scripts" / "train_predictive_array.sbatch"]
    repo_config_paths.extend(sorted((repo_root / "configs").glob("*.yml")))
    repo_config_paths.extend(sorted((repo_root / "configs").glob("*.yaml")))
    for path in repo_config_paths:
        copy_if_exists(repo_root, output_dir, path, repo_config_dir, copied_manifest, "repo_config")

    snapshot_created_at_utc = datetime.now(timezone.utc).isoformat()
    head_commit = run_cmd(repo_root, ["git", "rev-parse", "HEAD"])
    branch = run_cmd(repo_root, ["git", "branch", "--show-current"])
    status_short = run_cmd(repo_root, ["git", "status", "--short"])

    summary_rows.sort(key=lambda row: (row["dataset"], row["num_clients"], float(row["alpha"]), row["seed"]))
    fieldnames = list(summary_rows[0].keys())
    with (summary_dir / "performance_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    (summary_dir / "performance_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    (summary_dir / "copied_files_manifest.json").write_text(json.dumps(copied_manifest, indent=2) + "\n")
    (output_dir / "snapshot_metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "model_training_tracker_snapshot",
                "created_at_utc": snapshot_created_at_utc,
                "repository_root": str(repo_root),
                "git_head_commit": head_commit,
                "git_branch": branch,
                "git_status_short": status_short.splitlines() if status_short else [],
                "captured_config_count": len(summary_rows),
                "captured_run_count": sum(1 for row in summary_rows if row["run_id"]),
                "source_paths": source_paths,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "git_head.txt").write_text(head_commit + "\n")
    (output_dir / "git_status_short.txt").write_text(status_short + ("\n" if status_short else ""))
    write_summary_readme(output_dir, snapshot_created_at_utc, head_commit, branch, summary_rows)

    payload = {
        "tracker_root": str(output_dir.relative_to(repo_root)),
        "captured_configs": len(summary_rows),
        "captured_files": len(copied_manifest),
        "head_commit": head_commit,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
