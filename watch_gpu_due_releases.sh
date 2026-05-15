#!/usr/bin/env bash
# watch_gpu_due_releases.sh h200-141g 4h
set -euo pipefail

GPU_TYPE="${1:-a100-80g}"
WINDOW_RAW="${2:-1h}"
PARTITION="${PARTITION:-gpu}"

if ! command -v sinfo >/dev/null 2>&1; then
  echo "ERROR: sinfo not found in PATH." >&2
  exit 1
fi
if ! command -v squeue >/dev/null 2>&1; then
  echo "ERROR: squeue not found in PATH." >&2
  exit 1
fi
if ! command -v scontrol >/dev/null 2>&1; then
  echo "ERROR: scontrol not found in PATH." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

NODE_GRES_FILE="${TMP_DIR}/node_gres.tsv"
RUNNING_JOBS_FILE="${TMP_DIR}/running_jobs.tsv"

sinfo -p "${PARTITION}" -N -h -o '%N|%G' > "${NODE_GRES_FILE}"
squeue -p "${PARTITION}" -t R -h -o '%i|%u|%M|%l|%D|%R|%b' > "${RUNNING_JOBS_FILE}"

GPU_TYPE="${GPU_TYPE}" \
WINDOW_RAW="${WINDOW_RAW}" \
PARTITION="${PARTITION}" \
NODE_GRES_FILE="${NODE_GRES_FILE}" \
RUNNING_JOBS_FILE="${RUNNING_JOBS_FILE}" \
python3 - <<'PY'
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


GPU_TYPE = os.environ["GPU_TYPE"].strip()
WINDOW_RAW = os.environ["WINDOW_RAW"].strip().lower()
PARTITION = os.environ["PARTITION"].strip()


def parse_window_to_seconds(raw: str) -> int:
    text = raw.strip().lower()
    if not text:
        raise ValueError("window must not be empty")
    if text.isdigit():
        return int(text) * 3600
    match = re.fullmatch(r"(\d+)([smhd])", text)
    if not match:
        raise ValueError(
            "window must be an integer number of hours or use one of the suffixes s, m, h, d"
        )
    value = int(match.group(1))
    unit = match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


def parse_slurm_duration(raw: str) -> int:
    text = raw.strip()
    if not text or text in {"UNLIMITED", "NOT_SET"}:
        raise ValueError(f"unsupported Slurm duration: {raw!r}")
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = parts[0]
    else:
        raise ValueError(f"unsupported Slurm duration: {raw!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_seconds(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_gpu_count(tres_per_node: str) -> int:
    text = (tres_per_node or "").strip()
    if not text:
        return 0

    explicit_match = re.search(rf"gres/gpu:{re.escape(GPU_TYPE)}(?::(\d+))?\b", text)
    if explicit_match:
        count = explicit_match.group(1)
        return int(count) if count is not None else 1

    generic_match = re.search(r"gres/gpu(?::[^:,|]+)?:(\d+)\b", text)
    if generic_match:
        return int(generic_match.group(1))

    bare_generic = re.search(r"\bgres/gpu\b", text)
    if bare_generic:
        return 1

    return 0


def node_matches_gpu_type(node_gres: str) -> bool:
    return f"gpu:{GPU_TYPE}:" in (node_gres or "")


def expand_nodes(nodelist: str) -> list[str]:
    output = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


@dataclass
class JobRelease:
    jobid: str
    user: str
    elapsed_seconds: int
    timelimit_seconds: int
    remaining_seconds: int
    nodes: str
    tres_per_node: str
    gpu_count: int


window_seconds = parse_window_to_seconds(WINDOW_RAW)
node_gres_path = Path(os.environ["NODE_GRES_FILE"])
running_jobs_path = Path(os.environ["RUNNING_JOBS_FILE"])

node_gres: dict[str, str] = {}
with node_gres_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle, delimiter="|")
    for row in reader:
        if len(row) != 2:
            continue
        node_name, gres = row
        node_gres[node_name.strip()] = gres.strip()

releases: list[JobRelease] = []
with running_jobs_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle, delimiter="|")
    for row in reader:
        if len(row) != 7:
            continue
        jobid, user, elapsed_raw, timelimit_raw, _nodes_count, nodelist, tres_per_node = row
        nodelist = nodelist.strip()
        if not nodelist or nodelist.startswith("("):
            continue

        try:
            elapsed_seconds = parse_slurm_duration(elapsed_raw)
            timelimit_seconds = parse_slurm_duration(timelimit_raw)
        except ValueError:
            continue

        remaining_seconds = max(0, timelimit_seconds - elapsed_seconds)
        if remaining_seconds > window_seconds:
            continue

        try:
            expanded_nodes = expand_nodes(nodelist)
        except subprocess.CalledProcessError:
            continue

        matching_nodes = [
            node for node in expanded_nodes if node_matches_gpu_type(node_gres.get(node, ""))
        ]
        if not matching_nodes:
            continue

        gpu_per_node = parse_gpu_count(tres_per_node)
        if gpu_per_node <= 0:
            continue

        releases.append(
            JobRelease(
                jobid=jobid.strip(),
                user=user.strip(),
                elapsed_seconds=elapsed_seconds,
                timelimit_seconds=timelimit_seconds,
                remaining_seconds=remaining_seconds,
                nodes=",".join(matching_nodes),
                tres_per_node=tres_per_node.strip(),
                gpu_count=gpu_per_node * len(matching_nodes),
            )
        )

releases.sort(key=lambda item: (item.remaining_seconds, item.jobid))
total_gpus = sum(item.gpu_count for item in releases)

print(f"Partition: {PARTITION}")
print(f"GPU type: {GPU_TYPE}")
print(f"Window: {WINDOW_RAW} ({window_seconds} seconds)")
print(f"Running jobs due within window: {len(releases)}")
print(f"GPUs expected to be freed within window: {total_gpus}")
print()
print(
    f"{'JOBID':<18} {'USER':<10} {'REMAINING':<12} {'GPUS':<4} {'NODES':<24} TRES_PER_NODE"
)
for item in releases:
    print(
        f"{item.jobid:<18} "
        f"{item.user:<10} "
        f"{format_seconds(item.remaining_seconds):<12} "
        f"{item.gpu_count:<4} "
        f"{item.nodes:<24} "
        f"{item.tres_per_node}"
    )
PY
