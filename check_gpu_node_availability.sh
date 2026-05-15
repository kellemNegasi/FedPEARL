#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PARTITION="gpu"
PARTITION="$DEFAULT_PARTITION"
REQUESTED_GPU_TYPE=""
SHOW_DETAIL=0

print_help() {
  cat <<'EOF'
Usage:
  check_gpu_node_availability.sh [--detail] [partition] [gpu_type]
  check_gpu_node_availability.sh [--detail] [gpu_type]

Examples:
  check_gpu_node_availability.sh
  check_gpu_node_availability.sh gpu
  check_gpu_node_availability.sh main
  check_gpu_node_availability.sh gpu a100-40g
  check_gpu_node_availability.sh h200_141g
  check_gpu_node_availability.sh --detail gpu h200-141g

Notes:
- gpu_type should match the Slurm GRES type exactly when Slurm exposes it.
- Common aliases like h200_141g, h200-141gb, a100_40gb, and
  tesla_v100_32gb are normalized automatically.
- By default, only a compact Time_Left column is shown for matching jobs.
- Use --detail to also show shortened Job_IDs, launch times, and requested times.
- For nodes where Slurm only reports a generic GPU type like "tesla",
  this script applies a site-specific fallback mapping based on cluster docs.
- When gpu_type is provided, the output also includes running-job timing
  details for matching GPU allocations on each node.
- For partitions without GPU GRES, the script switches to a CPU/memory
  availability summary automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! have_cmd sinfo || ! have_cmd scontrol || ! have_cmd squeue; then
  echo "Error: this script requires 'sinfo', 'scontrol', and 'squeue'." >&2
  exit 1
fi

normalize_gpu_type() {
  local gpu_type="${1:-}"
  gpu_type="${gpu_type,,}"
  gpu_type="${gpu_type// /-}"
  gpu_type="${gpu_type//_/-}"
  gpu_type="${gpu_type//gb/g}"
  gpu_type="${gpu_type//tesla-v100-/v100-}"

  case "$gpu_type" in
    a100-40)  gpu_type="a100-40g" ;;
    a100-80)  gpu_type="a100-80g" ;;
    h200-141) gpu_type="h200-141g" ;;
    v100-16)  gpu_type="v100-16g" ;;
    v100-32)  gpu_type="v100-32g" ;;
  esac

  printf '%s\n' "$gpu_type"
}

partition_exists() {
  local candidate="${1%\*}"
  sinfo -h -o '%P' | sed 's/\*$//' | grep -Fxq -- "$candidate"
}

partition_has_gpu_resources() {
  local partition="$1"
  local nodes=""
  local node=""
  local line=""
  local gres=""

  nodes="$(sinfo -N -h -p "$partition" -o "%N" | sort -u)"

  while read -r node; do
    [[ -n "$node" ]] || continue
    line="$(scontrol show node -o "$node")"
    gres="$(field_value "$line" Gres)"
    if [[ -n "$(parse_gres_gpu_lines "${gres:-}")" ]]; then
      return 0
    fi
  done <<< "$nodes"

  return 1
}

parse_cli() {
  local positional_args=()
  local arg=""

  for arg in "$@"; do
    case "$arg" in
      --detail)
        SHOW_DETAIL=1
        ;;
      -h|--help)
        print_help
        exit 0
        ;;
      *)
        positional_args+=("$arg")
        ;;
    esac
  done

  parse_args "${positional_args[@]}"
}

parse_args() {
  case "$#" in
    0)
      PARTITION="$DEFAULT_PARTITION"
      REQUESTED_GPU_TYPE=""
      ;;
    1)
      if partition_exists "$1"; then
        PARTITION="${1%\*}"
        REQUESTED_GPU_TYPE=""
      else
        PARTITION="$DEFAULT_PARTITION"
        REQUESTED_GPU_TYPE="$(normalize_gpu_type "$1")"
      fi
      ;;
    2)
      if partition_exists "$1"; then
        PARTITION="${1%\*}"
        REQUESTED_GPU_TYPE="$(normalize_gpu_type "$2")"
      elif partition_exists "$2"; then
        PARTITION="${2%\*}"
        REQUESTED_GPU_TYPE="$(normalize_gpu_type "$1")"
      else
        echo "Error: neither '$1' nor '$2' is a known Slurm partition." >&2
        echo "Use either: check_gpu_node_availability.sh [partition] [gpu_type]" >&2
        echo "or:         check_gpu_node_availability.sh [gpu_type]" >&2
        exit 1
      fi
      ;;
    *)
      echo "Error: too many arguments." >&2
      echo "Use either: check_gpu_node_availability.sh [partition] [gpu_type]" >&2
      echo "or:         check_gpu_node_availability.sh [gpu_type]" >&2
      exit 1
      ;;
  esac
}

parse_cli "$@"

is_schedulable_state() {
  local state="${1,,}"
  local token=""
  local normalized="${state//+/ }"

  for token in $normalized; do
    case "$token" in
      drain*|drng*|down*|fail*|maint*|reserved|resv|powering_down|powered_down|pow_dn|unk*|inval*)
        return 1
        ;;
    esac
  done

  return 0
}

field_value() {
  local line="$1"
  local key="$2"
  awk -v key="$key" '
    {
      for (i = 1; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == key) {
          sub(/^[^=]+=/, "", $i)
          print $i
          exit
        }
      }
    }
  ' <<< "$line"
}

# Convert MB to integer GB, rounded to nearest GB.
mb_to_gb() {
  local mb="${1:-0}"
  awk -v mb="$mb" 'BEGIN { printf "%.0f", mb/1024 }'
}

# Site-specific fallback from documentation for nodes where Slurm only says "tesla".
fallback_gpu_type_for_node() {
  local node="$1"
  case "$node" in
    falcon1|falcon2) echo "v100-32g" ;;
    falcon3)         echo "v100-16g" ;;
    falcon4|falcon5|falcon6) echo "v100-32g" ;;
    pegasus)         echo "a100-40g" ;;
    pegasus2)        echo "a100-80g" ;;
    firefly1|firefly2|firefly3) echo "h200-141g" ;;
    *)               echo "" ;;
  esac
}

pretty_gpu_type() {
  local t="${1:-generic}"
  case "${t,,}" in
    a100-40g)  echo "A100_40GB" ;;
    a100-80g)  echo "A100_80GB" ;;
    h200-141g) echo "H200_141GB" ;;
    v100-16g)  echo "Tesla_V100_16GB" ;;
    v100-32g)  echo "Tesla_V100_32GB" ;;
    tesla)     echo "Tesla" ;;
    generic)   echo "generic" ;;
    *)         echo "$t" ;;
  esac
}

gpu_mem_from_type() {
  local t="${1:-}"
  case "${t,,}" in
    *141g*) echo "141" ;;
    *80g*)  echo "80" ;;
    *40g*)  echo "40" ;;
    *32g*)  echo "32" ;;
    *16g*)  echo "16" ;;
    *)      echo "unknown" ;;
  esac
}

# Parse Gres=... and print one GPU resource per line: type|count
parse_gres_gpu_lines() {
  local gres="$1"
  python3 - "$gres" <<'PY'
import re
import sys

gres = sys.argv[1].strip()
if not gres or gres == "(null)":
    raise SystemExit(0)

for item in gres.split(","):
    item = item.strip()
    item = re.sub(r"\(.*\)$", "", item)

    m = re.fullmatch(r"gpu(?::([^:]+))?(?::(\d+))?", item)
    if not m:
        continue

    gpu_type = m.group(1) or "generic"
    count = int(m.group(2) or "1")
    print(f"{gpu_type}|{count}")
PY
}

# Parse AllocTRES=... and print one GPU allocation per line: type|count
parse_alloc_gpu_lines() {
  local tres="$1"
  python3 - "$tres" <<'PY'
import re
import sys

tres = sys.argv[1].strip()
if not tres or tres == "(null)":
    raise SystemExit(0)

for part in tres.split(","):
    part = part.strip()
    m = re.fullmatch(r"gres/gpu(?::([^=]+))?=(\d+)", part)
    if not m:
        continue
    gpu_type = m.group(1) or "generic"
    count = int(m.group(2))
    print(f"{gpu_type}|{count}")
PY
}

# For a requested GPU type, build one TSV line per node:
# node<TAB>job_ids<TAB>launch_times<TAB>requested_times<TAB>remaining_times<TAB>time_left_summary
build_requested_gpu_job_timing_map() {
  local partition="$1"
  local requested_gpu_type="$2"

  python3 - "$requested_gpu_type" "$partition" <<'PY'
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

requested_gpu_type = sys.argv[1].strip().lower()
partition = sys.argv[2].strip()

fallback_gpu_type = {
    "falcon1": "v100-32g",
    "falcon2": "v100-32g",
    "falcon3": "v100-16g",
    "falcon4": "v100-32g",
    "falcon5": "v100-32g",
    "falcon6": "v100-32g",
    "pegasus": "a100-40g",
    "pegasus2": "a100-80g",
    "firefly1": "h200-141g",
    "firefly2": "h200-141g",
    "firefly3": "h200-141g",
}


def parse_tres_gpu_entries(tres):
    entries = []
    if not tres:
        return entries
    for part in tres.split(","):
        part = part.strip()
        match = re.fullmatch(r"gres/gpu(?::([^=]+))?=(\d+)", part)
        if not match:
            continue
        gpu_type = (match.group(1) or "generic").lower()
        count = int(match.group(2))
        entries.append((gpu_type, count))
    return entries


def parse_gres_detail_types(details):
    gpu_types = []
    for item in details or []:
        match = re.match(r"gpu(?::([^:()]+))?(?::\d+)?(?:\(.*\))?$", item)
        if not match:
            continue
        gpu_types.append((match.group(1) or "generic").lower())
    return gpu_types


def resolve_gpu_type(raw_gpu_type, node):
    gpu_type = (raw_gpu_type or "generic").lower()
    if gpu_type in {"generic", "tesla"}:
        return fallback_gpu_type.get(node, gpu_type)
    return gpu_type


def extract_nodes(job):
    allocated_nodes = (job.get("job_resources") or {}).get("allocated_nodes") or []
    nodes = [item.get("nodename", "") for item in allocated_nodes if item.get("nodename")]
    if nodes:
        return nodes

    node_field = (job.get("nodes") or "").strip()
    if node_field and "[" not in node_field and "," not in node_field:
        return [node_field]

    return []


def gpu_types_for_job_on_node(job, node):
    typed_entries = parse_tres_gpu_entries(job.get("tres_alloc_str", ""))
    typed_gpu_types = [
        resolve_gpu_type(gpu_type, node)
        for gpu_type, count in typed_entries
        if count > 0 and gpu_type != "generic"
    ]
    if typed_gpu_types:
        return typed_gpu_types

    gres_detail_types = [
        resolve_gpu_type(gpu_type, node)
        for gpu_type in parse_gres_detail_types(job.get("gres_detail"))
    ]
    if gres_detail_types:
        return gres_detail_types

    generic_gpu_count = sum(
        count for gpu_type, count in typed_entries if gpu_type == "generic"
    )
    if generic_gpu_count > 0:
        return [resolve_gpu_type("generic", node)]

    return []


def is_set_time(value):
    return bool(value and value.get("set"))


def format_epoch(value):
    if not is_set_time(value) or value.get("infinite"):
        return "n/a"

    epoch = int(value.get("number", 0) or 0)
    if epoch <= 0:
        return "n/a"

    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(total_seconds):
    if total_seconds < 0:
        total_seconds = 0

    days, remainder = divmod(int(total_seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_time_limit(value):
    if not is_set_time(value):
        return "n/a"
    if value.get("infinite"):
        return "UNLIMITED"

    minutes = int(value.get("number", 0) or 0)
    return format_duration(minutes * 60)


def format_time_left(job, now_epoch):
    time_limit = job.get("time_limit") or {}
    if time_limit.get("infinite"):
        return "UNLIMITED"

    end_time = job.get("end_time") or {}
    if is_set_time(end_time):
        end_epoch = int(end_time.get("number", 0) or 0)
        if end_epoch > 0:
            return format_duration(end_epoch - now_epoch)

    start_time = job.get("start_time") or {}
    if is_set_time(start_time) and is_set_time(time_limit):
        start_epoch = int(start_time.get("number", 0) or 0)
        limit_minutes = int(time_limit.get("number", 0) or 0)
        if start_epoch > 0 and limit_minutes > 0:
            return format_duration((start_epoch + limit_minutes * 60) - now_epoch)

    return "n/a"


def short_job_id(job_id):
    job_id = str(job_id or "")
    return job_id[-3:] if job_id else "n/a"


def load_squeue_time_left(partition):
    output = subprocess.check_output(
        [
            "squeue",
            "-h",
            "--states=RUNNING",
            "--partition",
            partition,
            "--format=%i|%L",
        ],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    time_left_by_job = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        job_id, time_left = line.split("|", 1)
        time_left_by_job[job_id.strip()] = time_left.strip() or "n/a"
    return time_left_by_job


jobs_by_node = defaultdict(list)
now_epoch = int(time.time())
time_left_by_job = load_squeue_time_left(partition)
payload = json.loads(
    subprocess.check_output(
        ["squeue", "--json", "--states=RUNNING", "--partition", partition],
        stderr=subprocess.DEVNULL,
        text=True,
    )
)

for job in payload.get("jobs", []):
    if (job.get("partition") or "") != partition:
        continue
    if "RUNNING" not in (job.get("job_state") or []):
        continue

    nodes = extract_nodes(job)
    if not nodes:
        continue

    start_time = format_epoch(job.get("start_time"))
    requested_time = format_time_limit(job.get("time_limit"))
    job_id = str(job.get("job_id", ""))
    remaining_time = time_left_by_job.get(job_id, format_time_left(job, now_epoch))
    job_id_short = short_job_id(job_id)

    for node in nodes:
        if requested_gpu_type not in gpu_types_for_job_on_node(job, node):
            continue

        jobs_by_node[node].append(
            {
                "job_id": job_id,
                "job_id_short": job_id_short,
                "start_time": start_time,
                "requested_time": requested_time,
                "remaining_time": remaining_time,
            }
        )

for node in sorted(jobs_by_node):
    node_jobs = sorted(jobs_by_node[node], key=lambda item: int(item["job_id"]))
    job_ids = ",".join(job["job_id_short"] for job in node_jobs)
    launch_times = ",".join(job["start_time"] for job in node_jobs)
    requested_times = ",".join(job["requested_time"] for job in node_jobs)
    remaining_times = ",".join(job["remaining_time"] for job in node_jobs)
    time_left_summary = ",".join(
        f'{job["job_id_short"]}:{job["remaining_time"]}'
        for job in node_jobs
    )
    print(
        "\t".join(
            [node, job_ids, launch_times, requested_times, remaining_times, time_left_summary]
        )
    )
PY
}

declare -A NODE_JOB_IDS=()
declare -A NODE_JOB_STARTS=()
declare -A NODE_JOB_REQS=()
declare -A NODE_JOB_LEFTS=()
declare -A NODE_JOB_TIME_LEFT_SUMMARY=()

nodes="$(sinfo -N -h -p "$PARTITION" -o "%N" | sort -u)"
available_count=0
matched_any=0

if ! partition_has_gpu_resources "$PARTITION"; then
  printf 'Partition: %s\n\n' "$PARTITION"

  if [[ -n "$REQUESTED_GPU_TYPE" ]]; then
    printf 'Error: partition "%s" does not expose GPU resources, so gpu_type "%s" cannot be matched.\n' \
      "$PARTITION" \
      "$REQUESTED_GPU_TYPE" >&2
    exit 1
  fi

  printf 'Partition "%s" does not expose GPU GRES; showing CPU/memory availability instead.\n\n' "$PARTITION"
  printf '%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n' \
    "Node" "State" "CPU_Tot" "CPU_Alloc" "CPU_Free" "Mem_Tot" "Mem_Alloc" "Mem_Free"

  while read -r node; do
    [[ -n "$node" ]] || continue

    line="$(scontrol show node -o "$node")"

    state="$(field_value "$line" State)"
    cpu_tot="$(field_value "$line" CPUTot)"
    cpu_alloc="$(field_value "$line" CPUAlloc)"
    real_mem="$(field_value "$line" RealMemory)"
    alloc_mem="$(field_value "$line" AllocMem)"

    cpu_tot="${cpu_tot:-0}"
    cpu_alloc="${cpu_alloc:-0}"
    real_mem="${real_mem:-0}"
    alloc_mem="${alloc_mem:-0}"

    cpu_free=$(( cpu_tot - cpu_alloc ))
    mem_free_mb=$(( real_mem - alloc_mem ))
    if (( cpu_free < 0 )); then cpu_free=0; fi
    if (( mem_free_mb < 0 )); then mem_free_mb=0; fi

    printf '%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n' \
      "$node" \
      "$state" \
      "$cpu_tot" \
      "$cpu_alloc" \
      "$cpu_free" \
      "${real_mem}MB" \
      "${alloc_mem}MB" \
      "${mem_free_mb}MB"
  done <<< "$nodes"

  printf '\nSchedulable nodes with free CPUs:\n'

  while read -r node; do
    [[ -n "$node" ]] || continue

    line="$(scontrol show node -o "$node")"

    state="$(field_value "$line" State)"
    cpu_tot="$(field_value "$line" CPUTot)"
    cpu_alloc="$(field_value "$line" CPUAlloc)"
    real_mem="$(field_value "$line" RealMemory)"
    alloc_mem="$(field_value "$line" AllocMem)"

    cpu_tot="${cpu_tot:-0}"
    cpu_alloc="${cpu_alloc:-0}"
    real_mem="${real_mem:-0}"
    alloc_mem="${alloc_mem:-0}"

    cpu_free=$(( cpu_tot - cpu_alloc ))
    mem_free_mb=$(( real_mem - alloc_mem ))
    if (( cpu_free < 0 )); then cpu_free=0; fi
    if (( mem_free_mb < 0 )); then mem_free_mb=0; fi

    if ! is_schedulable_state "$state"; then
      continue
    fi

    if (( cpu_free > 0 )); then
      printf '  - %s (cpu_free=%s/%s, mem_free=%sMB, state=%s)\n' \
        "$node" \
        "$cpu_free" \
        "$cpu_tot" \
        "$mem_free_mb" \
        "$state"
      available_count=$(( available_count + 1 ))
    fi
  done <<< "$nodes"

  if (( available_count == 0 )); then
    printf '  none\n'
  fi

  exit 0
fi

printf 'GPU partition: %s\n' "$PARTITION"
if [[ -n "$REQUESTED_GPU_TYPE" ]]; then
  printf 'Requested GPU type: %s\n' "$REQUESTED_GPU_TYPE"
  printf 'Note: timing columns reflect running jobs currently using GPUs of that type on each node.\n'
  if (( SHOW_DETAIL == 0 )); then
    printf 'Note: use --detail to also show job IDs, launch times, and requested times.\n'
  fi
else
  printf 'Requested GPU type: any\n'
  printf 'Note: without gpu_type, the script reports all GPU types in the partition.\n'
fi
printf '\n'

job_timing_notice=""
if [[ -n "$REQUESTED_GPU_TYPE" ]]; then
  if job_timing_output="$(build_requested_gpu_job_timing_map "$PARTITION" "$REQUESTED_GPU_TYPE")"; then
    while IFS=$'\t' read -r node job_ids launch_times requested_times remaining_times time_left_summary; do
      [[ -n "$node" ]] || continue
      NODE_JOB_IDS["$node"]="$job_ids"
      NODE_JOB_STARTS["$node"]="$launch_times"
      NODE_JOB_REQS["$node"]="$requested_times"
      NODE_JOB_LEFTS["$node"]="$remaining_times"
      NODE_JOB_TIME_LEFT_SUMMARY["$node"]="$time_left_summary"
    done <<< "$job_timing_output"
  else
    job_timing_notice="Warning: unable to query running job timing info from squeue; timing fields will show n/a."
  fi
fi

if [[ -n "$job_timing_notice" ]]; then
  printf '%s\n\n' "$job_timing_notice"
fi

if [[ -n "$REQUESTED_GPU_TYPE" ]]; then
  if (( SHOW_DETAIL == 1 )); then
    printf '%-10s %-10s %-18s %-8s %-9s %-9s %-10s %-10s %-10s | %-12s | %-24s | %-14s | %-18s\n' \
      "Node" "State" "GPU_Type" "GPU_Mem" "GPU_Tot" "GPU_Alloc" "GPU_Free" "CPU_Free" "Mem_Free" \
      "Job_IDs" "Launch_Time(s)" "Req_Time(s)" "Time_Left"
  else
    printf '%-10s %-10s %-18s %-8s %-9s %-9s %-10s %-10s %-10s | %-18s\n' \
      "Node" "State" "GPU_Type" "GPU_Mem" "GPU_Tot" "GPU_Alloc" "GPU_Free" "CPU_Free" "Mem_Free" \
      "Time_Left"
  fi
else
  printf '%-10s %-10s %-18s %-8s %-9s %-9s %-10s %-10s %-10s\n' \
    "Node" "State" "GPU_Type" "GPU_Mem" "GPU_Tot" "GPU_Alloc" "GPU_Free" "CPU_Free" "Mem_Free"
fi

while read -r node; do
  [[ -n "$node" ]] || continue

  line="$(scontrol show node -o "$node")"

  state="$(field_value "$line" State)"
  gres="$(field_value "$line" Gres)"
  alloc_tres="$(field_value "$line" AllocTRES)"
  cpu_tot="$(field_value "$line" CPUTot)"
  cpu_alloc="$(field_value "$line" CPUAlloc)"
  real_mem="$(field_value "$line" RealMemory)"
  alloc_mem="$(field_value "$line" AllocMem)"

  cpu_tot="${cpu_tot:-0}"
  cpu_alloc="${cpu_alloc:-0}"
  real_mem="${real_mem:-0}"
  alloc_mem="${alloc_mem:-0}"

  cpu_free=$(( cpu_tot - cpu_alloc ))
  mem_free_mb=$(( real_mem - alloc_mem ))
  if (( cpu_free < 0 )); then cpu_free=0; fi
  if (( mem_free_mb < 0 )); then mem_free_mb=0; fi
  mem_free_gb="$(mb_to_gb "$mem_free_mb")"

  while IFS='|' read -r raw_gpu_type gpu_total; do
    [[ -n "$raw_gpu_type" ]] || continue
    [[ -n "$gpu_total" ]] || gpu_total=0

    gpu_type="$raw_gpu_type"
    if [[ "${gpu_type,,}" == "tesla" || "${gpu_type,,}" == "generic" ]]; then
      fallback="$(fallback_gpu_type_for_node "$node")"
      if [[ -n "$fallback" ]]; then
        gpu_type="$fallback"
      fi
    fi

    if [[ -n "$REQUESTED_GPU_TYPE" && "$gpu_type" != "$REQUESTED_GPU_TYPE" ]]; then
      continue
    fi

    matched_any=1
    gpu_alloc=0

    while IFS='|' read -r alloc_type alloc_count; do
      [[ -n "$alloc_type" ]] || continue
      [[ -n "$alloc_count" ]] || alloc_count=0

      # Exact match
      if [[ "$alloc_type" == "$gpu_type" ]]; then
        gpu_alloc=$(( gpu_alloc + alloc_count ))
      fi
    done < <(parse_alloc_gpu_lines "${alloc_tres:-}")

    # If Slurm only exposes generic GPU allocation, use that as fallback.
    if (( gpu_alloc == 0 )); then
      while IFS='|' read -r alloc_type alloc_count; do
        [[ -n "$alloc_type" ]] || continue
        [[ -n "$alloc_count" ]] || alloc_count=0
        if [[ "$alloc_type" == "generic" ]]; then
          gpu_alloc=$(( gpu_alloc + alloc_count ))
        fi
      done < <(parse_alloc_gpu_lines "${alloc_tres:-}")
    fi

    gpu_free=$(( gpu_total - gpu_alloc ))
    if (( gpu_free < 0 )); then gpu_free=0; fi

    if [[ -n "$REQUESTED_GPU_TYPE" ]]; then
      if (( SHOW_DETAIL == 1 )); then
        printf '%-10s %-10s %-18s %-8s %-9s %-9s %-10s %-10s %-10s | %-12s | %-24s | %-14s | %-18s\n' \
          "$node" \
          "$state" \
          "$(pretty_gpu_type "$gpu_type")" \
          "$(gpu_mem_from_type "$gpu_type")" \
          "$gpu_total" \
          "$gpu_alloc" \
          "$gpu_free" \
          "$cpu_free" \
          "${mem_free_gb}GB" \
          "${NODE_JOB_IDS[$node]:-none}" \
          "${NODE_JOB_STARTS[$node]:-n/a}" \
          "${NODE_JOB_REQS[$node]:-n/a}" \
          "${NODE_JOB_LEFTS[$node]:-n/a}"
      else
        printf '%-10s %-10s %-18s %-8s %-9s %-9s %-10s %-10s %-10s | %-18s\n' \
          "$node" \
          "$state" \
          "$(pretty_gpu_type "$gpu_type")" \
          "$(gpu_mem_from_type "$gpu_type")" \
          "$gpu_total" \
          "$gpu_alloc" \
          "$gpu_free" \
          "$cpu_free" \
          "${mem_free_gb}GB" \
          "${NODE_JOB_TIME_LEFT_SUMMARY[$node]:-n/a}"
      fi
    else
      printf '%-10s %-10s %-18s %-8s %-9s %-9s %-10s %-10s %-10s\n' \
        "$node" \
        "$state" \
        "$(pretty_gpu_type "$gpu_type")" \
        "$(gpu_mem_from_type "$gpu_type")" \
        "$gpu_total" \
        "$gpu_alloc" \
        "$gpu_free" \
        "$cpu_free" \
        "${mem_free_gb}GB"
    fi

  done < <(parse_gres_gpu_lines "${gres:-}")
done <<< "$nodes"

printf '\n'
if [[ -n "$REQUESTED_GPU_TYPE" ]]; then
  printf 'Schedulable GPU nodes with free GPUs matching %s:\n' "$REQUESTED_GPU_TYPE"
else
  printf 'Schedulable GPU nodes with free GPUs:\n'
fi

while read -r node; do
  [[ -n "$node" ]] || continue

  line="$(scontrol show node -o "$node")"

  state="$(field_value "$line" State)"
  gres="$(field_value "$line" Gres)"
  alloc_tres="$(field_value "$line" AllocTRES)"
  real_mem="$(field_value "$line" RealMemory)"
  alloc_mem="$(field_value "$line" AllocMem)"
  cpu_tot="$(field_value "$line" CPUTot)"
  cpu_alloc="$(field_value "$line" CPUAlloc)"

  cpu_tot="${cpu_tot:-0}"
  cpu_alloc="${cpu_alloc:-0}"
  real_mem="${real_mem:-0}"
  alloc_mem="${alloc_mem:-0}"

  cpu_free=$(( cpu_tot - cpu_alloc ))
  mem_free_mb=$(( real_mem - alloc_mem ))
  if (( cpu_free < 0 )); then cpu_free=0; fi
  if (( mem_free_mb < 0 )); then mem_free_mb=0; fi
  mem_free_gb="$(mb_to_gb "$mem_free_mb")"

  if ! is_schedulable_state "$state"; then
    continue
  fi

  while IFS='|' read -r raw_gpu_type gpu_total; do
    [[ -n "$raw_gpu_type" ]] || continue
    [[ -n "$gpu_total" ]] || gpu_total=0

    gpu_type="$raw_gpu_type"
    if [[ "${gpu_type,,}" == "tesla" || "${gpu_type,,}" == "generic" ]]; then
      fallback="$(fallback_gpu_type_for_node "$node")"
      if [[ -n "$fallback" ]]; then
        gpu_type="$fallback"
      fi
    fi

    if [[ -n "$REQUESTED_GPU_TYPE" && "$gpu_type" != "$REQUESTED_GPU_TYPE" ]]; then
      continue
    fi

    gpu_alloc=0
    while IFS='|' read -r alloc_type alloc_count; do
      [[ -n "$alloc_type" ]] || continue
      [[ -n "$alloc_count" ]] || alloc_count=0
      if [[ "$alloc_type" == "$gpu_type" ]]; then
        gpu_alloc=$(( gpu_alloc + alloc_count ))
      fi
    done < <(parse_alloc_gpu_lines "${alloc_tres:-}")

    if (( gpu_alloc == 0 )); then
      while IFS='|' read -r alloc_type alloc_count; do
        [[ -n "$alloc_type" ]] || continue
        [[ -n "$alloc_count" ]] || alloc_count=0
        if [[ "$alloc_type" == "generic" ]]; then
          gpu_alloc=$(( gpu_alloc + alloc_count ))
        fi
      done < <(parse_alloc_gpu_lines "${alloc_tres:-}")
    fi

    gpu_free=$(( gpu_total - gpu_alloc ))
    if (( gpu_free < 0 )); then gpu_free=0; fi

    if (( gpu_free > 0 )); then
      if [[ -n "$REQUESTED_GPU_TYPE" ]]; then
        if (( SHOW_DETAIL == 1 )); then
          printf '  - %s (%s, free_gpus=%s/%s, cpu_free=%s, mem_free=%sGB, job_ids=%s, launch_time=%s, req_time=%s, time_left=%s)\n' \
            "$node" \
            "$(pretty_gpu_type "$gpu_type")" \
            "$gpu_free" \
            "$gpu_total" \
            "$cpu_free" \
            "$mem_free_gb" \
            "${NODE_JOB_IDS[$node]:-none}" \
            "${NODE_JOB_STARTS[$node]:-n/a}" \
            "${NODE_JOB_REQS[$node]:-n/a}" \
            "${NODE_JOB_LEFTS[$node]:-n/a}"
        else
          printf '  - %s (%s, free_gpus=%s/%s, cpu_free=%s, mem_free=%sGB, time_left=%s)\n' \
            "$node" \
            "$(pretty_gpu_type "$gpu_type")" \
            "$gpu_free" \
            "$gpu_total" \
            "$cpu_free" \
            "$mem_free_gb" \
            "${NODE_JOB_TIME_LEFT_SUMMARY[$node]:-n/a}"
        fi
      else
        printf '  - %s (%s, free_gpus=%s/%s, cpu_free=%s, mem_free=%sGB)\n' \
          "$node" \
          "$(pretty_gpu_type "$gpu_type")" \
          "$gpu_free" \
          "$gpu_total" \
          "$cpu_free" \
          "$mem_free_gb"
      fi
      available_count=$(( available_count + 1 ))
    fi
  done < <(parse_gres_gpu_lines "${gres:-}")
done <<< "$nodes"

if (( matched_any == 0 )); then
  printf '  no nodes found with GPU type "%s" in partition "%s"\n' "$REQUESTED_GPU_TYPE" "$PARTITION"
elif (( available_count == 0 )); then
  printf '  none\n'
fi
