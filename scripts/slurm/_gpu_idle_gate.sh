#!/bin/bash

# Shared physical-GPU admission guard. Slurm GRES is not authoritative on the
# approved nodes, so callers use repeated nvidia-smi observations and never
# interfere with already-running processes. The historical filename is kept
# for compatibility; memory-fit is the default on every approved machine.

mkdir_with_retry() {
  local target="${1:?directory path is required}"
  local attempts="${2:-10}"
  local wait_seconds="${3:-10}"
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if mkdir -p "${target}"; then
      return 0
    fi
    echo "Directory creation attempt ${attempt}/${attempts} failed for ${target}; retrying." >&2
    sleep "${wait_seconds}"
  done
  return 1
}

select_stably_idle_gpus() {
  local min_free_mib="${1:?minimum free memory is required}"
  local min_idle_gpus="${2:-1}"
  local wait_seconds="${3:-30}"
  local max_used_mib="${4:-500}"
  local max_utilization="${5:-10}"
  local stable_checks="${6:-2}"
  local stable_count=0 previous_signature="" signature=""
  local gpu_id used_mib free_mib utilization
  local -a all_gpu_ids=() idle_gpu_ids=()
  mapfile -t all_gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  if [ "${#all_gpu_ids[@]}" -eq 0 ]; then
    echo "No GPUs detected on $(hostname)." >&2
    return 1
  fi
  while true; do
    idle_gpu_ids=()
    for gpu_id in "${all_gpu_ids[@]}"; do
      IFS=',' read -r used_mib free_mib utilization < <(
        nvidia-smi -i "${gpu_id}" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
      )
      used_mib="${used_mib//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      utilization="${utilization//[[:space:]]/}"
      if [ "${free_mib}" -ge "${min_free_mib}" ] \
        && [ "${used_mib}" -le "${max_used_mib}" ] \
        && [ "${utilization}" -le "${max_utilization}" ]; then
        idle_gpu_ids+=("${gpu_id}")
      fi
    done
    signature="${idle_gpu_ids[*]}"
    if [ "${#idle_gpu_ids[@]}" -ge "${min_idle_gpus}" ]; then
      if [ "${signature}" = "${previous_signature}" ]; then
        stable_count=$((stable_count + 1))
      else
        stable_count=1
      fi
      previous_signature="${signature}"
      if [ "${stable_count}" -ge "${stable_checks}" ]; then
        GPU_IDS=("${idle_gpu_ids[@]}")
        echo "Selected stable idle GPUs on $(hostname): ${GPU_IDS[*]}"
        return 0
      fi
    else
      stable_count=0
      previous_signature=""
    fi
    echo "Waiting without interference: idle=${#idle_gpu_ids[@]}/${#all_gpu_ids[@]}, required>=${min_idle_gpus}, stable_checks=${stable_count}/${stable_checks}."
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
    sleep "${wait_seconds}"
  done
}

select_stably_memory_fit_gpus() {
  local min_free_mib="${1:?minimum free memory is required}"
  local min_gpus="${2:-1}"
  local wait_seconds="${3:-30}"
  local stable_checks="${4:-2}"
  local stable_count=0 previous_signature="" signature="" gpu_id free_mib
  local -a eligible_gpu_ids=()
  while true; do
    eligible_gpu_ids=()
    while IFS=',' read -r gpu_id free_mib; do
      gpu_id="${gpu_id//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      if [ "${free_mib}" -ge "${min_free_mib}" ]; then
        eligible_gpu_ids+=("${gpu_id}")
      fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    signature="${eligible_gpu_ids[*]}"
    if [ "${#eligible_gpu_ids[@]}" -ge "${min_gpus}" ]; then
      if [ "${signature}" = "${previous_signature}" ]; then
        stable_count=$((stable_count + 1))
      else
        stable_count=1
      fi
      previous_signature="${signature}"
      if [ "${stable_count}" -ge "${stable_checks}" ]; then
        mapfile -t GPU_IDS < <(
          for gpu_id in "${eligible_gpu_ids[@]}"; do
            free_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
            printf '%s %s\n' "${free_mib}" "${gpu_id}"
          done | sort -k1,1nr -k2,2n | awk '{print $2}'
        )
        echo "Selected stable memory-fit GPUs by descending free memory on $(hostname): ${GPU_IDS[*]}"
        return 0
      fi
    else
      stable_count=0
      previous_signature=""
    fi
    echo "Waiting for memory capacity without interference: eligible=${#eligible_gpu_ids[@]}/${min_gpus}."
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
    sleep "${wait_seconds}"
  done
}

select_gpus_for_approved_node() {
  local min_free_mib="${1:?minimum free memory is required}"
  local min_gpus="${2:-1}"
  local wait_seconds="${3:-30}"
  local stable_checks="${4:-2}"
  local max_used_mib="${5:-500}"
  local max_utilization="${6:-10}"
  local host admission_policy="${GPU_ADMISSION_POLICY:-memory_fit}"
  host="$(hostname -s | tr '[:upper:]' '[:lower:]')"
  case "${host}" in c30|c31|c32|c49) ;; *) echo "GPU admission is not registered for node ${host}." >&2; return 2 ;; esac
  case "${admission_policy}" in
    auto)
      select_stably_memory_fit_gpus "${min_free_mib}" "${min_gpus}" "${wait_seconds}" "${stable_checks}"
      ;;
    memory_fit) select_stably_memory_fit_gpus "${min_free_mib}" "${min_gpus}" "${wait_seconds}" "${stable_checks}" ;;
    idle) select_stably_idle_gpus "${min_free_mib}" "${min_gpus}" "${wait_seconds}" "${max_used_mib}" "${max_utilization}" "${stable_checks}" ;;
    *) echo "Unknown GPU_ADMISSION_POLICY=${admission_policy}." >&2; return 2 ;;
  esac
}
