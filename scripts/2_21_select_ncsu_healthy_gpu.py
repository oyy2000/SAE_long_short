#!/usr/bin/env python3
"""Select a healthy device strictly within an existing Slurm GPU allocation."""
import argparse
import json
import os
import sys
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-uuid", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("An active Slurm allocation is required")
    uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
    def normalize(value):
        return value.lower().removeprefix("gpu-").replace("-", "")
    candidates = [u for u in uuids if normalize(u) != normalize(args.exclude_uuid)]
    if len(uuids) != 2 or len(candidates) != 1:
        raise RuntimeError(f"Expected the allocated pair including excluded device; observed {uuids}")
    selected = candidates[0]
    if not selected.startswith("GPU-"):
        selected = "GPU-" + selected
    print(json.dumps({"job_id": os.environ["SLURM_JOB_ID"], "allocated_device_uuids": uuids,
                      "excluded_uuid": args.exclude_uuid, "selected_uuid": selected}), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("An explicit child command is required")
    os.execvp(command[0], command)
