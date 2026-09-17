#!/usr/bin/env python3
"""Replay a frozen extraction stage with explicitly selected math SDPA kernels."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from length_budget_distill.ncsu_reproduction import admission, verify, seal
from length_budget_distill.factorial import file_sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["result_root"])
    verify(root / "protocol/FROZEN.json")
    for filename, digest in json.loads((root / "protocol/sources.json").read_text()).items():
        if file_sha256(filename) != digest:
            raise ValueError(f"Frozen source changed: {filename}")
    admission(cfg)
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("Extraction runtime correction: math SDPA only; frozen model, corpus, batching and source unchanged", flush=True)
    count = cfg["teacher"]["generation_shards"]
    output = root / f"sae/activations/shard_{args.shard_index:02d}_of_{count:02d}"
    entry = Path(cfg["code_root"]) / "scripts/2_2_extract_residual_activations.py"
    wrapper_path = Path(__file__).resolve()
    sys.argv = [str(entry), "--config", str(root / "sae/protocol/frozen_protocol.json"),
                "--corpus", str(root / "sae/corpus/mixed_trajectories.jsonl"),
                "--shard-index", str(args.shard_index), "--shard-count", str(count),
                "--output-dir", str(output)]
    runpy.run_path(str(entry), run_name="__main__")
    seal(output / "RUNTIME_BACKEND.json", [wrapper_path, output / "activation_manifest.json",
         output / "ACTIVATIONS_COMPLETE"], sdpa_backend="math", job_id=os.getenv("SLURM_JOB_ID"),
         reason="Explicit kernel fallback after CUDA illegal access and stalled extraction")
