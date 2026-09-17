# NCSU SFT environment

Installed on 2026-09-12 at
`/rsstu/users/m/myoon2/satc_568382/yang_ouyang/envs/sft` using Python 3.10.21.
All 72 third-party pins from the user-provided C31 freeze match
`configs/0_1_ncsu_sft_requirements.txt`. The project package
`length-budget-distill==0.1.0` is installed in editable mode from this repository.

Activate in an interactive terminal:

```bash
source /usr/local/apps/conda/miniconda3/26.3.2/etc/profile.d/conda.sh
conda activate sft
```

Recreate with the configured shared-storage Conda environment and package directories:

```bash
conda create -y -n sft --override-channels -c conda-forge python=3.10.21 pip
conda activate sft
python -m pip install -r configs/0_1_ncsu_sft_requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

The existing Hugging Face and pip caches remain under
`/rsstu/users/m/myoon2/satc_568382/yang_ouyang/.cache` and the Conda package cache
under that user's `.conda/pkgs` directory. Historical experiment launchers may
still reference the original cluster's environments and need explicit NCSU
configuration before use. This installation does not add TRL or vLLM.

Validation artifacts are in `results/ncsu_sft_setup/smoke_20260912/`: the version
audit, pip dependency check, complete pip freeze, Conda environment export,
installation logs/report, and CUDA smoke-check output. Slurm job 811390 completed
on gpu17 (H100) with exit code 0. It verified key library imports and a tiny random
Qwen2 BF16 LoRA forward pass, backward pass with finite nonzero gradients, and
optimizer step. This is an environment smoke check, not formal experiment evidence.

## GSM8K pilot TokenSkip C++ runtime compatibility

On 2026-09-15, TokenSkip jobs 840409, 840478, 840480, and 840483 failed
during the NLTK / SQLite import chain. The node's `/lib64/libstdc++.so.6`
lacked `CXXABI_1.3.15`, required by the environment's `libicui18n.so.78`.
Retrying the same environment did not resolve this deterministic import error.

The pilot's frozen `execution_v6` sets `runtime.native_preload` to the existing
SFT environment's `lib/libstdc++.so.6`. The launcher
`scripts/slurm/13_6_run_frozen_python.sh` reads this setting and exports
`LD_PRELOAD` before starting the worker Python process. This selects the
compatible runtime before Torch and the compression dependencies are imported.
The resolved library is `libstdc++.so.6.0.36`; its hash and the `libgcc_s.so.1`
hash are recorded in the frozen protocol. No package upgrade is needed.
Keep this setting scoped to the registered pilot launcher; preserve legacy
overlays and historical execution snapshots.

Compute job 842428 passed 51 tests, including the Torch → SQLite → NLTK →
LLMLingua import regression. H200 job 842429 completed all four TokenSkip
smoke records. The recovery controller treats `ImportError` as a deterministic
failure requiring diagnosis rather than automatic GPU retries. These checks
validate the runtime repair, not completion of the student experiment.

Evidence: [frozen runtime bindings](../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v6/protocol/FROZEN.json),
[GPU smoke verification](../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v6/verification/NATIVE_FIX_VERIFIED.json),
and [2026-09-15 independent hash recheck](../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v6/verification/recheck_20260915T140728Z/hash_audit.json).

## Administrator-designated scratch and GSM pilot resumption

The project `tmp` symlink points to `/share/jekml/youyang7/tmp`. Each new
GSM pilot job uses `youyang7-phase13-frozen-<job-id>` beneath that directory.
`execution_v7` preserves the ABI preload and changes the frozen launcher,
Python storage admission, and explicit Trainer output path together. Temporary
files, dataset caches, compilation caches, and intermediate adapters use this
scratch; final adapters are copied to project checkpoints with hash verification.
Existing model caches are reused. Neither system temporary directories nor
project `data/runtime` are fallback locations.

Admission verifies TMPDIR/TMP/TEMP consistency, the real path, workload-specific
capacity, and GPFS Share01 user/group quotas. Failure stops the job. The Slurm
spooled batch script is read back separately from the working-tree launcher.
Historical snapshots and held MATH jobs retain their prior state and are not
migrated by the GSM pilot revision.

The same revision fixes the SFT answer-region diagnostic for patterns without a
named `marker` group. Named historical patterns retain their original boundary
semantics. Completed common-support evidence is verified and reused; partial
B0 files from job 842496 are archived under `execution_v7/preserved_failed_842496`.
See [revision and bindings](../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v7/protocol/FROZEN.json).
