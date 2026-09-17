# Historical C30/C31/C32/C49 compute rules

Archived on 2026-09-13 when the user approved NCSU as the active environment. These rules apply only when explicitly returning to that cluster; they are not the NCSU node allowlist.

## Compute Resources

- Approved machines are C30, C31, C32, and C49.
- For interactive access to C30, C31, or C32, use `salloc` to obtain the allocation instead of using a bare `srun` request. Use the node's registered partition, for example `salloc --partition=a6000 --nodelist=c31 --nodes=1 --ntasks=1 --cpus-per-task=16 --time=<duration>` for C30/C31 and `--partition=a5000ada --nodelist=c32` for C32.
- After an interactive `salloc` is granted, run node checks or manual commands inside it with `srun --jobid="${SLURM_JOB_ID}" --overlap ...`. Do not rely on direct SSH without an active allocation.
- Use `sbatch` for queued, unattended experiment jobs and dependency-ordered DAGs. Do not use a pending `salloc` request as the experiment queue or as a replacement for `sbatch`.
- Keep an interactive `salloc` allocation only while the interactive session is active; this rule does not replace the separate requirement to keep user-owned `grabgpu` allocations open.
- Keep `grabgpu` allocations open unless the user explicitly asks to close them.
- On C30, C31, C32, and C49, use memory-fit admission: a GPU is available whenever its remaining free memory is sufficient for the workload's expected peak allocation plus a safety margin. Do not require low utilization or near-zero used memory on any approved machine.
- `/home/youyang7/bin/gg` processes launched by the user's active `grabgpu` allocations are GPU keepalive/occupancy processes, not competing workloads. Ignore `gg` utilization when deciding availability and account for its memory only by subtracting it from the available capacity.
- Do not terminate approved-machine `gg` processes or close their allocations. Run project workloads inside an existing allocation with `srun --jobid=<allocation> --overlap` when applicable; continue to identify non-`gg` GPU processes before launch and include their memory in the free-capacity calculation without interfering with them.
- Inspect `nvidia-smi` before every launch. Never terminate or interfere with another process or allocation.
- Slurm `Gres=(null)` and `--exclusive` do not prove GPU isolation on these nodes; verify physical GPU occupancy directly.
- On oversubscribed partitions, use the shared GPU admission gate in `scripts/slurm/_gpu_idle_gate.sh`; select GPUs by sufficient remaining memory across repeated checks. The filename is retained for compatibility, but memory-fit is the default policy on every approved machine.
- Prefer one process per available GPU, disjoint problem shards, per-shard outputs, and an audited merge.
- Use model/tensor parallelism only when a model does not fit safely on one GPU.
- Use `srun --jobid=<allocation> --overlap` inside an existing `salloc` or `grabgpu` allocation rather than cancelling it.
- Do not request Slurm memory from the node's misleading `RealMemory` field; enforce process memory only when the workload requires it.
- Put large checkpoints and formal experiment artifacts on BeeGFS while retaining stable project paths.
- For factorial SFT/evaluation, keep Hugging Face dataset caches, temporary files, and Trainer intermediate checkpoints on node-local `/var/tmp`; publish only final hash-verified LoRA files and registered evidence to BeeGFS with bounded retries.

