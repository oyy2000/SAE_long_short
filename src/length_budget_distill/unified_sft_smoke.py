"""Synthetic-only capacity checks for the unified baseline student interface."""
from pathlib import Path
import importlib.metadata
import logging
import os
import shutil
import time

from .experiment_io import read_json, publish_files_hash_verified
from .records import read_jsonl, write_jsonl
from .factorial import file_sha256
from .ncsu_reproduction import save, seal, verify, evidence, admission
from .baseline_reproduction import record_hardware
from .compression_baselines import tokenskip_prompt
from .completion_supervision import encode_completion, TrainingExposureAudit

CODE = Path(__file__).resolve().parents[2]


def synthetic_rows(tokenizer, cfg):
    rows = []
    for index, limit in enumerate(cfg["smoke"]["sequence_limits"]):
        ratio = cfg["smoke"]["ratios"][index % len(cfg["smoke"]["ratios"])]
        question = f"Synthetic interface case {index}: start with 1 and repeatedly add 0. What value remains?"
        prompt = tokenskip_prompt(question, ratio) if index % 2 else cfg["question_template"].format(question=question)
        base = {"problem_id": f"synthetic-sft-{index:03d}", "question_role":"synthetic_interface_only",
                "prompt":prompt, "ratio":ratio if index % 2 else None,
                "prompt_family":"TokenSkip" if index % 2 else "common_math"}
        # Search a synthetic repetition count; no real trace is shortened to fit.
        low, high = 0, limit
        while low < high:
            repetitions = (low + high + 1)//2
            completion = "1 + 0 = 1.\n"*repetitions + "The value remains \\boxed{1}."
            try: encode_completion(tokenizer, {**base,"completion":completion}, max_length=limit)
            except ValueError: high = repetitions-1
            else: low = repetitions
        row = encode_completion(tokenizer, {**base, "completion":"1 + 0 = 1.\n"*low + "The value remains \\boxed{1}."}, max_length=limit)
        row.update(requested_sequence_limit=limit, synthetic_repetitions=low)
        if limit-len(row["input_ids"]) > 16:
            raise ValueError("Synthetic stress input does not exercise its requested sequence boundary")
        rows.append(row)
    return rows


def prepare(config_path):
    from transformers import AutoTokenizer
    cfg = read_json(config_path); root = Path(cfg["result_root"])
    if root.exists(): raise FileExistsError(root)
    if cfg["evidence_class"] != "synthetic_interface_smoke": raise ValueError("Synthetic preparation only")
    root.mkdir(parents=True)
    inventories = {}; audits = {}
    for name, model in cfg["students"].items():
        path = Path(model["snapshot_path"])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        rows = synthetic_rows(tokenizer, cfg)
        write_jsonl(root/"inputs"/(name+".jsonl"), rows)
        inventories[name] = evidence([p for p in sorted(path.iterdir()) if p.is_file()])
        audits[name] = {"records":len(rows), "sequence_tokens":[len(r["input_ids"]) for r in rows],
                       "supervision_tokens":[r["supervision_tokens"] for r in rows],
                       "development_or_calibration_records":0, "token_truncations":0}
    save(root/"inputs/model_hashes.json", inventories)
    save(root/"inputs/data_audit.json", audits)
    save(root/"inputs/versions.json", {name:importlib.metadata.version(name) for name in
        ("torch", "transformers", "trl", "peft", "datasets", "accelerate")})
    for folder in ("src", "scripts", "configs", "tests"):
        shutil.copytree(CODE/folder, root/"code"/folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"), symlinks=True)
    cfg["code_root"] = str(root/"code")
    seal(root/"protocol/SOURCES.json", [p for p in (root/"code").rglob("*") if p.is_file() and not p.is_symlink()])
    save(root/"protocol/frozen_config.json", cfg)
    seal(root/"protocol/FROZEN.json", [Path(config_path), root/"protocol/frozen_config.json",
        root/"protocol/SOURCES.json", *sorted((root/"inputs").glob("*"))], student_training_data_ready=False)
    logging.info("Synthetic SFT inputs frozen: %s", audits)


def smoke(config_path, name):
    import torch
    from transformers import set_seed
    from safetensors.torch import load_file
    from .training import run_trl_sft
    cfg = read_json(config_path); root = Path(cfg["result_root"])
    verify(root/"protocol/FROZEN.json"); verify(root/"protocol/SOURCES.json")
    if CODE != Path(cfg["code_root"]): raise ValueError("Use the frozen SFT source")
    if cfg["evidence_class"] != "synthetic_interface_smoke": raise ValueError("Synthetic SFT only")
    out = root/"training"/name; adapter = Path(cfg["checkpoint_root"])/name
    if out.exists() or adapter.exists(): raise FileExistsError("Preserve the existing SFT attempt")
    for path, digest in read_json(root/"inputs/model_hashes.json")[name].items():
        if file_sha256(path) != digest: raise ValueError("Student model/tokenizer changed: "+path)
    rows = list(read_jsonl(root/"inputs"/(name+".jsonl")))
    if any(r["question_role"] != "synthetic_interface_only" for r in rows):
        raise ValueError("Real development/calibration records cannot train a smoke adapter")
    admission(cfg); out.mkdir(parents=True); record_hardware(out)
    seed = cfg["seed"]; local = Path(os.environ["TMPDIR"])/"unified_sft"/name
    model = cfg["students"][name]
    student = {"model_name":model["snapshot_path"], "torch_dtype":"bfloat16", "use_lora":True,
               "lora":cfg["lora"], "tokenizer_kwargs":{"local_files_only":True,"padding_side":"right"}}
    training = {**cfg["training"], "seed":seed, "data_seed":seed, "output_dir":str(adapter),
                "max_steps":cfg["smoke"]["optimizer_steps"], "warmup_ratio":0., "warmup_steps":0,
                "save_strategy":"no", "model_init_kwargs":{"local_files_only":True,"attn_implementation":"sdpa"}}
    run = {"student":student,"training":training,
           "data":{"train_path":str(root/"inputs"/(name+".jsonl")),"text_format":"pretokenized_completion"}}
    save(out/"run_config.json", run)
    meter = TrainingExposureAudit(rows, max_length=training["max_length"])
    os.environ["LBD_RUNTIME_OUTPUT_DIR"] = str(local); set_seed(seed)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.monotonic()
    trainer = run_trl_sft(run, before_train=meter.install)
    torch.cuda.synchronize(); seconds = time.monotonic()-start
    audit = meter.summary()
    if trainer.state.global_step != cfg["smoke"]["optimizer_steps"] or set(audit["problem_exposures"].values()) != {1}:
        raise ValueError("Synthetic smoke did not finish two updates and exactly one exposure per stress record")
    weights = load_file(str(local/"adapter_model.safetensors"))
    b_values = [v for k,v in weights.items() if "lora_B" in k]
    if not b_values or not all(torch.isfinite(v).all() for v in weights.values()) or not any(torch.count_nonzero(v) for v in b_values):
        raise ValueError("Missing, nonfinite, or unchanged LoRA adapter")
    metrics = {**audit, "optimizer_steps":trainer.state.global_step, "actual_epoch":trainer.state.epoch,
               "elapsed_seconds_including_batch_audit":seconds,
               "peak_gpu_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
               "peak_gpu_reserved_mib":torch.cuda.max_memory_reserved()/2**20,
               "log_history":trainer.state.log_history,"synthetic_only":True,
               "largest_input_tokens":max(len(r["input_ids"]) for r in rows),
               "nonzero_lora_b_tensors":sum(bool(torch.count_nonzero(v)) for v in b_values),
               "formal_training_complete":False, "student_quality_claim_allowed":False}
    save(local/"training_metrics.json", metrics)
    publish_files_hash_verified(local, adapter, ("adapter_model.safetensors","adapter_config.json","training_metrics.json"))
    bindings = [root/"protocol/FROZEN.json", root/"protocol/SOURCES.json", root/"inputs"/(name+".jsonl"),
                out/"run_config.json", out/"hardware.json", *sorted(adapter.glob("*"))]
    seal(adapter/"TRAIN_COMPLETE.json", bindings, synthetic_only=True, formal_training_complete=False)
    seal(out/"COMPLETE.json", [adapter/"TRAIN_COMPLETE.json", adapter/"training_metrics.json"], synthetic_only=True)
    logging.info("Synthetic unified SFT complete: %s %s", name, metrics)
