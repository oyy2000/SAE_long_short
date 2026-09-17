"""NCSU orchestration reusing the registered SFT, evaluation, and SAE implementations."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

from .experiment_io import read_json, write_json_exclusive, publish_files_hash_verified
from .factorial import file_sha256, canonical_sha256
from .records import read_jsonl, write_jsonl
from .ranked_sampling import build_length_agnostic_teacher_prompt
from .student_prompts import build_student_math_prompt
from .verifiers import extract_final_answer, verify_answer

CODE = Path(__file__).resolve().parents[2]


def resolve(p):
    value = Path(p)
    return value if value.is_absolute() else CODE / value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(path, value)


def evidence(paths):
    return {str(Path(p).resolve()): file_sha256(p) for p in paths}


def seal(path, paths, **extra):
    save(path, {"status": "complete", "hashes": evidence(paths), **extra})


def verify(path):
    doc = read_json(path)
    if doc["status"] != "complete":
        raise ValueError(f"Incomplete marker: {path}")
    for name, digest in doc["hashes"].items():
        if file_sha256(name) != digest:
            raise ValueError(f"Changed artifact: {name}")
    return doc


def run_script(name, **kwargs):
    cmd = [sys.executable, str(CODE / "scripts" / name)]
    for key, value in kwargs.items():
        cmd.extend(["--" + key.replace("_", "-"), str(value)])
    logging.info("Running %s", cmd)
    subprocess.run(cmd, check=True)


def gsm8k_question_record(index, row):
    """Keep official rationale provenance but pass only the final numeric target to verifiers."""
    answer = extract_final_answer(row["answer"])
    if answer is None:
        raise ValueError(f"Official GSM8K answer has no final-answer marker: {index}")
    return {"problem_id": f"hf-{index:06d}", "source_index": index, "question": row["question"],
            "answer": answer, "raw_answer": row["answer"],
            "prompt": build_length_agnostic_teacher_prompt(row["question"]),
            "student_prompt": build_student_math_prompt(row["question"])}


def prepare(config_path):
    cfg = read_json(config_path)
    root = resolve(cfg["result_root"])
    protocol = root / "protocol"
    if protocol.exists():
        raise FileExistsError(protocol)
    cfg["result_root"] = str(root)
    cfg["checkpoint_root"] = str(resolve(cfg["checkpoint_root"]))
    cfg["project_root"] = str(CODE)
    # Read only the requested complete member; the rest of the transferred archive
    # is not used or asserted to be complete. All question contents are independently
    # checked against the public GSM8K dataset below.
    question_bytes = subprocess.check_output([
        "tar", "--zstd", "--occurrence=1", "-xOf", str(resolve(cfg["question_archive"])),
        cfg["question_archive_member"],
    ])
    imported = [json.loads(line) for line in question_bytes.splitlines() if line.strip()]
    if len(imported) != cfg["question_count"]:
        raise ValueError("Unexpected question count")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(cfg["dataset"]).sha
    dataset = load_dataset(cfg["dataset"], cfg["dataset_config"], revision=revision)
    questions = []
    for row in imported:
        index = int(row["problem_id"].removeprefix("hf-"))
        original = dataset["train"][index]
        if row["prompt"] != build_length_agnostic_teacher_prompt(original["question"]):
            raise ValueError(f"Training question mismatch: {index}")
        if row["gold_answer"] != original["answer"]:
            raise ValueError(f"Gold answer mismatch: {index}")
        questions.append(gsm8k_question_record(index, original))
    questions.sort(key=lambda row: row["source_index"])
    if len({r["problem_id"] for r in questions}) != len(questions):
        raise ValueError("Duplicate training questions")
    evaluation = [gsm8k_question_record(i, row) for i, row in enumerate(dataset["test"]) if 50 <= i < 1319]
    if len(evaluation) != cfg["evaluation"]["count"]:
        raise ValueError("Evaluation count mismatch")
    if {r["question"] for r in questions} & {r["question"] for r in evaluation}:
        raise ValueError("Training/evaluation question overlap")
    write_jsonl(root / "inputs/questions.jsonl", questions)
    write_jsonl(root / "inputs/evaluation.jsonl", evaluation)
    save(root / "inputs/smoke_evaluation.json", {"questions": [gsm8k_question_record(i, dataset["test"][i]) for i in range(2)]})
    cfg["dataset_revision"] = revision
    cfg["question_member_sha256"] = hashlib.sha256(question_bytes).hexdigest()
    paths = read_json(root / "setup/model_paths.json")
    model_evidence = {}
    for role in ("teacher", "student"):
        snapshot = Path(paths[cfg[role]["model_name"]])
        cfg[role]["snapshot_path"] = str(snapshot)
        files = sorted(snapshot.glob("*.safetensors")) + sorted(snapshot.glob("*.json"))
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError(f"Missing model weights: {role}")
        model_evidence[role] = evidence(files)
    save(root / "inputs/model_hashes.json", model_evidence)
    frozen_code = root / "code"
    for name in ("src", "scripts", "configs"):
        shutil.copytree(CODE / name, frozen_code / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
                        symlinks=True)
    cfg["code_root"] = str(frozen_code)
    cfg["sae_template"] = str(frozen_code / cfg["sae_template"])
    cfg["feature_template"] = str(frozen_code / cfg["feature_template"])
    source_files = [p for dirname in ("src", "scripts", "configs")
                    for p in (frozen_code / dirname).rglob("*")
                    if p.is_file() and not p.is_symlink()]
    save(protocol / "sources.json", evidence(source_files))
    save(protocol / "frozen_protocol.json", cfg)
    seal(protocol / "FROZEN.json", [protocol / "frozen_protocol.json", protocol / "sources.json",
         root / "inputs/questions.jsonl", root / "inputs/evaluation.jsonl", root / "inputs/model_hashes.json",
         root / "inputs/smoke_evaluation.json"],
         source_config=str(config_path), source_config_sha256=file_sha256(config_path))
    logging.info("Frozen %s", protocol / "frozen_protocol.json")


def admission(cfg):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one Slurm-assigned visible GPU")
    # Repeated checks are confined to the assigned device, never another user's GPU.
    import time
    for _ in range(2):
        free, total = torch.cuda.mem_get_info(0)
        logging.info("Assigned GPU free_mib=%d total_mib=%d", free // 2**20, total // 2**20)
        if free < cfg["runtime"]["minimum_free_mib"] * 2**20:
            raise RuntimeError("Assigned GPU lacks the registered memory margin")
        time.sleep(1)


def isolated_gpu_preflight(config_path, output_dir, expected_name=None):
    """Release the probe CUDA context before launching a separate GPU worker.

    Exclusive-process GPUs permit only one context-owning process. Orchestrators
    using this helper must not initialize CUDA themselves before their worker.
    """
    code = (
        "import sys, torch; from pathlib import Path; "
        "from length_budget_distill.experiment_io import read_json; "
        "from length_budget_distill.ncsu_reproduction import admission; "
        "from length_budget_distill.baseline_reproduction import record_hardware; "
        "cfg=read_json(sys.argv[1]); admission(cfg); "
        "record_hardware(Path(sys.argv[2])); "
        "assert not sys.argv[3] or sys.argv[3] in torch.cuda.get_device_name(0), 'Unexpected GPU type'"
    )
    subprocess.run([sys.executable, '-c', code, str(config_path), str(output_dir), expected_name or ''], check=True)


def teacher_bundle(cfg):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    snapshot = cfg["teacher"]["snapshot_path"]
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    return model, tok


def sample_question(cfg, bundle, question):
    import torch
    from transformers import set_seed
    model, tok = bundle
    spec = cfg["teacher"]
    seed = int.from_bytes(hashlib.sha256(f"{spec['base_seed']}:{question['problem_id']}".encode()).digest()[:4], "big")
    set_seed(seed)
    rendered = tok.apply_chat_template([{"role": "user", "content": question["prompt"]}],
                                      tokenize=False, add_generation_prompt=True)
    inputs = tok([rendered] * spec["num_rollouts"], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=True, temperature=spec["temperature"],
              top_p=spec["top_p"], max_new_tokens=spec["max_new_tokens"],
              eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    rows = []
    for candidate, suffix in enumerate(out[:, inputs["input_ids"].shape[1]:]):
        tokens = suffix.tolist()
        eos = tokens.index(tok.eos_token_id) if tok.eos_token_id in tokens else len(tokens)
        text = tok.decode(tokens[:eos], skip_special_tokens=True).strip()
        prediction = extract_final_answer(text)
        rows.append({**question, "trace_id": f"fresh-{question['problem_id']}-{candidate:02d}",
                     "candidate_index": candidate, "seed": seed, "solution": text,
                     "solution_token_count": len(tok.encode(text, add_special_tokens=False)),
                     "predicted_answer": prediction, "gold_answer": question["answer"],
                     "is_correct": verify_answer(prediction, question["answer"]),
                     "hit_max_new_tokens": eos == len(tokens) and len(tokens) >= spec["max_new_tokens"]})
    return rows


def generate(cfg, index):
    root = Path(cfg["result_root"])
    count = cfg["teacher"]["generation_shards"]
    if not 0 <= index < count:
        raise ValueError(index)
    out = root / f"generation/shard_{index:02d}.jsonl"
    marker = out.with_suffix(".complete.json")
    if marker.exists():
        verify(marker)
        return
    questions = list(read_jsonl(root / "inputs/questions.jsonl"))[index::count]
    partial = out.with_suffix(".partial.jsonl")
    # Resume only whole, audited question groups; an interrupted JSON write is not
    # silently repaired or accepted as completed evidence.
    existing = list(read_jsonl(partial)) if partial.exists() else []
    seen = Counter(r["problem_id"] for r in existing)
    if any(v != cfg["teacher"]["num_rollouts"] for v in seen.values()):
        raise ValueError("Partial question group requires explicit recovery")
    if not set(seen) <= {r["problem_id"] for r in questions}:
        raise ValueError("Partial shard has foreign questions")
    bundle = teacher_bundle(cfg)
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a") as handle:
        for i, q in enumerate(questions):
            if q["problem_id"] in seen:
                continue
            rows = sample_question(cfg, bundle, q)
            handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            handle.flush()
            os.fsync(handle.fileno())
            logging.info("generation shard=%d question=%d/%d correct=%d/%d", index, i+1,
                         len(questions), sum(r["is_correct"] for r in rows), len(rows))
    rows = list(read_jsonl(partial))
    expected = {(q["problem_id"], c) for q in questions for c in range(cfg["teacher"]["num_rollouts"])}
    if len(rows) != len(expected) or {(r["problem_id"], r["candidate_index"]) for r in rows} != expected:
        raise ValueError("Shard missing/duplicate record audit failed")
    partial.rename(out)
    seal(marker, [out, root / "protocol/frozen_protocol.json", root / "inputs/model_hashes.json"],
         records=len(rows), shard=index, job_id=os.getenv("SLURM_JOB_ID"))


def merge(cfg):
    from trace_length_observation.trace_observation import select_quantile_band_candidates
    root = Path(cfg["result_root"])
    all_rows = []
    paths = []
    for i in range(cfg["teacher"]["generation_shards"]):
        p = root / f"generation/shard_{i:02d}.jsonl"
        verify(p.with_suffix(".complete.json"))
        paths.append(p)
        all_rows.extend(read_jsonl(p))
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["problem_id"]].append(row)
    questions = list(read_jsonl(root / "inputs/questions.jsonl"))
    expected = {(q["problem_id"], c) for q in questions for c in range(cfg["teacher"]["num_rollouts"])}
    if len(all_rows) != len(expected) or {(r["problem_id"], r["candidate_index"]) for r in all_rows} != expected:
        raise ValueError("Global candidate audit failed")
    selected, excluded = {}, []
    for pid, rows in sorted(grouped.items()):
        eligible = [r for r in rows if not r["hit_max_new_tokens"]]
        choices = select_quantile_band_candidates(eligible,
            minimum_unique_correct=cfg["selection"]["minimum_unique_correct"],
            tail_fraction=cfg["selection"]["tail_fraction"])
        if choices and all(r["solution_token_count"] > 0 for r in rows):
            selected[pid] = choices
        else:
            excluded.append(pid)
    if len(selected) < 20:
        raise ValueError("Insufficient common problem support")
    for rank in cfg["ranks"]:
        rows = []
        for pid, choices in selected.items():
            row = choices[rank]
            rows.append({"prompt": row["student_prompt"], "completion": row["solution"],
                         "metadata": {"problem_id": pid, "trace_id": row["trace_id"],
                                      "solution_token_count": row["solution_token_count"], "rank": rank}})
        write_jsonl(root / f"sft_data/{rank}.jsonl", rows)
    sae_rows = [r for r in all_rows if r["problem_id"] in selected]
    raw = root / "sae/raw/eligible_candidates.jsonl"
    write_jsonl(raw, sorted(sae_rows, key=lambda r: (r["problem_id"], r["candidate_index"])))
    audit_path = root / "generation/generation_audit.json"
    save(audit_path, {"status": "passed", "records": len(all_rows), "expected_records": len(expected),
         "shards": evidence(paths), "missing_records": 0, "duplicate_records": 0,
         "raw_correct_count": sum(r["is_correct"] for r in all_rows),
         "cap_hit_count": sum(r["hit_max_new_tokens"] for r in all_rows),
         "eligible_questions": len(selected), "excluded_questions": excluded,
         "selection": cfg["selection"], "fresh_generation": True})
    n = len(selected)
    sae = read_json(cfg["sae_template"])
    sae["experiment_name"] = cfg["experiment_name"] + "_sae"
    sae["entry_decision"] = {"decision": "user_authorized_fresh_NCSU_reproduction", "formal_claim_allowed": False}
    sae["source_pool"] = {"generation_audit_path": str(audit_path), "generation_audit_sha256": file_sha256(audit_path),
        "raw_shards": [{"path": str(raw), "sha256": file_sha256(raw), "records": len(sae_rows)}],
        "question_count": n, "rollouts_per_question": cfg["teacher"]["num_rollouts"],
        "trajectory_count": len(sae_rows), "minimum_correct_per_question": 4}
    sae["question_split"].update(train=round(.7*n), dev=round(.15*n), test=n-round(.7*n)-round(.15*n))
    sae["teacher"]["snapshot_path"] = cfg["teacher"]["snapshot_path"]
    sae["activation_extraction"]["trajectory_shards"] = cfg["teacher"]["generation_shards"]
    sae_path = root / "sae/protocol/frozen_protocol.json"
    save(sae_path, sae)
    run_script("2_1_build_sae_corpus.py", config=sae_path, output_dir=root / "sae/corpus")
    seal(root / "MERGE_COMPLETE.json", [audit_path, sae_path, raw] + [root / f"sft_data/{r}.jsonl" for r in cfg["ranks"]])


def student_settings(cfg, rank, seed, data_path):
    student = dict(cfg["student"])
    student["model_name"] = student["snapshot_path"]
    student["tokenizer_kwargs"] = {"local_files_only": True}
    training = {**cfg["training"], "seed": seed, "data_seed": seed,
                "model_init_kwargs": {"local_files_only": True, "attn_implementation": "sdpa"}}
    return {"student": student, "training": training,
            "data": {"train_path": str(data_path), "text_format": "prompt_completion"}}


def evaluate(cfg, adapter, questions):
    from .student_evaluation import load_student_for_evaluation, evaluate_explicit_questions
    spec = {**cfg["student"], "model_name": cfg["student"]["snapshot_path"]}
    bundle = load_student_for_evaluation(spec, adapter_path=str(adapter) if adapter else None)
    rows = evaluate_explicit_questions(bundle, questions, batch_size=cfg["evaluation"]["batch_size"],
                                      max_new_tokens=cfg["evaluation"]["max_new_tokens"])
    del bundle
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return rows


def student(cfg, name):
    import torch
    from transformers import set_seed
    from .training import run_trl_sft
    from .student_evaluation import summarize_predictions
    root = Path(cfg["result_root"])
    dest = root / f"student_evaluation/{name}"
    if (dest / "COMPLETE.json").exists():
        verify(dest / "COMPLETE.json")
        return
    adapter = None
    bindings = [root / "protocol/frozen_protocol.json", root / "inputs/evaluation.jsonl"]
    if name != "base":
        verify(root / "MERGE_COMPLETE.json")
        rank, seed_string = name.split("__seed_")
        seed = int(seed_string)
        if rank not in cfg["ranks"] or seed not in cfg["student_seeds"]:
            raise ValueError(name)
        train_path = root / f"sft_data/{rank}.jsonl"
        rows = list(read_jsonl(train_path))
        run = student_settings(cfg, rank, seed, train_path)
        adapter = Path(cfg["checkpoint_root"]) / "students" / name
        local = Path(os.environ["TMPDIR"]) / "training" / name
        run["training"]["output_dir"] = str(adapter)
        run_path = root / f"student_configs/{name}.json"
        if not run_path.exists():
            save(run_path, run)
        elif read_json(run_path) != run:
            raise ValueError("Run config changed")
        if (adapter / "TRAIN_COMPLETE.json").exists():
            verify(adapter / "TRAIN_COMPLETE.json")
        else:
            if adapter.exists() or local.exists():
                raise FileExistsError("Partial training output requires explicit recovery")
            os.environ["LBD_RUNTIME_OUTPUT_DIR"] = str(local)
            set_seed(seed)
            trainer = run_trl_sft(run)
            steps = math.ceil(len(rows) / cfg["training"]["per_device_train_batch_size"])
            if trainer.state.global_step != steps:
                raise ValueError("Training step audit failed")
            save(local / "training_metrics.json", {"status": "complete", "records": len(rows),
                "steps": trainer.state.global_step, "seed": seed, "log_history": trainer.state.log_history})
            publish_files_hash_verified(local, adapter, ("adapter_model.safetensors", "adapter_config.json", "training_metrics.json"))
            seal(adapter / "TRAIN_COMPLETE.json", [train_path, run_path, CODE / "src/length_budget_distill/training.py",
                 Path(__file__), adapter / "adapter_model.safetensors", adapter / "adapter_config.json", adapter / "training_metrics.json"])
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
        bindings.extend([adapter / "TRAIN_COMPLETE.json", train_path, run_path,
                         adapter / "adapter_model.safetensors", adapter / "adapter_config.json"])
    questions = list(read_jsonl(root / "inputs/evaluation.jsonl"))
    rows = evaluate(cfg, adapter, questions)
    if len(rows) != cfg["evaluation"]["count"] or len({r["problem_id"] for r in rows}) != len(rows):
        raise ValueError("Evaluation missing/duplicate audit failed")
    if [r["problem_id"] for r in rows] != [r["problem_id"] for r in questions]:
        raise ValueError("Evaluation cohort mismatch")
    write_jsonl(dest / "predictions.jsonl", rows)
    save(dest / "metrics.json", summarize_predictions(rows))
    seal(dest / "COMPLETE.json", bindings + [dest / "predictions.jsonl", dest / "metrics.json"], name=name)


def smoke(cfg):
    import torch
    from transformers import set_seed
    from .training import run_trl_sft
    root = Path(cfg["result_root"]) / "smoke"
    if root.exists():
        raise FileExistsError(root)
    # Smoke generation is distinct and is never merged into the registered pool.
    small = json.loads(json.dumps(cfg))
    small["teacher"]["num_rollouts"] = 4
    bundle = teacher_bundle(small)
    questions = list(read_jsonl(Path(cfg["result_root"]) / "inputs/questions.jsonl"))[:2]
    rows = [r for q in questions for r in sample_question(small, bundle, q)]
    write_jsonl(root / "teacher_candidates.jsonl", rows)
    del bundle
    gc.collect()
    torch.cuda.empty_cache()
    train_path = root / "sft.jsonl"
    good = [r for r in rows if r["is_correct"] and not r["hit_max_new_tokens"]]
    if len(good) < 4:
        raise ValueError("Smoke teacher did not provide four complete correct examples")
    write_jsonl(train_path, [{"prompt": r["student_prompt"], "completion": r["solution"]} for r in good[:4]])
    run = student_settings(cfg, "short", 17, train_path)
    run["training"]["max_steps"] = 1
    output = Path(os.environ["TMPDIR"]) / "smoke_training"
    run["training"]["output_dir"] = str(output)
    set_seed(17)
    trainer = run_trl_sft(run)
    if trainer.state.global_step != 1:
        raise ValueError("Smoke SFT failed")
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    # Use official test[:50] only for smoke, rather than the locked formal cohort.
    ev = read_json(Path(cfg["result_root"]) / "inputs/smoke_evaluation.json")["questions"]
    predictions = evaluate(cfg, output, ev)
    write_jsonl(root / "predictions.jsonl", predictions)
    seal(root / "SMOKE_COMPLETE.json", [train_path, root / "teacher_candidates.jsonl", root / "predictions.jsonl"],
         teacher_correct=len(good), student_steps=1)


def auxiliary_stage(cfg, args):
    root = Path(cfg["result_root"])
    sae_root = root / "sae"
    sae_cfg = sae_root / "protocol/frozen_protocol.json"
    samples = sae_root / "token_samples"
    training = sae_root / "sae_training"
    features = root / "sae_features"
    feature_cfg = features / "protocol/frozen_protocol.json"
    cells = [(layer, k) for layer in (10, 17, 23) for k in (32, 64)]
    if args.stage == "submit":
        return submit(cfg)
    if args.stage == "finalize":
        return finalize(cfg)
    if args.stage == "sae_extract":
        count = cfg["teacher"]["generation_shards"]
        return run_script("2_2_extract_residual_activations.py", config=sae_cfg,
            corpus=sae_root / "corpus/mixed_trajectories.jsonl", shard_index=args.index,
            shard_count=count, output_dir=sae_root / f"activations/shard_{args.index:02d}_of_{count:02d}")
    if args.stage == "sae_sample":
        return run_script("2_3_build_sae_token_samples.py", config=sae_cfg,
            corpus=sae_root / "corpus/mixed_trajectories.jsonl", activation_root=sae_root / "activations", output_dir=samples)
    if args.stage == "sae_train":
        layer, k = cells[args.index]
        name = f"layer_{layer:02d}_k_{k:03d}"
        return run_script("2_4_train_topk_sae.py", config=sae_cfg, sample_root=samples,
            layer_index=layer, k=k, output_dir=training / name,
            checkpoint_dir=Path(cfg["checkpoint_root"]) / "sae" / name)
    if args.stage == "sae_audit":
        return run_script("2_6_audit_sae_pilot.py", config=sae_cfg, training_root=training, output_dir=sae_root / "audit")
    if args.stage == "features_freeze":
        spec = read_json(cfg["feature_template"])
        spec["experiment_name"] = cfg["experiment_name"] + "_features"
        spec["parent_sae"] = {"config_path": str(sae_cfg),
            "completion_marker_path": str(root / "SAE_PILOT_COMPLETE"),
            "corpus_path": str(sae_root / "corpus/mixed_trajectories.jsonl"),
            "activation_root": str(sae_root / "activations"), "training_root": str(training),
            "checkpoint_root": str(Path(cfg["checkpoint_root"]) / "sae")}
        spec["outputs"]["result_root"] = str(features)
        spec["outputs"]["figure_root"] = str(root / "analysis/sae_figures")
        source = root / "protocol/feature_source.json"
        save(source, spec)
        return run_script("2_7_freeze_short_long_feature_protocol.py", config=source)
    if args.stage == "features_score":
        layer, k = cells[args.index]
        return run_script("2_8_score_short_long_features.py", config=feature_cfg, layer_index=layer,
                          k=k, output_dir=features / f"feature_scores/layer_{layer:02d}_k_{k:03d}")
    if args.stage == "features_analyze":
        run_script("2_10_analyze_short_long_features.py", config=feature_cfg)
        return run_script("2_11_audit_short_long_feature_analysis.py", config=feature_cfg)
    raise ValueError(args.stage)


def submit(cfg):
    root = Path(cfg["result_root"])
    verify(root / "smoke/SMOKE_COMPLETE.json")
    submission = root / "submission.json"
    if submission.exists():
        raise FileExistsError("Submission already exists; inspect jobs instead of duplicating them")
    # Append after every successful submission, so interruption never hides live jobs.
    ledger = root / "submission_ledger.jsonl"
    if ledger.exists():
        raise FileExistsError("Partial submission ledger exists; resume the recorded DAG explicitly")
    runtime = cfg["runtime"]
    jobs = []
    def queue(stage, *, deps=(), gpu=False, index=None, name=None):
        cmd = ["sbatch", "--parsable", f"--account={runtime['gpu_account'] if gpu else runtime['cpu_account']}",
               f"--partition={runtime['gpu_partition'] if gpu else 'compute'}",
               f"--qos={runtime['gpu_qos'] if gpu else 'normal'}", "--nodes=1", "--ntasks=1",
               f"--cpus-per-task={runtime['cpus']}", f"--mem={runtime['memory']}",
               f"--time={runtime['gpu_time'] if gpu else runtime['cpu_time']}",
               f"--job-name=p12_{stage}_{name or index or 0}",
               f"--output={root}/logs/{stage}_{name or index or 0}_%j.log"]
        if gpu:
            cmd.append(f"--gres=gpu:{runtime['gpu_type']}:1")
        if deps:
            cmd.append("--dependency=afterok:" + ":".join(str(j) for j in deps))
            cmd.append("--kill-on-invalid-dep=yes")
        cmd.extend([str(CODE / "scripts/slurm/1_40_ncsu_reproduction.sh"), str(CODE),
                    str(root / "protocol/frozen_protocol.json"), runtime["python"], runtime["overlay"], stage])
        if index is not None:
            cmd.extend(["--index", str(index)])
        if name is not None:
            cmd.extend(["--name", name])
        job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Unexpected sbatch result {job}")
        entry = {"job_id": job, "stage": stage, "index": index, "name": name, "dependencies": list(deps), "command": cmd}
        with ledger.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        jobs.append(entry)
        logging.info("Submitted %s job=%s", stage, job)
        return job
    (root / "logs").mkdir(exist_ok=True)
    generation = [queue("generate", gpu=True, index=i) for i in range(cfg["teacher"]["generation_shards"])]
    base = queue("student", gpu=True, name="base")
    merge_job = queue("merge", deps=generation)
    students = [queue("student", gpu=True, deps=[merge_job], name=f"{rank}__seed_{seed}")
                for rank in cfg["ranks"] for seed in cfg["student_seeds"]]
    extract = [queue("sae_extract", gpu=True, deps=[merge_job], index=i)
               for i in range(cfg["teacher"]["generation_shards"])]
    sample = queue("sae_sample", deps=extract)
    sae = [queue("sae_train", gpu=True, deps=[sample], index=i) for i in range(6)]
    audit = queue("sae_audit", deps=sae)
    frozen = queue("features_freeze", deps=[audit])
    scores = [queue("features_score", gpu=True, deps=[frozen], index=i) for i in range(6)]
    figures = queue("features_analyze", deps=scores)
    final = queue("finalize", deps=[base, *students, figures])
    save(submission, {"status": "submitted_not_completed", "jobs": jobs, "final_job": final,
                     "config_sha256": file_sha256(root / "protocol/frozen_protocol.json")})


def finalize(cfg):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .ranked_multiseed_analysis import crossed_seed_problem_bootstrap
    root = Path(cfg["result_root"])
    verify(root / "MERGE_COMPLETE.json")
    names = ["base"] + [f"{r}__seed_{s}" for r in cfg["ranks"] for s in cfg["student_seeds"]]
    predictions, metrics, markers = {}, {}, []
    for name in names:
        directory = root / "student_evaluation" / name
        marker = directory / "COMPLETE.json"
        verify(marker)
        markers.append(marker)
        predictions[name] = {r["problem_id"]: float(r["is_correct"]) for r in read_jsonl(directory / "predictions.jsonl")}
        metrics[name] = read_json(directory / "metrics.json")
    expected = {r["problem_id"] for r in read_jsonl(root / "inputs/evaluation.jsonl")}
    if any(set(rows) != expected for rows in predictions.values()):
        raise ValueError("Prediction cohort audit failed")
    contrasts = {}
    for other in ("base", "medium", "long"):
        effects = {s: {p: predictions[f"short__seed_{s}"][p] - predictions[other if other == "base" else f"{other}__seed_{s}"][p]
                       for p in sorted(expected)} for s in cfg["student_seeds"]}
        contrasts[f"short_minus_{other}"] = crossed_seed_problem_bootstrap(effects,
            samples=cfg["bootstrap_samples"], seed=cfg["bootstrap_seed"])
    feature_marker = root / "sae_features/SHORT_LONG_FEATURE_ANALYSIS_COMPLETE"
    from .factorial import read_key_value_marker
    if read_key_value_marker(feature_marker).get("status") != "passed":
        raise ValueError("SAE feature audit did not pass")
    # Re-run the feature audit's hash checks without overwriting its artifacts.
    feature_audits = list((root / "sae_features/audit").glob("*.json"))
    if len(feature_audits) != 1:
        raise ValueError("Missing feature audit JSON")
    feature_audit = feature_audits[0]
    if read_key_value_marker(feature_marker)["audit_sha256"] != file_sha256(feature_audit):
        raise ValueError("SAE feature audit hash mismatch")
    for layer in (10, 17, 23):
        for k in (32, 64):
            run = root / f"sae/sae_training/layer_{layer:02d}_k_{k:03d}"
            record = read_json(run / "training_metrics.json")
            marker = read_key_value_marker(run / "SAE_TRAINING_COMPLETE")
            if marker["model_sha256"] != file_sha256(record["model_path"]):
                raise ValueError("SAE checkpoint changed")
            score = root / f"sae_features/feature_scores/layer_{layer:02d}_k_{k:03d}"
            score_marker = read_key_value_marker(score / "FEATURE_SCORING_COMPLETE")
            if score_marker["summary_sha256"] != file_sha256(score / "scoring_summary.json"):
                raise ValueError("Feature summary changed")
            for artifact in read_json(score / "scoring_summary.json")["artifacts"].values():
                if file_sha256(artifact["path"]) != artifact["sha256"]:
                    raise ValueError("Feature artifact changed")
            markers.extend([run / "SAE_TRAINING_COMPLETE", score / "FEATURE_SCORING_COMPLETE"])
    from .sae_feature_analysis import holm_adjust
    adjusted = holm_adjust([v["bootstrap_p_value"] for v in contrasts.values()])
    for effect, p in zip(contrasts.values(), adjusted):
        effect["holm_p_value"] = float(p)
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    summary = {"status": "complete", "metrics": metrics, "contrasts": contrasts,
               "claim_boundary": cfg["claim_boundary"], "formal_claim_allowed": False,
               "generation_audit": read_json(root / "generation/generation_audit.json")}
    save(analysis / "student_summary.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"short": "#2a9d8f", "medium": "#e9c46a", "long": "#e76f51"}
    for i, rank in enumerate(cfg["ranks"]):
        values = [100 * metrics[f"{rank}__seed_{s}"]["accuracy"] for s in cfg["student_seeds"]]
        axes[0].bar(i, np.mean(values), color=colors[rank], alpha=.8)
        axes[0].scatter([i]*len(values), values, color="black", s=20, zorder=3)
        lengths = [r["metadata"]["solution_token_count"] for r in read_jsonl(root / f"sft_data/{rank}.jsonl")]
        axes[1].bar(i, np.mean(lengths), color=colors[rank])
    axes[0].axhline(100*metrics["base"]["accuracy"], color="gray", linestyle="--", label="Base student")
    for ax in axes:
        ax.set_xticks(range(3), cfg["ranks"])
    axes[0].set_ylabel("GSM8K accuracy (%)")
    axes[0].set_title("Three training seeds; equal examples")
    axes[0].legend()
    axes[1].set_ylabel("Mean teacher completion tokens")
    axes[1].set_title("Freshly sampled SFT data")
    fig.savefig(analysis / "student_comparison.png", dpi=180)
    fig.savefig(analysis / "student_comparison.pdf")
    plt.close(fig)
    wins = all(contrasts[f"short_minus_{r}"]["ci_low"] > 0 and contrasts[f"short_minus_{r}"]["holm_p_value"] < .05 for r in ("base", "long"))
    report = ["# NCSU Phase 1/2 reproduction", "", "Fresh teacher sampling, student SFT, and SAE analysis are complete.", "",
              "Short outperformed both base and long under the registered comparisons." if wins else
              "The registered comparisons did not establish that short outperforms both base and long.", "",
              f"Base student accuracy: {100*metrics['base']['accuracy']:.2f}%.", ""]
    for rank in cfg["ranks"]:
        values = [metrics[f"{rank}__seed_{s}"]["accuracy"] for s in cfg["student_seeds"]]
        report.append(f"- {rank}: mean {100*np.mean(values):.2f}%; sample SD across seeds {100*np.std(values, ddof=1):.2f} percentage points.")
    report.extend(["", "![Student comparison](student_comparison.png)", "", cfg["claim_boundary"], "",
                   "SAE figures: [analysis artifacts](sae_figures/). Student contrasts and confidence intervals: [summary](student_summary.json).", ""])
    (analysis / "report.md").write_text("\n".join(report))
    seal(root / "EXPERIMENT_COMPLETE.json", markers + [feature_marker, feature_audit,
         root / "MERGE_COMPLETE.json", analysis / "student_summary.json", analysis / "report.md",
         analysis / "student_comparison.png", analysis / "student_comparison.pdf"], formal_claim_allowed=False)


def dispatch(args):
    config_path = Path(args.config).resolve()
    if args.stage == "prepare":
        return prepare(config_path)
    cfg = read_json(config_path)
    verify(Path(cfg["result_root"]) / "protocol/FROZEN.json")
    for p, digest in read_json(Path(cfg["result_root"]) / "protocol/sources.json").items():
        if file_sha256(p) != digest:
            raise ValueError(f"Frozen source changed: {p}")
    logging.info("stage=%s index=%s name=%s config=%s", args.stage, args.index, args.name, config_path)
    if args.stage in {"smoke", "generate", "student", "sae_extract", "sae_train", "features_score"}:
        admission(cfg)
    if args.stage in {"submit", "finalize", "sae_extract", "sae_sample", "sae_train", "sae_audit", "features_freeze", "features_score", "features_analyze"}:
        return auxiliary_stage(cfg, args)
    if args.stage == "generate":
        return generate(cfg, args.index)
    if args.stage == "student":
        return student(cfg, args.name)
    return globals()[args.stage](cfg)
