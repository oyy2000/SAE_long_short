#!/usr/bin/env python3
"""Train one registered TopK SAE from a deterministic activation-token sample."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    validated_artifact_marker,
    write_json_exclusive,
)
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument(
        "--sample-root", default="results/phase2_sae_pilot_v1/formal/token_samples"
    )
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    sample_root = _resolve(args.sample_root)
    output_dir = _resolve(args.output_dir)
    checkpoint_dir = _resolve(args.checkpoint_dir)
    if output_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir} or {checkpoint_dir}")
    config = read_json(config_path)
    extraction = config["activation_extraction"]
    sae_config = config["sae"]
    if args.layer_index not in [
        int(value) for value in extraction["layer_indices_zero_based"]
    ]:
        raise ValueError("Layer is not registered in the frozen protocol.")
    if args.k not in [int(value) for value in sae_config["k_values"]]:
        raise ValueError("k is not registered in the frozen protocol.")
    manifest_path = sample_root / "sample_manifest.json"
    validated_artifact_marker(
        sample_root / "TOKEN_SAMPLES_COMPLETE",
        expected_status="complete",
        hash_bindings={"manifest_sha256": manifest_path},
    )
    sample_manifest = read_json(manifest_path)
    if sample_manifest["config_hash"] != canonical_sha256(config):
        raise ValueError("Token samples were built under another protocol.")
    layer_manifest = next(
        row
        for row in sample_manifest["layers"]
        if int(row["layer_index"]) == args.layer_index
    )
    for evidence in layer_manifest["samples"]:
        if file_sha256(evidence["path"]) != evidence["sha256"]:
            raise ValueError(f"Token sample hash mismatch: {evidence['path']}")
    if (
        file_sha256(layer_manifest["normalizer_path"])
        != layer_manifest["normalizer_sha256"]
    ):
        raise ValueError("Activation normalizer hash mismatch.")
    import torch
    from safetensors.torch import load_file, save_file

    from length_budget_distill.topk_sae import (
        RunningFeatureActivity,
        TopKSAE,
        auxiliary_dead_feature_loss,
        reconstruction_metrics,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAE training.")
    seed = int(sae_config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    samples = {
        row["split"]: load_file(row["path"], device="cpu")["activations"]
        for row in layer_manifest["samples"]
    }
    normalizer = load_file(layer_manifest["normalizer_path"], device="cpu")
    mean = normalizer["mean"].float()
    scale = float(normalizer["scale"].item())
    hidden_size = int(config["teacher"]["hidden_size"])
    feature_count = int(sae_config["feature_count"])
    if feature_count != hidden_size * int(sae_config["expansion_factor"]):
        raise ValueError("SAE feature count does not match expansion factor.")
    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16
    model = TopKSAE(hidden_size, feature_count, args.k).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(sae_config["learning_rate"]),
        betas=(
            float(sae_config["adam_beta1"]),
            float(sae_config["adam_beta2"]),
        ),
        weight_decay=float(sae_config["weight_decay"]),
    )
    max_steps = int(sae_config["max_steps"])
    warmup_steps = int(sae_config["warmup_steps"])

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    activity = RunningFeatureActivity.create(feature_count, device)
    batch_size = int(sae_config["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed + args.layer_index)
    permutation = torch.randperm(len(samples["train"]), generator=generator)
    cursor = 0
    logs = []
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.train()
    for step in range(1, max_steps + 1):
        if cursor + batch_size > len(permutation):
            permutation = torch.randperm(len(samples["train"]), generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + batch_size]
        cursor += batch_size
        inputs = _normalize(samples["train"][indices], mean, scale).to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            reconstruction, values, feature_indices, pre = model(inputs)
            reconstruction_loss = (reconstruction - inputs).square().mean()
            activity.update(feature_indices, values)
            dead_mask = activity.steps_since_active >= int(
                sae_config["dead_feature_window_steps"]
            )
            auxiliary_loss = auxiliary_dead_feature_loss(
                model,
                inputs - reconstruction,
                pre,
                dead_mask,
                int(sae_config["auxiliary_k"]),
            )
            loss = (
                reconstruction_loss
                + float(sae_config["auxiliary_loss_coefficient"]) * auxiliary_loss
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite SAE loss at step {step}")
        loss.backward()
        model.remove_decoder_parallel_gradients_()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(sae_config["gradient_clip_norm"])
        )
        optimizer.step()
        model.normalize_decoder_()
        scheduler.step()
        if step == 1 or step % int(sae_config["evaluation_interval_steps"]) == 0:
            model.eval()
            dev_summary = reconstruction_metrics(
                model,
                _batches(samples["dev"], mean, scale, batch_size),
                device=device,
                autocast_dtype=autocast_dtype,
            )
            model.train()
            row = {
                "step": step,
                "train_loss": float(loss.detach().item()),
                "train_reconstruction_loss": float(reconstruction_loss.detach().item()),
                "train_auxiliary_loss": float(auxiliary_loss.detach().item()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "dev_mse": float(dev_summary["mse"]),
                "dev_explained_variance": float(dev_summary["explained_variance"]),
                "dev_mean_l0": float(dev_summary["mean_l0"]),
                "dev_dead_feature_fraction": float(
                    dev_summary["dead_feature_fraction"]
                ),
                "elapsed_seconds": time.time() - started,
            }
            logs.append(row)
            print(json.dumps(row), flush=True)
        if step % int(sae_config["checkpoint_interval_steps"]) == 0:
            _save_checkpoint(
                save_file,
                model,
                mean,
                scale,
                checkpoint_dir / f"step_{step:05d}.safetensors",
                args.layer_index,
                args.k,
                step,
            )
    model.eval()
    final_metrics = {
        split: reconstruction_metrics(
            model,
            _batches(samples[split], mean, scale, batch_size),
            device=device,
            autocast_dtype=autocast_dtype,
        )
        for split in ("train", "dev", "test")
    }
    model_path = checkpoint_dir / "sae_model.safetensors"
    _save_checkpoint(
        save_file,
        model,
        mean,
        scale,
        model_path,
        args.layer_index,
        args.k,
        max_steps,
    )
    metrics = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "sample_manifest_path": str(manifest_path),
        "sample_manifest_sha256": file_sha256(manifest_path),
        "layer_index": args.layer_index,
        "k": args.k,
        "hidden_size": hidden_size,
        "feature_count": feature_count,
        "max_steps": max_steps,
        "elapsed_seconds": time.time() - started,
        "training_log": logs,
        "final_metrics": final_metrics,
        "lifetime_activation_count": activity.activation_counts.cpu().tolist(),
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "runtime": runtime_metadata(("python", "torch", "safetensors")),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    metrics_path = output_dir / "training_metrics.json"
    write_json_exclusive(metrics_path, metrics)
    marker_text = (
        f"status=complete\nconfig_hash={metrics['config_hash']}\n"
        f"model_sha256={metrics['model_sha256']}\n"
        f"training_metrics_sha256={file_sha256(metrics_path)}\n"
        f"layer_index={args.layer_index}\nk={args.k}\nformal_claim_allowed=false\n"
    )
    (output_dir / "SAE_TRAINING_COMPLETE").write_text(marker_text, encoding="utf-8")
    (checkpoint_dir / "SAE_TRAINING_COMPLETE").write_text(marker_text, encoding="utf-8")


def _normalize(batch, mean, scale):
    return (batch.float() - mean) * scale


def _batches(values, mean, scale, batch_size):
    for start in range(0, len(values), batch_size):
        yield _normalize(values[start : start + batch_size], mean, scale)


def _save_checkpoint(save_file, model, mean, scale, path, layer_index, k, step) -> None:
    tensors = {
        "encoder_weight": model.encoder_weight.detach().cpu().contiguous(),
        "encoder_bias": model.encoder_bias.detach().cpu().contiguous(),
        "decoder_weight": model.decoder_weight.detach().cpu().contiguous(),
        "decoder_bias": model.decoder_bias.detach().cpu().contiguous(),
        "activation_mean": mean.contiguous(),
        "activation_scale": mean.new_tensor([scale]).contiguous(),
    }
    partial = path.parent / f".{path.name}.partial-{os.getpid()}"
    save_file(
        tensors,
        partial,
        metadata={
            "layer_index": str(layer_index),
            "k": str(k),
            "step": str(step),
            "architecture": "TopK",
        },
    )
    os.replace(partial, path)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
