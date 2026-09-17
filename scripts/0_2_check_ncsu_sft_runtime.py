"""Check the SFT stack on an allocated GPU using a tiny random Qwen2 model."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import accelerate
import datasets
import length_budget_distill
import matplotlib
import pandas
import scipy
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(17)
    config = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, use_cache=False,
    )
    model = get_peft_model(
        Qwen2ForCausalLM(config),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                   task_type="CAUSAL_LM"),
    ).to(device="cuda", dtype=torch.bfloat16)
    model.train()
    tokens = torch.randint(0, 128, (2, 32), device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(input_ids=tokens, labels=tokens).loss
    assert torch.isfinite(loss), "Non-finite loss"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all().item() for g in grads)
    assert any(torch.count_nonzero(g).item() for g in grads), "No nonzero LoRA gradient"
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "status": "passed", "scope": "synthetic SFT runtime smoke check; no experimental evidence",
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda, "loss": loss.item(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "peft", "accelerate", "datasets", "numpy",
            "scipy", "pandas", "matplotlib", "length-budget-distill")},
        "project_import": length_budget_distill.__file__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
