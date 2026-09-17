"""Explicit native-chat completion labels and observed SFT exposure accounting.

Reuse TRL/Transformers optimization and collators. Encode across the same fixed
prefill boundary as generation, and never infer the boundary from a delimiter
inside a teacher's answer. Historical text-format training stays unchanged.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from .factorial import canonical_sha256

TRAIN_COLUMNS = ("input_ids", "attention_mask", "labels")


def validate_encoded_record(row: dict, *, max_length: int) -> int:
    """Reject truncation, prompt supervision, internal gaps, and shifted labels."""
    arrays = [row[k] for k in TRAIN_COLUMNS]
    ids, attention, labels = arrays
    if not ids or len(ids) > max_length or any(len(a) != len(ids) for a in arrays):
        raise ValueError("Empty, mismatched, or over-length completion record")
    if any(type(x) is not int or x < 0 for x in ids) or attention != [1] * len(ids):
        raise ValueError("Prepared completion records must be unpadded integer token sequences")
    supervised = [i for i, label in enumerate(labels) if label != -100]
    if not supervised or supervised[0] == 0:
        raise ValueError("A nonempty masked prompt and supervised completion are required")
    boundary = supervised[0]
    if labels != [-100] * boundary + ids[boundary:]:
        raise ValueError("Completion labels must exactly match the contiguous unshifted suffix")
    if "prompt_tokens" in row and row["prompt_tokens"] != boundary:
        raise ValueError("Recorded prompt boundary differs from the explicit labels")
    return boundary


def encode_completion(tokenizer: Any, row: dict, *, max_length: int, chat_kwargs: dict | None = None) -> dict:
    """Encode prompt and assistant suffix independently, preserving native EOT.

    The prompt IDs match generation even when tokenizing the combined text would
    merge across its final newline. The assistant suffix includes the template's
    end-of-turn token and whitespace; those are part of the supervision budget.
    """
    if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
        raise ValueError("A nonempty prompt is required")
    if not isinstance(row.get("completion"), str) or not row["completion"].strip():
        raise ValueError("A nonempty completion is required")
    kwargs = dict(chat_kwargs or {})
    if set(kwargs) & {"tokenize", "add_generation_prompt", "continue_final_message"}:
        raise ValueError("Chat options cannot override the registered boundary policy")
    messages = [{"role": "user", "content": row["prompt"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    complete = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["completion"]}],
        tokenize=False, add_generation_prompt=False, **kwargs)
    if not complete.startswith(prompt):
        raise ValueError("The native assistant response does not extend its generation prompt")
    suffix = complete[len(prompt):]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(suffix, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is None or response_ids.count(eos) != 1:
        raise ValueError("Expected exactly one native EOS/end-of-turn token in the completion")
    if tokenizer.decode(response_ids[response_ids.index(eos)+1:], skip_special_tokens=False).strip():
        raise ValueError("Only native whitespace may follow the end-of-turn token")
    result = {**row, "input_ids": prompt_ids + response_ids,
              "attention_mask": [1] * (len(prompt_ids) + len(response_ids)),
              "labels": [-100] * len(prompt_ids) + response_ids,
              "prompt_tokens": len(prompt_ids), "supervision_tokens": len(response_ids),
              "completion_text_tokens": len(tokenizer.encode(row["completion"], add_special_tokens=False)),
              "chat_template_sha256": canonical_sha256(tokenizer.chat_template),
              "chat_kwargs": kwargs, "encoding_policy": "native_prefill_plus_native_assistant_suffix_v1"}
    validate_encoded_record(result, max_length=max_length)
    result["training_tokens_sha256"] = canonical_sha256({k: result[k] for k in TRAIN_COLUMNS})
    return result


def make_completion_collator(tokenizer: Any) -> Any:
    from transformers import DataCollatorForSeq2Seq
    if tokenizer.padding_side != "right":
        raise ValueError("Completion SFT requires right padding")
    # Unlike causal-LM collators that mask every pad-token ID, this pads labels
    # by sequence length and keeps genuine EOS labels when pad_token == EOS.
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt")


class TrainingExposureAudit:
    """Check actual trainer batches and count successful training forwards.

    Deliberately single-process: a rank-local counter is not a global training
    budget. The compute_loss wrapper delegates unchanged to the pinned trainer.
    Gradient-checkpoint recomputation does not re-enter this wrapper.
    """

    def __init__(self, rows: list[dict], *, max_length: int):
        if not rows or len({r["problem_id"] for r in rows}) != len(rows):
            raise ValueError("Exposure auditing requires unique, nonempty problem records")
        self.rows = deepcopy(rows)
        self.lookup = {}
        for row in self.rows:
            validate_encoded_record(row, max_length=max_length)
            key = canonical_sha256(row["input_ids"])
            if key in self.lookup:
                raise ValueError("Two problem IDs have identical token sequences")
            self.lookup[key] = row
        self.counts = Counter({r["problem_id"]: 0 for r in self.rows})
        self.microbatches = self.input_tokens = self.supervision_tokens = 0
        self.padded_tokens = 0
        self.installed = False

    def _check_batch(self, batch: dict) -> list[dict]:
        arrays = {k: batch[k].detach().cpu().tolist() for k in TRAIN_COLUMNS}
        found = []
        for ids, attention, labels in zip(*(arrays[k] for k in TRAIN_COLUMNS)):
            n = sum(attention)
            if attention != [1]*n + [0]*(len(attention)-n):
                raise ValueError("Actual batch is not right padded")
            row = self.lookup.get(canonical_sha256(ids[:n]))
            if row is None or labels != row["labels"] + [-100]*(len(labels)-n):
                raise ValueError("Actual input or completion labels differ from the frozen record")
            found.append(row)
        if not found:
            raise ValueError("Empty training batch")
        return found

    def install(self, trainer: Any) -> None:
        if self.installed or int(trainer.args.world_size) != 1:
            raise ValueError("Install the exposure audit once in a single-process trainer")
        dataset = trainer.train_dataset
        if len(dataset) != len(self.rows):
            raise ValueError("TRL changed the number of prepared records")
        for expected, actual in zip(self.rows, dataset):
            if any(expected[k] != actual[k] for k in TRAIN_COLUMNS):
                raise ValueError("TRL changed the prepared tokens or labels")
        probes = sorted(self.rows, key=lambda r: len(r["input_ids"]))
        self._check_batch(trainer.data_collator([{k: r[k] for k in TRAIN_COLUMNS} for r in (probes[0], probes[-1])]))
        original = trainer.compute_loss

        def compute_loss(model: Any, inputs: dict, *args: Any, **kwargs: Any) -> Any:
            training = model.training
            matched = self._check_batch(inputs) if training else []
            result = original(model, inputs, *args, **kwargs)
            if training:
                self.microbatches += 1
                self.padded_tokens += int(inputs["input_ids"].numel())
                for row in matched:
                    self.counts[row["problem_id"]] += 1
                    self.input_tokens += len(row["input_ids"])
                    self.supervision_tokens += sum(x != -100 for x in row["labels"][1:])
            return result

        trainer.compute_loss = compute_loss
        self.installed = True

    def summary(self) -> dict:
        return {"installed": self.installed, "training_microbatches": self.microbatches,
                "observed_sequences": sum(self.counts.values()), "actual_input_tokens": self.input_tokens,
                "actual_supervision_tokens_after_causal_shift": self.supervision_tokens,
                "actual_padded_input_tokens": self.padded_tokens,
                "problem_exposures": dict(self.counts),
                "measurement": "successful training compute_loss calls; single process; no optimizer modification"}
