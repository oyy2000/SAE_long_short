import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PROJECT_ROOT / "src",):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trace_length_observation.controlled_sft import (
    controlled_causal_loss,
    encode_training_record,
)
from trace_length_observation.trace_observation import (
    answer_character_span,
    build_run_matrix,
    materialize_budget_schedules,
    select_quantile_band_candidates,
    validate_trace_observation_config,
)
from trace_length_observation.trace_observation_analysis import (
    analyze_cells,
    attach_training_accounting,
    decide_gate0,
)


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        values = [2 + (ord(character) % 17) for character in text]
        return ([19] if add_special_tokens else []) + values

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        payload = {
            "input_ids": self.encode(text, add_special_tokens=add_special_tokens)
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return payload


class ChatCharacterTokenizer(CharacterTokenizer):
    chat_template = "registered-test-template"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if messages != [{"role": "user", "content": "Question"}]:
            raise AssertionError("Unexpected chat messages")
        if not tokenize or not add_generation_prompt:
            raise AssertionError("Expected a tokenized assistant-generation prefix")
        return [31, 32, 33]


def candidate(index, tokens, problem_id="p0"):
    return {
        "trace_id": f"{problem_id}:trace:{index}",
        "problem_id": problem_id,
        "candidate_index": index,
        "solution": f"Distinct reasoning {index}.\nAnswer: {index + 10}",
        "solution_token_count": tokens,
        "max_solution_tokens": 512,
        "is_correct": True,
        "metadata": {},
        "prompt": f"Problem {problem_id}",
        "student_prompt": f"Student problem {problem_id}",
    }


class TraceObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (PROJECT_ROOT / "configs/trace_length_observation_gate0_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registered_config_is_valid(self):
        validate_trace_observation_config(self.config)
        self.assertEqual(self.config["training"]["expected_run_count"], 108)

    def test_quantile_selection_avoids_absolute_extremes(self):
        rows = [candidate(index, 10 * (index + 1)) for index in range(10)]
        selected = select_quantile_band_candidates(rows)
        self.assertEqual(selected["short"]["solution_token_count"], 20)
        self.assertEqual(selected["medium"]["solution_token_count"], 50)
        self.assertEqual(selected["long"]["solution_token_count"], 90)
        self.assertFalse(
            selected["short"]["metadata"]["phase0_length_selection"]["absolute_extreme"]
        )
        self.assertFalse(
            selected["long"]["metadata"]["phase0_length_selection"]["absolute_extreme"]
        )

    def test_quantile_selection_requires_four_correct(self):
        rows = [candidate(index, index + 1) for index in range(4)]
        rows[-1]["is_correct"] = False
        self.assertEqual(select_quantile_band_candidates(rows), {})

    def test_budget_schedules_separate_unique_and_update_estimands(self):
        tokenizer = CharacterTokenizer()
        selected = {}
        for problem_index in range(4):
            problem_id = f"p{problem_index}"
            selected[problem_id] = {}
            for rank, padding in (
                ("short", ""),
                ("medium", "mmmm"),
                ("long", "llllllll"),
            ):
                row = candidate(problem_index, 1, problem_id=problem_id)
                row["solution"] = f"r{problem_index}{padding}\nAnswer: {problem_index}"
                row["length_rank"] = rank
                selected[problem_id][rank] = row
        schedules, summary = materialize_budget_schedules(
            selected, tokenizer=tokenizer, seed=17, max_token_gap=64
        )
        self.assertEqual(len(schedules[("equal_examples", "short")]), 4)
        self.assertLessEqual(
            max(
                summary["schedules"][f"equal_target_tokens__{rank}"][
                    "actual_completion_tokens"
                ]
                for rank in ("short", "medium", "long")
            )
            - min(
                summary["schedules"][f"equal_target_tokens__{rank}"][
                    "actual_completion_tokens"
                ]
                for rank in ("short", "medium", "long")
            ),
            64,
        )
        self.assertLessEqual(
            max(
                summary["schedules"][f"equal_processed_tokens__{rank}"][
                    "actual_model_input_tokens"
                ]
                for rank in ("short", "medium", "long")
            )
            - min(
                summary["schedules"][f"equal_processed_tokens__{rank}"][
                    "actual_model_input_tokens"
                ]
                for rank in ("short", "medium", "long")
            ),
            64,
        )
        self.assertGreater(
            len(schedules[("equal_processed_tokens", "short")]),
            len(schedules[("equal_examples", "short")]),
        )
        expected_support = {f"p{index}" for index in range(4)}
        for regime in (
            "equal_examples",
            "equal_target_tokens",
            "equal_processed_tokens",
        ):
            for rank in ("short", "medium", "long"):
                self.assertEqual(
                    {row["problem_id"] for row in schedules[(regime, rank)]},
                    expected_support,
                )

    def test_run_matrix_contains_full_factorial(self):
        evidence = {}
        for regime in self.config["training"]["budget_regimes"]:
            for rank in ("short", "medium", "long"):
                evidence[(regime, rank)] = {
                    "train_path": f"{regime}/{rank}.jsonl",
                    "train_sha256": "a" * 64,
                    "record_count": 10,
                    "unique_problem_count": 10,
                    "budget_token_basis": "completion_tokens",
                    "target_budget_tokens": 100,
                    "actual_budget_tokens": 99,
                    "actual_completion_tokens": 99,
                    "actual_model_input_tokens": 140,
                    "token_gap": 1,
                    "duplicate_exposures": 0,
                }
        runs = build_run_matrix(self.config, evidence)
        self.assertEqual(len(runs), 108)
        self.assertEqual(len({run["run_name"] for run in runs}), 108)

    def test_answer_span_and_answer_only_encoding(self):
        completion = "Reasoning.\nAnswer: 42\n"
        start, end = answer_character_span(completion)
        self.assertEqual(completion[start:end], "42")
        encoded = encode_training_record(
            CharacterTokenizer(),
            {
                "prompt": "Question",
                "completion": completion,
                "problem_id": "p",
                "trace_id": "t",
                "occurrence_index": 0,
            },
            loss_mask="answer_only",
            max_length=128,
        )
        self.assertEqual(
            sum(encoded.loss_weights), 3.0
        )  # two answer characters plus EOS

    def test_answer_span_supports_parent_verifier_formats(self):
        for completion, expected in (
            ("Reasoning.\n#### 42", "42"),
            ("Reasoning. Therefore \\boxed{42}", "42"),
            ("Reasoning.\nFinal Answer = 42", "42"),
        ):
            start, end = answer_character_span(completion)
            self.assertEqual(completion[start:end], expected)
            encoded = encode_training_record(
                CharacterTokenizer(),
                {
                    "prompt": "Question",
                    "completion": completion,
                    "problem_id": "p",
                    "trace_id": "t",
                },
                loss_mask="answer_only",
                max_length=128,
            )
            self.assertGreater(sum(encoded.loss_weights), 1.0)

    def test_training_encoding_uses_registered_chat_template(self):
        encoded = encode_training_record(
            ChatCharacterTokenizer(),
            {
                "prompt": "Question",
                "completion": "Reasoning.\nAnswer: 7",
                "problem_id": "p",
                "trace_id": "t",
            },
            loss_mask="full_completion",
            max_length=128,
            completion_separator="",
        )
        self.assertEqual(encoded.input_ids[:3], [31, 32, 33])
        self.assertEqual(encoded.loss_weights[:3], [0.0, 0.0, 0.0])

    def test_token_and_sequence_normalization_are_distinct(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        input_ids = torch.zeros((2, 4), dtype=torch.long)
        logits = torch.zeros((2, 4, 2), dtype=torch.float32)
        logits[1, 0, 0] = -4.0
        weights = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        token_mean = controlled_causal_loss(
            logits, input_ids, weights, normalization="token_mean", torch_module=torch
        )
        sequence_mean = controlled_causal_loss(
            logits,
            input_ids,
            weights,
            normalization="sequence_mean",
            torch_module=torch,
        )
        self.assertGreater(float(sequence_mean), float(token_mean))

    def test_gate_requires_all_controlled_cells(self):
        contrasts = []
        for cell in self.config["gate0"]["required_short_vs_long_cells"]:
            contrasts.append(
                {
                    **cell,
                    "loss_mask": "full_completion",
                    "left_rank": "short",
                    "right_rank": "long",
                    "estimate": 0.02,
                    "ci_low": 0.001,
                    "ci_high": 0.04,
                    "bootstrap_holm_p_value": 0.04,
                }
            )
        self.assertEqual(decide_gate0(contrasts, self.config)["status"], "passed")
        contrasts[-1]["ci_low"] = -0.001
        self.assertEqual(decide_gate0(contrasts, self.config)["status"], "failed")

    def test_analysis_preserves_seed_and_compute_accounting(self):
        indexed = {}
        training_runs = []
        for rank, correctness in (("short", True), ("medium", True), ("long", False)):
            key = ("equal_processed_tokens", "token_mean", "full_completion", rank)
            indexed[key] = {}
            for seed_index, seed in enumerate((17, 42, 73)):
                indexed[key][seed] = {
                    "p0": {"is_correct": correctness, "output_token_count": 10},
                    "p1": {"is_correct": correctness, "output_token_count": 12},
                }
                training_runs.append(
                    {
                        "budget_regime": key[0],
                        "loss_normalization": key[1],
                        "loss_mask": key[2],
                        "length_rank": rank,
                        "record_count": 2,
                        "unique_problem_count": 2,
                        "completion_token_updates": 20,
                        "model_input_token_updates": 40,
                        "padded_model_token_updates": 44 + seed_index,
                        "effective_loss_token_updates": 22,
                        "optimizer_steps": 1,
                        "total_parameter_count": 100,
                        "trainable_parameter_count": 10,
                        "approximate_nonpadding_training_flops": 24000,
                        "approximate_padded_training_flops": 27000,
                        "mean_train_loss": 1.0,
                        "elapsed_seconds": 2.0 + seed_index,
                    }
                )
        arms, contrasts = analyze_cells(
            indexed,
            rank_contrasts=(("short", "medium"), ("medium", "long"), ("short", "long")),
            bootstrap_samples=100,
            bootstrap_seed=17,
        )
        attach_training_accounting(arms, training_runs)
        self.assertEqual(len(arms), 3)
        self.assertEqual(len(contrasts), 3)
        self.assertEqual(arms[0]["mean_padded_model_token_updates"], 45.0)
        self.assertEqual(arms[0]["mean_training_elapsed_seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
