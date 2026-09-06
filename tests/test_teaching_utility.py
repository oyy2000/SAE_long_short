import sys
import tempfile
import unittest
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.credit_allocation import (
    relative_budget_gap,
    segment_reasoning_steps,
    select_step_knapsack,
    supervision_weights,
)
from length_budget_distill.experiment_io import (
    validated_artifact_marker,
    validated_gate_decision,
    validated_training_artifacts,
)
from length_budget_distill.factorial import file_sha256
from length_budget_distill.ranked_sampling import (
    build_length_agnostic_teacher_prompt,
)
from length_budget_distill.sae_data import (
    assign_within_question_length_labels,
    build_mixed_trace_corpus,
    stable_question_split,
    trace_shard,
)
from length_budget_distill.sae_sampling import PriorityReservoir, token_priorities
from length_budget_distill.teaching_utility import (
    choose_calibrated_eta,
    deterministic_question_split,
    normalized_linearization_error,
    select_candidate_roles,
    within_question_fixed_effect_regression,
)
from length_budget_distill.topk_sae import TopKSAE
from length_budget_distill.trace_baselines import (
    lark_g_hat,
    rank_surprisal_ratio,
    scas_blocks,
    scas_score,
)
from length_budget_distill.utility_analysis import (
    compare_rank_metrics,
    paired_question_bootstrap,
    per_question_spearman,
)
from length_budget_distill.weighted_sft import encode_weighted_training_record


class CharacterTokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        self.last_messages = messages
        values = [100, 101]
        return values if tokenize else "chat"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        payload = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            payload["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return payload


class TeachingUtilityTests(unittest.TestCase):
    def test_sae_question_split_is_question_disjoint(self):
        ids = [f"p{index}" for index in range(12)]
        first = stable_question_split(
            ids, train_count=8, dev_count=2, test_count=2, seed=17
        )
        second = stable_question_split(
            reversed(ids), train_count=8, dev_count=2, test_count=2, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(first.values()), Counter({"train": 8, "dev": 2, "test": 2})
        )

    def test_sae_analysis_labels_do_not_filter_training_corpus(self):
        rows = [
            {
                "trace_id": f"t{index}",
                "problem_id": "p0",
                "solution_token_count": index + 1,
                "candidate_index": index,
                "is_correct": index != 7,
            }
            for index in range(8)
        ]
        labels = assign_within_question_length_labels(rows)
        self.assertEqual(Counter(labels.values())["short"], 2)
        self.assertEqual(Counter(labels.values())["long"], 2)
        self.assertEqual(Counter(labels.values())["medium"], 1)
        corpus, summary = build_mixed_trace_corpus(
            rows, question_splits={"p0": "train"}
        )
        self.assertEqual(len(corpus), len(rows))
        self.assertTrue(summary["all_trajectories_mixed_for_training"])
        self.assertTrue(all(row["sae_training_included"] for row in corpus))

    def test_sae_trace_shards_are_stable(self):
        assignments = [trace_shard(f"trace-{index}", 3) for index in range(20)]
        self.assertEqual(
            assignments, [trace_shard(f"trace-{index}", 3) for index in range(20)]
        )
        self.assertTrue(set(assignments).issubset({0, 1, 2}))

    def test_priority_reservoir_keeps_global_lowest_keys(self):
        import numpy as np
        import torch

        reservoir = PriorityReservoir(capacity=4, hidden_size=2)
        activations = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        fields = torch.arange(8, dtype=torch.int32)
        priorities = np.asarray([80, 10, 70, 20, 60, 30, 50, 40], dtype=np.uint64)
        reservoir.add(
            activations[:3], fields[:3], fields[:3], fields[:3], priorities[:3]
        )
        reservoir.add(
            activations[3:], fields[3:], fields[3:], fields[3:], priorities[3:]
        )
        sampled = reservoir.tensors()
        observed = sampled["priorities"].numpy().view(np.uint64).tolist()
        self.assertEqual(observed, [10, 20, 30, 40])
        self.assertEqual(sampled["trace_indices"].tolist(), [1, 3, 5, 7])

    def test_token_priorities_are_deterministic_and_layer_specific(self):
        import numpy as np

        fields = np.arange(5, dtype=np.int64)
        first = token_priorities(
            fields, fields, fields, seed=17, layer_index=10, split_code=0
        )
        second = token_priorities(
            fields, fields, fields, seed=17, layer_index=10, split_code=0
        )
        other = token_priorities(
            fields, fields, fields, seed=17, layer_index=17, split_code=0
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, other))

    def test_topk_sae_respects_sparsity_and_decoder_norm(self):
        import torch

        model = TopKSAE(input_dim=8, feature_count=32, k=4)
        inputs = torch.randn(5, 8)
        reconstruction, values, indices, pre = model(inputs)
        self.assertEqual(reconstruction.shape, inputs.shape)
        self.assertEqual(values.shape, (5, 4))
        self.assertEqual(indices.shape, (5, 4))
        self.assertEqual(pre.shape, (5, 32))
        self.assertLessEqual(int((values > 0).sum(dim=1).max()), 4)
        model.normalize_decoder_()
        self.assertTrue(
            torch.allclose(model.decoder_weight.norm(dim=1), torch.ones(32), atol=1e-5)
        )

    def test_registered_sae_protocol_is_internally_consistent(self):
        protocol = json.loads(
            (PROJECT_ROOT / "configs/phase2_sae_pilot_v1.json").read_text(
                encoding="utf-8"
            )
        )
        source = protocol["source_pool"]
        self.assertEqual(
            source["question_count"] * source["rollouts_per_question"],
            source["trajectory_count"],
        )
        split = protocol["question_split"]
        self.assertEqual(
            split["train"] + split["dev"] + split["test"], source["question_count"]
        )
        sae = protocol["sae"]
        self.assertEqual(
            sae["feature_count"],
            protocol["teacher"]["hidden_size"] * sae["expansion_factor"],
        )
        self.assertEqual(
            protocol["pilot_outputs"]["required_sae_count"],
            len(protocol["activation_extraction"]["layer_indices_zero_based"])
            * len(sae["k_values"]),
        )

    def test_generic_artifact_marker_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            marker = root / "COMPLETE"
            artifact.write_text("{}\n", encoding="utf-8")
            marker.write_text(
                f"status=complete\nartifact_sha256={file_sha256(artifact)}\n",
                encoding="utf-8",
            )
            evidence = validated_artifact_marker(
                marker,
                expected_status="complete",
                hash_bindings={"artifact_sha256": artifact},
            )
            self.assertEqual(evidence["status"], "complete")
            artifact.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                validated_artifact_marker(
                    marker,
                    expected_status="complete",
                    hash_bindings={"artifact_sha256": artifact},
                )

    def test_extension_teacher_prompt_matches_sealed_parent_template(self):
        expected = (
            "You are a careful math teacher. Solve the problem correctly with visible "
            "step-by-step reasoning. Do not target a particular response length; use the "
            "amount of detail that follows naturally from your solution. End with a line "
            "in the form: Answer: <final answer>.\n\nProblem:\nQ"
        )
        self.assertEqual(build_length_agnostic_teacher_prompt("Q"), expected)

    def test_registered_phase1_protocols_are_internally_consistent(self):
        phase1 = json.loads(
            (PROJECT_ROOT / "configs/phase1_teaching_utility_v1.json").read_text(
                encoding="utf-8"
            )
        )
        phase1_5 = json.loads(
            (PROJECT_ROOT / "configs/phase1_5_credit_allocation_v1.json").read_text(
                encoding="utf-8"
            )
        )
        source = phase1["source_pool"]
        self.assertEqual(
            source["existing_problem_count"] + source["extension_problem_count"],
            source["target_problem_count"],
        )
        split = phase1["dataset"]["question_split"]
        self.assertEqual(
            split["train"] + split["dev"] + split["test"],
            source["target_problem_count"],
        )
        exact = phase1["exact_utility"]
        self.assertEqual(phase1["calibration"]["split"], "train")
        self.assertEqual(exact["split"], "dev")
        self.assertEqual(
            exact["candidate_count"],
            exact["question_count"] * exact["candidates_per_question"],
        )
        self.assertEqual(
            exact["micro_update_count"],
            exact["candidate_count"]
            * len(phase1["utility"]["candidate_loss_reductions"]),
        )
        pilot_count = len(phase1_5["pilot"]["conditions"])
        confirmation = phase1_5["confirmation"]
        added = len(confirmation["conditions"]) * (len(confirmation["seeds"]) - 1)
        self.assertEqual(confirmation["unique_adapter_count"], pilot_count + added)

    def test_gate_decision_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = root / "decision.json"
            marker = root / "COMPLETE"
            decision.write_text('{"status": "passed"}\n', encoding="utf-8")
            marker.write_text(
                "status=complete\ngate_status=passed\n"
                f"gate_decision_sha256={file_sha256(decision)}\n",
                encoding="utf-8",
            )
            evidence = validated_gate_decision(
                root,
                gate_name="gate",
                decision_path=decision,
                completion_marker_path=marker,
            )
            self.assertEqual(evidence["status"], "passed")
            decision.write_text(
                '{"status": "passed", "tampered": true}\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validated_gate_decision(
                    root,
                    gate_name="gate",
                    decision_path=decision,
                    completion_marker_path=marker,
                )

    def test_question_split_is_disjoint_and_deterministic(self):
        ids = [f"p{index}" for index in range(10)]
        first = deterministic_question_split(
            ids, train_count=6, dev_count=2, test_count=2, seed=17
        )
        second = deterministic_question_split(
            ids, train_count=6, dev_count=2, test_count=2, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(
            set(first["train"]) | set(first["dev"]) | set(first["test"]), set(ids)
        )
        self.assertFalse(set(first["train"]) & set(first["dev"]))

    def test_candidate_roles_avoid_extreme_short_definition(self):
        rows = [
            {"trace_id": f"t{index}", "solution_token_count": 10 * (index + 1)}
            for index in range(10)
        ]
        roles = select_candidate_roles(rows, seed=17)
        self.assertEqual(roles["quantile_short"]["solution_token_count"], 20)
        self.assertEqual(roles["quantile_long"]["solution_token_count"], 90)
        self.assertEqual(len({row["trace_id"] for row in roles.values()}), 4)

    def test_candidate_roles_remain_distinct_when_lengths_tie(self):
        rows = [
            {"trace_id": f"t{index}", "solution_token_count": 10} for index in range(4)
        ]
        roles = select_candidate_roles(rows, seed=17)
        self.assertEqual(len({row["trace_id"] for row in roles.values()}), 4)

    def test_fixed_effect_regression_recovers_within_question_signal(self):
        rows = []
        for question, offset in (("a", 10.0), ("b", -7.0), ("c", 3.0)):
            for length, nll in ((1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)):
                rows.append(
                    {
                        "problem_id": question,
                        "utility": offset - 2.0 * length + 0.5 * nll,
                        "log_length": length,
                        "student_nll": nll,
                    }
                )
        result = within_question_fixed_effect_regression(
            rows, outcome="utility", predictors=("log_length", "student_nll")
        )
        # The synthetic predictors are collinear, so verify fitted values via
        # their identifiable contrast instead of individual coefficients.
        coefficients = result["coefficients"]
        self.assertAlmostEqual(
            coefficients["log_length"] - coefficients["student_nll"], -2.5
        )

    def test_eta_calibration_chooses_largest_passing(self):
        rows = [
            {
                "relative_update": 1e-6,
                "sign_agreement": 0.95,
                "adjacent_spearman": 0.95,
                "median_linearization_error": 0.05,
            },
            {
                "relative_update": 3e-6,
                "sign_agreement": 0.91,
                "adjacent_spearman": 0.92,
                "median_linearization_error": 0.08,
            },
            {
                "relative_update": 1e-5,
                "sign_agreement": 0.80,
                "adjacent_spearman": 0.95,
                "median_linearization_error": 0.05,
            },
        ]
        selected = choose_calibrated_eta(
            rows,
            minimum_sign_agreement=0.9,
            minimum_adjacent_spearman=0.9,
            maximum_median_linearization_error=0.1,
        )
        self.assertEqual(selected["relative_update"], 3e-6)
        self.assertAlmostEqual(normalized_linearization_error(1.0, 0.9), 0.1)

    def test_baseline_formulas(self):
        self.assertAlmostEqual(rank_surprisal_ratio([-1.0, -2.0], [2.0, 4.0]), 2.0)
        self.assertAlmostEqual(scas_score(0.2, 0.8, weight=0.5), 0.5)
        blocks = scas_blocks(
            answer_mean_nll=2.0,
            question_mean_nll=3.0,
            answer_answer_similarity=0.25,
            answer_question_similarity=0.5,
            weight=0.5,
        )
        self.assertAlmostEqual(blocks["scas_score"], 2.0)
        scores = lark_g_hat([1.0, 2.0], [0.5, 0.25])
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])

    def test_step_knapsack_keeps_answer_supervised(self):
        completion = "First calculation is written here.\nSecond calculation is also written here.\nAnswer: 42"
        steps = segment_reasoning_steps(
            completion, CharacterTokenizer(), minimum_step_tokens=8
        )
        reasoning = [step for step in steps if not step.is_answer]
        selected = select_step_knapsack(
            steps,
            [2.0 if step == reasoning[0] else 1.0 for step in steps],
            target_reasoning_tokens=reasoning[0].token_count,
        )
        weights = supervision_weights(len(completion), steps, selected)
        self.assertIn(reasoning[0].index, selected)
        answer = next(step for step in steps if step.is_answer)
        self.assertTrue(
            all(
                weights[index] == 1.0
                for index in range(answer.token_start, answer.token_end)
            )
        )
        self.assertLessEqual(
            relative_budget_gap(reasoning[0].token_count, reasoning[0].token_count),
            0.05,
        )

    def test_step_knapsack_fills_budget_even_for_negative_scores(self):
        completion = "First calculation is written here. Answer: 42"
        steps = segment_reasoning_steps(
            completion, CharacterTokenizer(), minimum_step_tokens=8
        )
        reasoning = next(step for step in steps if not step.is_answer)
        selected = select_step_knapsack(
            steps,
            [-2.0 if not step.is_answer else 0.0 for step in steps],
            target_reasoning_tokens=reasoning.token_count,
        )
        self.assertEqual(selected, [reasoning.index])

    def test_inline_answer_is_a_separate_step(self):
        completion = "First calculate a useful intermediate value. Answer: 42"
        steps = segment_reasoning_steps(
            completion, CharacterTokenizer(), minimum_step_tokens=8
        )
        self.assertEqual(sum(step.is_answer for step in steps), 1)
        self.assertGreater(sum(not step.is_answer for step in steps), 0)

    def test_weighted_record_requires_exact_completion_mask(self):
        tokenizer = CharacterTokenizer()
        row = {
            "problem_id": "p",
            "trace_id": "t",
            "prompt": "question",
            "completion": "abc",
            "completion_loss_weights": [1.0, 0.0, 1.0],
        }
        encoded = encode_weighted_training_record(tokenizer, row, max_length=32)
        self.assertEqual(encoded.completion_token_count, 3)
        self.assertEqual(encoded.loss_weights[-4:], [1.0, 0.0, 1.0, 1.0])

    def test_paired_question_rank_analysis(self):
        rows = []
        for question in ("a", "b", "c", "d"):
            for index in range(4):
                rows.append(
                    {
                        "problem_id": question,
                        "utility": float(index),
                        "primary": float(index),
                        "baseline": float(3 - index),
                    }
                )
        primary = per_question_spearman(rows, outcome="utility", score="primary")
        baseline = per_question_spearman(rows, outcome="utility", score="baseline")
        self.assertEqual(set(primary), {"a", "b", "c", "d"})
        contrast = paired_question_bootstrap(primary, baseline, samples=100, seed=17)
        self.assertAlmostEqual(contrast["estimate"], 2.0)
        result = compare_rank_metrics(
            rows,
            outcome="utility",
            metrics={"primary": "primary", "baseline": "baseline"},
            primary_metric="primary",
            baseline_metrics=["baseline"],
            samples=100,
            seed=17,
        )
        self.assertTrue(result["paired_contrasts"][0]["passed"])

    def test_training_artifact_validation_is_marker_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (root / "adapter_model.safetensors").write_bytes(b"test")
            (root / "training_metrics.json").write_text(
                '{"status":"complete"}\n', encoding="utf-8"
            )
            (root / "TRAIN_COMPLETE").write_text(
                "status=complete\n"
                f"adapter_config_sha256={file_sha256(root / 'adapter_config.json')}\n"
                f"adapter_model_sha256={file_sha256(root / 'adapter_model.safetensors')}\n"
                f"training_metrics_sha256={file_sha256(root / 'training_metrics.json')}\n",
                encoding="utf-8",
            )
            evidence = validated_training_artifacts(root)
            self.assertEqual(evidence["metrics"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
