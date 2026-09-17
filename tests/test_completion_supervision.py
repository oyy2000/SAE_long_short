"""Exercise actual collators/loss/trainer and corruption/format counterexamples."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from length_budget_distill.completion_supervision import (
    TRAIN_COLUMNS, TrainingExposureAudit, encode_completion,
    make_completion_collator, validate_encoded_record,
)


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {word: i for i, word in enumerate(
        ["[UNK]", "[BOS]", "[EOS]", "user", "assistant", "one", "two", "three", "answer", "question"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
                                 eos_token="[EOS]", pad_token="[EOS]", model_max_length=128)
    tok.chat_template = "[BOS] {% for m in messages %}{{m['role']}} {{m['content']}} {% endfor %}{% if add_generation_prompt %}assistant {% else %}[EOS]{% endif %}"
    return tok


class TestCompletionSupervision(unittest.TestCase):
    def setUp(self):
        self.tok = tiny_tokenizer()
        self.rows = [encode_completion(self.tok, {"problem_id": str(i), "prompt": "question " + "one "*i,
                     "completion": "answer " + "two "*(i+1)}, max_length=128) for i in range(3)]

    def test_actual_collator_keeps_eos_and_masks_only_padding(self):
        import torch
        batch = make_completion_collator(self.tok)([{k:r[k] for k in TRAIN_COLUMNS} for r in self.rows])
        for i, row in enumerate(self.rows):
            n = len(row["input_ids"])
            self.assertEqual(batch["labels"][i,:n].tolist(), row["labels"])
            self.assertEqual(batch["labels"][i,n-1].item(), self.tok.eos_token_id)
            self.assertTrue(torch.all(batch["labels"][i,n:] == -100))

    def test_template_delimiter_inside_answer_does_not_remask(self):
        row = encode_completion(self.tok, {"problem_id":"nested", "prompt":"question",
            "completion":"answer assistant one"}, max_length=128)
        self.assertEqual(row["labels"][row["prompt_tokens"]:], row["input_ids"][row["prompt_tokens"]:])
        self.assertIn(self.tok.convert_tokens_to_ids("assistant"), row["labels"])

    def test_rejects_overflow_gaps_empty_and_missing_native_eos(self):
        with self.assertRaises(ValueError): validate_encoded_record(self.rows[0], max_length=2)
        bad = deepcopy(self.rows[0]); bad["labels"][-2] = -100
        with self.assertRaises(ValueError): validate_encoded_record(bad, max_length=128)
        bad = deepcopy(self.rows[0]); bad["labels"][0] = bad["input_ids"][0]
        with self.assertRaises(ValueError): validate_encoded_record(bad, max_length=128)
        with self.assertRaises(ValueError):
            encode_completion(self.tok, {"prompt":"question", "completion":" "}, max_length=128)
        self.tok.chat_template = self.tok.chat_template.replace("[EOS]", "")
        with self.assertRaises(ValueError):
            encode_completion(self.tok, {"prompt":"question", "completion":"one"}, max_length=128)

    def test_batch_corruption_and_duplicate_identity_fail(self):
        with self.assertRaises(ValueError): TrainingExposureAudit([self.rows[0], self.rows[0]], max_length=128)
        meter = TrainingExposureAudit(self.rows, max_length=128)
        batch = make_completion_collator(self.tok)([{k:r[k] for k in TRAIN_COLUMNS} for r in self.rows])
        batch["labels"][0,0] = self.tok.eos_token_id
        with self.assertRaises(ValueError): meter._check_batch(batch)

    def test_actual_causal_model_loss_matches_shifted_completion_loss(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(self.tok), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)).eval()
        batch = make_completion_collator(self.tok)([{k:r[k] for k in TRAIN_COLUMNS} for r in self.rows])
        output = model(**batch)
        expected = torch.nn.functional.cross_entropy(output.logits[:,:-1].reshape(-1,len(self.tok)),
                                                       batch["labels"][:,1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(output.loss, expected)

    def test_real_trl_sft_preserves_labels_and_counts_partial_epoch(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from length_budget_distill.training import run_trl_sft
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model_path = root/"base"
            Qwen2ForCausalLM(Qwen2Config(vocab_size=len(self.tok), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=128, eos_token_id=self.tok.eos_token_id)).save_pretrained(model_path)
            self.tok.save_pretrained(model_path)
            data = root/"train.jsonl"
            data.write_text("".join(json.dumps(r)+"\n" for r in self.rows))
            meter = TrainingExposureAudit(self.rows, max_length=128)
            trainer = run_trl_sft({"student":{"model_name":str(model_path), "use_lora":True, "torch_dtype":"float32",
                    "lora":{"r":2,"alpha":4,"dropout":0.,"target_modules":["q_proj","v_proj"]},
                    "tokenizer_kwargs":{"local_files_only":True}},
                "data":{"train_path":str(data),"text_format":"pretokenized_completion"},
                "training":{"max_steps":1,"num_train_epochs":3,"max_length":128,
                    "bf16":False,"gradient_checkpointing":False,"per_device_train_batch_size":1,
                    "gradient_accumulation_steps":2,"output_dir":str(root/"adapter"),"save_strategy":"no",
                    "model_init_kwargs":{"device_map":None,"local_files_only":True},"dataset_kwargs":{},
                    "learning_rate":.001,"report_to":"none","disable_tqdm":True}}, before_train=meter.install)
            measured = meter.summary()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(measured["observed_sequences"], 2)
            self.assertEqual(measured["training_microbatches"], 2)
            self.assertEqual(measured["actual_supervision_tokens_after_causal_shift"],
                sum(measured["problem_exposures"][r["problem_id"]]*r["supervision_tokens"] for r in self.rows))
            self.assertEqual(sum(v==0 for v in measured["problem_exposures"].values()),1)


if __name__ == "__main__": unittest.main()
