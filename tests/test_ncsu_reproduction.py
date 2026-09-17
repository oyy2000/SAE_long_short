import unittest
from length_budget_distill.ncsu_reproduction import gsm8k_question_record
from length_budget_distill.verifiers import verify_answer


class GSM8KInputTests(unittest.TestCase):
    def test_gold_target_is_final_answer_not_first_rationale_number(self):
        original = {"question": "How many clips?", "answer": "48 + 24 = 72 clips.\n#### 72"}
        row = gsm8k_question_record(12, original)
        self.assertEqual(row["raw_answer"], original["answer"])
        self.assertEqual(row["answer"], "72")
        self.assertTrue(verify_answer("72", row["answer"]))
        self.assertFalse(verify_answer("48", row["answer"]))
        self.assertEqual(row["problem_id"], "hf-000012")

    def test_missing_official_final_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            gsm8k_question_record(0, {"question": "How many?", "answer": "48 + 24 = 72"})
