import unittest
import tempfile
from pathlib import Path
from length_budget_distill.token_kd_baselines import paired_ids,publish_prepared_rows


class BaselinePairingTests(unittest.TestCase):
    def test_rejects_duplicate_and_wrong_order(self):
        with self.assertRaises(ValueError):paired_ids([{'problem_id':'a'},{'problem_id':'a'}])
        with self.assertRaises(ValueError):paired_ids([{'problem_id':'b'},{'problem_id':'a'}],['a','b'])

    def test_rejects_missing_cell_examples(self):
        with self.assertRaises(ValueError):paired_ids([{'problem_id':'a'}],['a','b'])
        self.assertEqual(paired_ids([{'problem_id':'a'},{'problem_id':'b'}],['a','b']),['a','b'])

    def test_prepared_b1_is_verified_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'B1.jsonl';rows=[{'problem_id':'a'}]
            publish_prepared_rows(path,rows)
            original=path.read_bytes()
            publish_prepared_rows(path,rows,reuse=True)
            self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError):publish_prepared_rows(path,[{'problem_id':'b'}],reuse=True)


if __name__=='__main__':unittest.main()
