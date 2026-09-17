"""Guard the staged matrix against duplicated runs and inflated method counts."""
import copy,json,unittest
from pathlib import Path
from collections import Counter
from length_budget_distill.baseline_plan import compile_matrix
ROOT=Path(__file__).resolve().parents[1]
class TestBaselineMatrix(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/phase13_baseline_expansion_v1.json').read_text())
    def test_reference_matrix_has_84_distinct_runs(self):
        rows=compile_matrix(self.cfg)
        self.assertEqual(Counter(r['module'] for r in rows),{'primary':48,'cross_family':12,'cross_teacher':12,'token_budget':12})
        self.assertEqual(sum(r['condition']=='B6' for r in rows),6)
    def test_duplicate_student_is_rejected(self):
        self.cfg['models']['primary_students'][1]=self.cfg['models']['primary_students'][0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):compile_matrix(self.cfg)
    def test_unregistered_method_cannot_enter_matrix(self):
        self.cfg['matrix']['primary']['conditions'][0]='unknown'
        with self.assertRaisesRegex(ValueError,'Unknown'):compile_matrix(self.cfg)
if __name__=='__main__':unittest.main()
