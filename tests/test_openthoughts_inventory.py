import unittest
from length_budget_distill.openthoughts_inventory import reference_screen, exact_overlap
from length_budget_distill.baseline_data_preflight import near_matches
from length_budget_distill.dap_paired_sources import question_key


class InventoryTests(unittest.TestCase):
    def test_generated_answer_cannot_replace_missing_reference(self):
        row = {'problem': 'What is 2+2?', 'ground_truth_solution': None,
               'deepseek_solution': r'The answer is \boxed{4}.'}
        result = reference_screen(row)
        self.assertIsNone(result['reference_box_candidate'])
        self.assertIn('missing_reference', result['review_flags'])
        self.assertFalse(result['formal_training_ready'])

    def test_box_does_not_make_proof_or_diagram_eligible(self):
        row = {'problem': 'Prove the relation in the figure.', 'ground_truth_solution': r'\boxed{x=y}'}
        result = reference_screen(row)
        self.assertEqual(set(result['review_flags']), {'proof_wording_review', 'visual_dependency_review'})
        self.assertFalse(result['formal_training_ready'])

    def test_balanced_reference_is_only_a_candidate(self):
        row = {'problem': 'Find the ratio.', 'ground_truth_solution': r'It is \boxed{\frac{1}{2}}.'}
        result = reference_screen(row)
        self.assertEqual(result['reference_box_candidate'], r'\frac{1}{2}')
        self.assertEqual(result['review_flags'], [])
        self.assertFalse(result['formal_training_ready'])

    def test_overlap_keeps_holdout_and_training_roles_separate(self):
        query = [{'problem_id': 'q', 'question': r'Return your final response within \boxed{}. What is 2+2?'}]
        references = [{'problem_id': 'h', 'question': 'WHAT is 2+2?', 'reference_group': 'eval', 'exclusion_scope': 'holdout'},
                      {'problem_id': 't', 'question': 'What is 2+2?', 'reference_group': 'student', 'exclusion_scope': 'informational'}]
        matches = exact_overlap(query, references)
        self.assertEqual({m['exclusion_scope'] for m in matches}, {'holdout', 'informational'})
        # The shared near matcher excludes these exact normalized matches.
        q = [{**r, 'question': question_key(r['question'])} for r in query]
        refs = [{**r, 'question': question_key(r['question'])} for r in references]
        self.assertEqual(near_matches(q, refs, n=5, threshold=.8, exclude_identical=True), [])
