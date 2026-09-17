import copy
import unittest
from length_budget_distill.factorial import canonical_sha256
from length_budget_distill.sae_mechanism_readback import response_regions, probe_positions
from length_budget_distill.sae_readback_analysis import audit_probes


class ProbeAuditTests(unittest.TestCase):
    def fixtures(self):
        text = 'x\nAnswer: 2'
        source = {'q': {'solution': text, 'token_offsets': [(i, i+1) for i in range(len(text))]}}
        pattern = r'(?m)^(?P<marker>Answer):'
        positions = probe_positions(response_regions(text, source['q']['token_offsets'], 'unmodified_generated', pattern)[0])
        rows = []
        for pos, index in positions.items():
            common = {'problem_id':'q', 'position_name':pos, 'processed_response_token_index':index,
                'response_text_sha256':canonical_sha256(text), 'incoming_state_identical_across_branches':True,
                'prefix_tokens':100+index, 'prefix_token_sha256':str(index), 'feature_ids':[0, 1],
                'feature_before':[1., 0.], 'feature_after':[1., 0.], 'forward_kl':0.,
                'actual_delta_fraction':0., 'answer_next_probability':.1, 'eos_next_probability':.2}
            rows += [{**common, 'direction':'zero', 'requested_rho':0.},
                     {**common, 'direction':'sae', 'requested_rho':.1, 'actual_delta_fraction':.1, 'forward_kl':.01}]
        return rows, ['q'], ['sae'], [.1], source, pattern

    def test_complete_grid_accepts_and_missing_or_duplicate_rejects(self):
        args = self.fixtures()
        self.assertEqual(len(audit_probes(*args)), len(args[0]))
        for rows in (args[0][:-1], args[0]+[args[0][0]]):
            with self.assertRaises(ValueError): audit_probes(rows, *args[1:])

    def test_identity_and_numerical_corruption_reject(self):
        args = self.fixtures()
        changes = {'prefix_token_sha256':'wrong', 'feature_before':[2., 0.],
                   'response_text_sha256':'other text', 'requested_rho':.2,
                   'processed_response_token_index':999, 'forward_kl':float('nan'),
                   'answer_next_probability':2., 'incoming_state_identical_across_branches':False}
        for field, value in changes.items():
            with self.subTest(field=field):
                rows = copy.deepcopy(args[0]); rows[1][field] = value
                with self.assertRaises(ValueError): audit_probes(rows, *args[1:])
        rows = copy.deepcopy(args[0]); rows[0]['feature_after'] = [0., 0.]
        with self.assertRaises(ValueError): audit_probes(rows, *args[1:])


if __name__ == '__main__': unittest.main()
