import unittest

from length_budget_distill.sae_body_training import body_positions, paired_trace_rows


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids':[ord(c) for c in text],
                'offset_mapping':[(i,i+1) for i in range(len(text))]}

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


class AnswerFreeSupportTests(unittest.TestCase):
    def setUp(self):
        self.settings={'answer_heading_pattern':r'(?im)^[ \t]*(?P<marker>(?:final[ \t]+)?answer)[ \t]*:',
                       'exclude_last_body_tokens':2,'maximum_native_position':128,
                       'minimum_eligible_positions':3,'exclude_words':['frac','boxed']}
        self.tokenizer=CharacterTokenizer()

    def test_excludes_marker_suffix_and_last_body_positions(self):
        value,why=body_positions(self.tokenizer,'a1b2c3d4e5\nAnswer: 42',self.settings)
        self.assertIsNone(why)
        self.assertEqual(value['eligible_positions'],[0,1,2,3,4,5,6,7,8])
        self.assertLess(max(value['eligible_positions']),value['body_cutoff_token'])

    def test_missing_or_multiple_headings_are_rejected(self):
        for text,reason in [('abcd','missing_heading'),('abcd\nAnswer: 1\nAnswer: 2','multiple_headings')]:
            self.assertEqual(body_positions(self.tokenizer,text,self.settings),(None,reason))

    def test_pairing_keeps_short_and_long_with_other_correct_rollouts(self):
        rows=[{'question_split':'dev','problem_id':'a','analysis_length_label':label}
              for label in ('short','other_correct','long')]
        rows += [{'question_split':'dev','problem_id':'b','analysis_length_label':'short'}]
        self.assertEqual([r['analysis_length_label'] for r in paired_trace_rows(rows)],['short','long'])


if __name__=='__main__':unittest.main()
