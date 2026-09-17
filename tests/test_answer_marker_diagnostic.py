"""Tests of boundary and denominator alternatives in the diagnostic."""
import unittest
import numpy as np
from length_budget_distill.answer_marker_diagnostic import region_masks
from length_budget_distill.sae_feature_analysis import paired_feature_statistics
PATTERN=r'(?im)^[ \t]*(?P<marker>(?:final[ \t]+)?answer)[ \t]*:'
class TestAnswerRegions(unittest.TestCase):
    def test_pilot_unnamed_headings_preserve_body_marker_and_answer(self):
        from length_budget_distill.gsm8k_student_pilot import ANSWER_REGION_PATTERN
        for heading in ('Answer:', 'Final Answer:', r'\boxed{'):
            text='Reasoning. '+heading+'42'
            offsets=[(i,i+1) for i in range(len(text))]
            masks,found=region_masks(text,offsets,ANSWER_REGION_PATTERN)
            self.assertTrue(found)
            np.testing.assert_array_equal(masks,[0]*11+[1]*len(heading)+[2]*2)
    def test_no_heading_retains_body_even_with_word_in_sentence(self):
        text='The answer follows a calculation.'
        masks,found=region_masks(text,[(i,i+1) for i in range(len(text))],PATTERN)
        self.assertFalse(found); self.assertTrue((masks==0).all())
    def test_heading_and_suffix_are_separate_with_tabs(self):
        text='2+2=4\n\tAnswer: 4'
        offsets=[(0,5),(5,7),(7,13),(13,14),(14,16)]
        masks,found=region_masks(text,offsets,PATTERN)
        self.assertTrue(found)
        np.testing.assert_array_equal(masks,[0,0,1,1,2])
    def test_constant_single_marker_creates_length_effect(self):
        rows=[{'problem_id':str(q),'analysis_length_label':label} for q in range(3) for label in ['short','long']]
        lengths=np.array([100,200,120,240,140,280],dtype=float)
        original=(10/lengths)[:,None]
        self.assertGreater(paired_feature_statistics(original,rows)['effect'][0],0)
        body=np.zeros_like(original)
        self.assertEqual(paired_feature_statistics(body,rows)['effect'][0],0)
if __name__=='__main__':unittest.main()
