import unittest
import numpy as np
from length_budget_distill.sae_clean_features import clean_positions
from length_budget_distill.sae_clean_analysis import residualize_from_discovery


class WordTokenizer:
    def __init__(self):
        self.words=[]
    def __call__(self,text,**kwargs):
        import re
        matches=list(re.finditer(r'\S+',text));self.words=[m.group() for m in matches]
        return {'input_ids':list(range(len(matches))),'offset_mapping':[(m.start(),m.end()) for m in matches]}
    def decode(self,ids):return self.words[ids[0]]


class CleanFeatureTests(unittest.TestCase):
    def test_answer_and_tail_are_not_eligible(self):
        settings={'count':4,'exclude_last_tokens':2,'maximum_native_position':20,'exclude_words':['answer','text']}
        _,positions=clean_positions(WordTokenizer(),'think , text reason calculate verify\nAnswer: 42 tail tail',settings)
        self.assertEqual(positions,[0,3,4,5])

    def test_fixed_early_support_does_not_depend_on_extra_tail(self):
        settings={'count':4,'exclude_last_tokens':2,'maximum_native_position':10,'exclude_words':['answer']}
        text='think reason calculate verify extra extra'
        a=clean_positions(WordTokenizer(),text,settings)[1]
        b=clean_positions(WordTokenizer(),text+' more more more\nAnswer: 42',settings)[1]
        self.assertEqual(a,b)
        self.assertEqual(len(a),4)

    def test_confirmation_never_fits_nuisance_coefficients(self):
        nuisance=np.arange(8,dtype=float)[:,None];discovery=np.arange(8)<4
        values=3+2*nuisance
        residual,coef=residualize_from_discovery(values,nuisance,discovery)
        altered=values.copy();altered[~discovery]+=100
        other,other_coef=residualize_from_discovery(altered,nuisance,discovery)
        np.testing.assert_allclose(coef,other_coef)
        np.testing.assert_allclose(residual,0,atol=1e-12)
        np.testing.assert_allclose(other[~discovery],100)


if __name__=='__main__':unittest.main()
