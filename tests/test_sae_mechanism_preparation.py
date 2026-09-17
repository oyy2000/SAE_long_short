import unittest
from fractions import Fraction
from length_budget_distill.sae_mechanism_preparation import arithmetic_value,reference_variants,partition_questions


class MechanismInputTests(unittest.TestCase):
    def test_mutation_is_wrong_with_identical_final_answer(self):
        rows={r['variant']:r for r in reference_variants('Half is 48/2 = <<48/2=24>>24. Total is 72.\n#### 72')}
        good,bad=rows['reference'],rows['wrong_calculation']
        self.assertTrue(good['text'].endswith('Answer: 72'))
        self.assertTrue(bad['text'].endswith('Answer: 72'))
        self.assertEqual(sum(a!=b for a,b in zip(good['text'],bad['text'])),1)
        edit=bad['equation_mutation']
        self.assertEqual(arithmetic_value(edit['expression']),Fraction(edit['original']))
        self.assertNotEqual(arithmetic_value(edit['expression']),Fraction(edit['replacement']))

    def test_no_false_verification_of_inconsistent_annotation(self):
        rows=reference_variants('2+2=<<2+2=5>>5.\n#### 5')
        self.assertNotIn('wrong_calculation',{r['variant'] for r in rows})
        self.assertEqual(arithmetic_value('1/3+2/3'),1)
        with self.assertRaises(ValueError):arithmetic_value('__import__("os")')

    def test_alignment_reconstructs_both_texts(self):
        rows=reference_variants('2+2=<<2+2=4>>4.\n#### 4');base=rows[0]['text']
        for row in rows:
            self.assertEqual(''.join(base[s['base_span'][0]:s['base_span'][1]] for s in row['base_alignment']),base)
            self.assertEqual(''.join(row['text'][s['variant_span'][0]:s['variant_span'][1]] for s in row['base_alignment']),row['text'])
        variants={r['variant']:r['text'] for r in rows}
        self.assertEqual(variants['neutral_note'].replace('Entry.','Answer.',1),variants['answer_note'])

    def test_split_is_disjoint_stable_and_rejects_duplicates(self):
        rows=[{'problem_id':str(i),'question':f'Question {i}'} for i in range(30)]
        smoke,cohort=partition_questions(rows,{'discovery':10,'dev':5,'confirmation':5},3,17)
        self.assertEqual((smoke,cohort),partition_questions(list(reversed(rows)),{'discovery':10,'dev':5,'confirmation':5},3,17))
        self.assertFalse({r['problem_id'] for r in smoke}&{r['problem_id'] for r in cohort})
        with self.assertRaises(ValueError):partition_questions(rows+[rows[0]],{'dev':2},1,17)
        with self.assertRaises(ValueError):partition_questions(rows,{'dev':30},1,17)


if __name__=='__main__':unittest.main()
