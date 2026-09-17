import unittest
from length_budget_distill.dap_paired_sources import question_key,conversation_pair


class DAPSourceTests(unittest.TestCase):
    def test_normalization_preserves_different_mathematical_questions(self):
        self.assertEqual(question_key(r'Return your final response within \boxed{}. Solve x+1=3.'),question_key('Solve x+1=3.'))
        self.assertNotEqual(question_key('Solve x+1=3.'),question_key('Solve x+1=4.'))
        self.assertNotEqual(question_key('Compute a+b.'),question_key('Compute a-b.'))

    def test_full_assistant_trace_is_retained(self):
        row={'problem':'Q','solution':'short final solution only','messages':[{'role':'user','content':'Q'},
             {'role':'assistant','content':'Full thought and full solution'}]}
        self.assertEqual(conversation_pair(row,'smallthoughts'),('Q','Full thought and full solution'))
        with self.assertRaises(ValueError):conversation_pair({**row,'problem':'Different Q'},'smallthoughts')
        row['messages'].append({'role':'user','content':'Extra turn'})
        with self.assertRaises(ValueError):conversation_pair(row,'smallthoughts')


if __name__=='__main__':unittest.main()
