import unittest
class NativeRuntimeTests(unittest.TestCase):
    def test_torch_then_sqlite_nltk_llmlingua(self):
        import torch
        import sqlite3
        import nltk
        from llmlingua import PromptCompressor
        with sqlite3.connect(':memory:') as conn:
            self.assertEqual(conn.execute('select 1').fetchone(),(1,))
        self.assertEqual(nltk.tokenize.wordpunct_tokenize('one two'),['one','two'])
        self.assertTrue(callable(PromptCompressor))
