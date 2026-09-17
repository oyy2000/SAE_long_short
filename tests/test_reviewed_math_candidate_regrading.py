import unittest
import json
from pathlib import Path
import tempfile

from length_budget_distill.reviewed_math_candidate_regrading import regraded_record, GRADE_FIELDS, generation_provenance
from length_budget_distill.unified_math_candidates import grade_candidate, require_same_grading_method
from length_budget_distill.ncsu_reproduction import save, seal
from length_budget_distill.factorial import file_sha256


class CandidateRegradingTest(unittest.TestCase):
    def test_preserves_generation_and_removes_stale_error_fields(self):
        original = {'problem_id': 'p', 'response': 'trace', 'token_ids': [1, 2], 'seed': 17,
                    'generated_tokens': 1, 'amortized_generation_wall_seconds': 2.5,
                    'is_correct': False, 'status': 'grading_error', 'error': 'old error'}
        revised = regraded_record(original, {'is_correct': True, 'status': 'graded_reviewed'})
        for key, value in original.items():
            if key not in GRADE_FIELDS:
                self.assertEqual(revised[key], value)
        self.assertNotIn('error', revised)
        self.assertEqual(revised['historical_grading']['error'], 'old error')
        self.assertFalse(original['is_correct'])

    def test_backend_mismatch_cannot_silently_use_old_targets(self):
        with self.assertRaisesRegex(ValueError, 'backends differ'):
            require_same_grading_method({}, {'grading_method': 'reviewed_math_v1'})
        source = {'answer': '3/2', 'reviewed_answer': {'mode': 'all', 'kind': 'symbolic', 'answers': ['-3', '3/2']}}
        with self.assertRaisesRegex(ValueError, 'declared grading backend'):
            grade_candidate({'grading': {}}, r'\boxed{3/2}', source)


class GenerationProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)/'reviewed';self.parent=Path(self.temporary.name)/'original'
        original={'teacher':{'revision':'fixed'},'generation':{'seed':17},'methods':{'B1':{}},'shards':1,
                  'candidates_per_question':4,'candidate_seed_stride':100003,'selection_seed':17,
                  'decode_policy':{},'selection_policy':'shortest-correct'}
        self.cfg={**original,'grading_method':'reviewed_math_v1','regrading_only':True,'generation_parent_root':str(self.parent)}
        config=self.parent/'protocol/frozen_config.json';save(config,original)
        seal(self.parent/'protocol/FROZEN.json',[config]);seal(self.parent/'protocol/SOURCES.json',[config])
        path=self.parent/'generation/student_pool/shard_00';self.hardware=path/'hardware.json'
        save(self.hardware,{'inventory_csv':'NVIDIA H100'});marker=path/'COMPLETE.json';seal(marker,[self.hardware])
        seal(self.parent/'selection/student_pool/COMPLETE.json',[marker])
        self.manifest=self.root/'selection/student_pool/original_shards.json'
        doc=json.loads(marker.read_text())
        save(self.manifest,{'shards':[{'shard':0,'marker':str(marker),'marker_sha256':file_sha256(marker),'hashes':doc['hashes']}]})
        self.view_marker=self.root/'selection/student_pool/COMPLETE.json';seal(self.view_marker,[self.manifest])

    def test_preserved_hardware_is_resolved_without_copying_generation(self):
        original,bindings=generation_provenance(self.cfg,self.root,'student_pool')
        self.assertEqual(original,self.parent);self.assertIn(self.view_marker,bindings)
        self.assertFalse((self.root/'generation').exists())

    def test_original_hardware_change_and_sampler_change_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'generation settings'):
            generation_provenance({**self.cfg,'generation':{'seed':42}},self.root,'student_pool')
        self.hardware.write_text('{"inventory_csv":"NVIDIA H200"}')
        with self.assertRaisesRegex(ValueError,'Changed artifact'):
            generation_provenance(self.cfg,self.root,'student_pool')

    def test_resealed_empty_manifest_cannot_hide_original_shards(self):
        self.manifest.write_text('{"shards":[]}');self.view_marker.unlink();seal(self.view_marker,[self.manifest])
        with self.assertRaisesRegex(ValueError,'Incomplete original generation'):
            generation_provenance(self.cfg,self.root,'student_pool')


if __name__=='__main__':unittest.main()
