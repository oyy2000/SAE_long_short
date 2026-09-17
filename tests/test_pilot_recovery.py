import tempfile
from pathlib import Path
import unittest
from length_budget_distill.pilot_recovery import arguments,archive_incomplete,retry_paths,retry_route
class RecoveryTests(unittest.TestCase):
    def test_exact_stage_arguments(self):
        self.assertEqual(arguments(['sbatch','--stage','generate','--shard','2','--method','student_pool'])['shard'],'2')
    def test_archive_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'partial';p.mkdir();(p/'predictions.jsonl').write_text('partial\n');out=root/'attempt';out.mkdir()
            archive_incomplete([p],out)
            self.assertFalse(p.exists());self.assertEqual((out/'preserved_outputs/0/predictions.jsonl').read_text(),'partial\n')
    def test_completed_artifact_blocks_archive(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'done';p.mkdir();(p/'COMPLETE.json').write_text('{}')
            with self.assertRaises(ValueError):archive_incomplete([p],root/'attempt')
            self.assertTrue(p.exists())
    def test_cpu_or_unknown_stage_requires_diagnosis(self):
        with self.assertRaises(ValueError):retry_paths({'result_root':'/tmp'},{'stage':'advance'})

    def test_cuda_failure_changes_gpu_family(self):
        self.assertEqual(retry_route('FAILED','l40s','CUDA error: an illegal memory access'),'h200')
        self.assertEqual(retry_route('FAILED','h200','CUDA error: an illegal memory access'),'h100')
    def test_configuration_failure_is_not_blindly_retried(self):
        with self.assertRaises(ValueError):retry_route('FAILED','h200','ValueError: changed input hash')
