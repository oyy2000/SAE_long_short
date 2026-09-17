import json
from pathlib import Path
import tempfile
import unittest
import os
import pwd
from unittest.mock import patch
from length_budget_distill.pilot_storage import choose_location, estimate, validate_scratch

class StorageTests(unittest.TestCase):
    def test_small_workload_does_not_require_gib(self):
        self.assertEqual(choose_location(40*1024**2,1000*1024**2,32*1024**2,32*1024**2),'designated_scratch')
    def test_no_fallback_even_when_project_has_room(self):
        with self.assertRaises(RuntimeError):choose_location(1,300,100,150)
    def test_combined_staging_and_publication(self):
        with self.assertRaises(RuntimeError):choose_location(1,200,100,150)
    def test_no_publication_capacity(self):
        with self.assertRaises(RuntimeError):choose_location(1000,10,20,100)
    def test_system_tmp_or_mismatched_environment_rejected(self):
        user=pwd.getpwuid(os.getuid()).pw_name
        root=Path('/share/jekml')/user/'tmp'
        job=root/f'{user}-phase13-frozen-123'
        env=dict(SLURM_JOB_ID='123',TMPDIR=str(job),TMP=str(job),TEMP=str(job))
        with patch.object(Path,'is_dir',return_value=True):
            self.assertEqual(validate_scratch({'scratch_root':str(root)},env),job.resolve())
            for key in ('TMPDIR','TMP','TEMP'):
                with self.assertRaises(RuntimeError):validate_scratch({'scratch_root':str(root)},{**env,key:'/var/tmp'})
            with self.assertRaises(RuntimeError):validate_scratch({'scratch_root':'/tmp'},env)
    def test_adapter_budget_from_model_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();(model/'config.json').write_text(json.dumps(dict(hidden_size=100,intermediate_size=200,num_hidden_layers=2,num_key_value_heads=2,num_attention_heads=4)))
            p=root/'sft/encoded/qwen3b_student/B0.jsonl';p.parent.mkdir(parents=True);p.write_text('x'*1000)
            cfg=dict(result_root=str(root),students={'qwen3b_student':{'snapshot_path':str(model)}},lora={'r':8},runtime={'storage_policy':dict(small_cache_bytes=1024,encoded_copies=4,adapter_copies=3,safety_factor=2)})
            first=estimate(cfg,'train','B0');self.assertGreater(first['temporary_required_bytes'],4000)
            cfg['result_root']=str(root/'sft');self.assertEqual(first,estimate(cfg,'train','B0'))
            cfg['lora']['r']=16;self.assertGreater(estimate(cfg,'train','B0')['temporary_required_bytes'],first['temporary_required_bytes'])
