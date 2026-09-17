"""Resume audited MATH source preparation under administrator-designated scratch."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

from .experiment_io import read_json
from .ncsu_reproduction import save, seal, verify
from .pilot_storage import validate_scratch, quota_capacity, choose_location
from .records import read_jsonl, write_jsonl

CODE=Path(__file__).resolve().parents[2]


def archive_raw_candidates(cfg):
    """Audit immutable sampled records without certifying inconsistent old scores."""
    from transformers import AutoTokenizer
    from .unified_math_candidates import audit_candidate_grid
    from .sae_generation_analysis import audit_batches
    parent=Path(cfg['parent_root']);out=Path(cfg['archive_root'])
    out.mkdir(parents=True,exist_ok=False)
    pcfg=read_json(parent/'protocol/frozen_config.json')
    tokenizer=AutoTokenizer.from_pretrained(pcfg['teacher']['snapshot_path'],local_files_only=True)
    sources=list(read_jsonl(parent/'inputs/student_pool.jsonl'))
    rows=[];bindings=[parent/'protocol/FROZEN.json',parent/'protocol/SOURCES.json',parent/'inputs/student_pool.jsonl']
    costs=[]
    for shard in range(32):
        folder=parent/f'generation/student_pool/shard_{shard:02d}'
        marker=folder/'COMPLETE.json';verify(marker);bindings.append(marker)
        part=list(read_jsonl(folder/'predictions.jsonl'))
        questions=[q for i,q in enumerate(sources) if i%32==shard]
        audit_candidate_grid(part,questions,pcfg,eos_token_id=tokenizer.eos_token_id)
        source_map={q['problem_id']:q for q in questions}
        for row in part:
            if tokenizer.decode(row['token_ids'],skip_special_tokens=True)!=row['response']:
                raise ValueError('Immutable token/text mismatch')
            if row['question']!=source_map[row['problem_id']]['question']:
                raise ValueError('Immutable question identity mismatch')
        costs.append({'shard':shard,'seconds_by_method':audit_batches(part,list(read_jsonl(folder/'batches.jsonl')))})
        rows.extend(part)
    audit_candidate_grid(rows,sources,pcfg,eos_token_id=tokenizer.eos_token_id)
    write_jsonl(out/'predictions.jsonl',rows)
    save(out/'summary.json',{'records':len(rows),'questions':len(sources),'shards':32,
        'token_text_and_candidate_grid_verified':True,'historical_grades_preserved':True,
        'historical_grade_reproducibility_certified':False,'student_selection_complete':False,
        'training_release':False,'costs':costs})
    seal(out/'COMPLETE.json',[Path(cfg['launch_root'])/'FROZEN.json',*bindings,*sorted(out.glob('*'))],
        stage='immutable_raw_candidate_archive_for_scoring_diagnosis',training_release=False)
    return out/'COMPLETE.json'


def run(config_path,stage):
    cfg=read_json(config_path);root=Path(cfg['launch_root'])
    verify(root/'FROZEN.json');verify(root/'SOURCES.json')
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen MATH preparation code')
    scratch=validate_scratch(cfg['runtime']['storage_policy'],os.environ)
    available,reports=quota_capacity(cfg['runtime']['storage_policy'],scratch)
    free=shutil.disk_usage(scratch).free
    if available is not None:free=min(free,available)
    parent=Path(cfg['parent_root'])
    source_files=sorted((parent/'generation/student_pool').glob('*/predictions.jsonl'))
    if len(source_files)!=32:raise ValueError('Incomplete source shards')
    source_bytes=sum(p.stat().st_size for p in source_files)
    # No model/checkpoint copies: cache plus a full JSON serialization buffer with 2x margin.
    temp_required=2*(64*2**20+source_bytes)
    publication_required=2*(4*source_bytes+128*2**20)
    choose_location(free,shutil.disk_usage(root).free,temp_required,publication_required)
    out=root/'stages'/stage;out.mkdir(parents=True,exist_ok=False)
    save(out/'storage.json',{'stage':stage,'scratch':str(scratch),'source_bytes':source_bytes,
        'temporary_required_bytes':temp_required,'publication_required_bytes':publication_required,
        'scratch_available_including_quota':free,'quota_reports':reports,
        'rationale':'No GPU, model copies, or checkpoints. Allow complete JSON staging, four selected/changed-output copies, code snapshot, and 2x margin.',
        'job_id':os.environ['SLURM_JOB_ID'],'TMPDIR':os.environ['TMPDIR'],'TMP':os.environ['TMP'],'TEMP':os.environ['TEMP']})
    if stage=='merge':
        for shard in range(32):verify(parent/f'generation/student_pool/shard_{shard:02d}/COMPLETE.json')
        verify(parent/'protocol/FROZEN.json');verify(parent/'protocol/SOURCES.json')
        marker=archive_raw_candidates(cfg);verify(marker)
    elif stage=='impact':
        verify(root/'stages/merge/COMPLETE.json')
        from .reviewed_math_prediction_impact import audit
        audit(config_path)
        marker=Path(cfg['result_root'])/'COMPLETE.json';verify(marker)
    else:raise ValueError(stage)
    seal(out/'COMPLETE.json',[root/'FROZEN.json',out/'storage.json',marker],training_release=False)
