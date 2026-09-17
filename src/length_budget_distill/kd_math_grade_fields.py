"""Locate historical merge failures using already recomputed per-record grades."""
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from .records import read_jsonl,write_jsonl
from .experiment_io import read_json
from .factorial import canonical_sha256
from .ncsu_reproduction import save,seal,verify


def run(config_path):
    cfg=read_json(config_path);launch=Path(cfg['launch_root'])
    verify(launch/'FROZEN.json');verify(launch/'SOURCES.json')
    if Path(__file__).resolve().parents[2]!=Path(cfg['code_root']):raise ValueError('Use frozen diagnostic code')
    from .pilot_storage import configure
    configure(cfg,'analyze')
    archive=Path(cfg['archive_root']);impact=Path(cfg['impact_root']);out=Path(cfg['output_root'])
    verify(archive/'COMPLETE.json');verify(impact/'COMPLETE.json');out.mkdir(parents=True,exist_ok=False)
    changed=[];counts=Counter();n=0
    for original,graded in zip_longest(read_jsonl(archive/'predictions.jsonl'),read_jsonl(impact/'raw_student_pool.jsonl')):
        if original is None or graded is None or canonical_sha256(original)!=graded['source_record_sha256']:
            raise ValueError('Incomplete or unaligned audited predictions')
        diff={k:{'stored':original.get(k),'recomputed':v} for k,v in graded['recomputed_v2_grading'].items() if original.get(k)!=v}
        if diff:
            changed.append({'problem_id':original['problem_id'],'record_key':graded['record_key'],
                'source_record_sha256':graded['source_record_sha256'],'differences':diff,
                'response_tail':original['response'][-1600:]})
            counts.update(diff)
        n+=1
    write_jsonl(out/'field_differences.jsonl',changed)
    save(out/'summary.json',{'records':n,'records_with_field_differences':len(changed),
        'changed_fields':dict(counts),'correctness_label_changes':counts['is_correct'],
        'no_scores_or_source_text_modified':True,'training_release':False})
    seal(out/'COMPLETE.json',[launch/'FROZEN.json',archive/'COMPLETE.json',impact/'COMPLETE.json',*sorted(out.glob('*'))],training_release=False)
