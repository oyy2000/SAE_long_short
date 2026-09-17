"""Audit a finished GSM pilot without changing frozen experiment artifacts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def audit_marker_graph(markers, roots):
    """Verify direct bindings and recurse through completion markers in this run."""
    roots=[Path(p).resolve() for p in roots]
    queue=[Path(p) for p in markers];visited=set();cache={};checks=[]
    marker_names={'COMPLETE.json','FROZEN.json','SOURCES.json','TRAIN_COMPLETE.json',
                  'B1_PREPARED.json','TEST_COMPLETE.json','SUBMISSION_VERIFIED.json'}
    while queue:
        marker=queue.pop().resolve()
        if marker in visited:continue
        visited.add(marker);doc=read(marker)
        if doc.get('status')!='complete':raise ValueError('Incomplete marker: '+str(marker))
        for filename,want in doc['hashes'].items():
            path=Path(filename).resolve()
            if path not in cache:cache[path]=digest(path)
            if cache[path]!=want:raise ValueError('Hash mismatch: '+str(path))
            checks.append({'marker':str(marker),'path':str(path),'sha256':want})
            if path.name in marker_names and any(path.is_relative_to(root) for root in roots):queue.append(path)
    return {'verified_marker_count':len(visited),'verified_hash_bindings':len(checks),
            'unique_files_hashed':len(cache),'checks':checks}


def audit(config_path):
    review=read(config_path);cfg=read(review['execution_config'])
    root=Path(cfg['result_root']); checkpoints=Path(cfg['checkpoint_root'])
    out=Path(review['output_root']);out.mkdir(parents=True,exist_ok=False)
    # Reuse the registered job-specific scratch and quota admission.
    from .pilot_storage import configure
    configure(cfg,'analyze')
    queue=[root/'EXPERIMENT_COMPLETE.json',root/'sft/protocol/FROZEN.json',root/'execution_v7/protocol/FROZEN.json']
    protocol=read(root/'evaluation_protocol/protocol.json');selection=read(root/'selection/operating_points.json')
    for method,seed in protocol['cells']:
        if method!='base':queue.append(root/'sft/training/qwen3b_student'/method/f'seed_{seed}'/'COMPLETE.json')
    graph=audit_marker_graph(queue,[root,checkpoints]);checks=graph['checks']
    questions=list(rows(root/'inputs/evaluation.jsonl'));expected_ids={r['problem_id'] for r in questions}
    assert len(expected_ids)==len(questions)==protocol['test_questions']
    assert protocol['locked_source_indices']==list(range(50,1319))
    support=read(root/'sft/inputs/common_support.json');train_ids=support['common_ids']
    assert len(train_ids)==len(set(train_ids))==support['training_questions']
    assert not set(train_ids)&expected_ids
    reported={(r['condition'],r['seed']):r for r in rows(root/'analysis/metrics.jsonl')}
    recomputed=[];training=[];identities={};shards=0
    for method,seed in protocol['cells']:
        values=[]
        for shard in range(cfg['evaluation_shards']):
            part=root/'evaluation'/method/f'seed_{seed}'/f'shard_{shard:02d}'
            batch=list(rows(part/'predictions.jsonl'));expected={q['problem_id'] for q in questions[shard::cfg['evaluation_shards']]}
            assert len(batch)==len(expected) and {r['problem_id'] for r in batch}==expected
            assert all(r['method']==method and r['seed']==seed and r['max_new_tokens']==protocol['max_new_tokens'] for r in batch)
            for r in batch:
                key=(method,seed,r['problem_id']);assert key not in identities
                identities[key]=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
            values.extend(batch);shards+=1
        n=len(values);assert n==len(expected_ids)
        metric={'condition':method,'seed':seed,'questions':n,'correct':sum(bool(r['grade']['is_correct']) for r in values),
                'accuracy':sum(bool(r['grade']['is_correct']) for r in values)/n,
                'mean_output_tokens':sum(r['output_tokens'] for r in values)/n,
                'cap_hit_rate':sum(bool(r['hit_max_new_tokens']) for r in values)/n}
        for k in ('accuracy','mean_output_tokens','cap_hit_rate'):assert math.isclose(metric[k],reported[(method,seed)][k],abs_tol=1e-12)
        recomputed.append(metric)
        if method!='base':
            text=list(rows(root/'sft/text'/f'{method}.jsonl'));assert [r['problem_id'] for r in text]==train_ids
            adapter=checkpoints/'qwen3b_student'/method/f'seed_{seed}'
            for name in ('adapter_model.safetensors','adapter_config.json','TRAIN_COMPLETE.json'):assert (adapter/name).is_file()
            t=read(adapter/'training_metrics.json');assert t['training_complete'] and set(t['problem_exposures'])==set(train_ids)
            assert all(v==cfg['training']['num_train_epochs'] for v in t['problem_exposures'].values())
            assert t['optimizer_steps']==len(train_ids)*cfg['training']['num_train_epochs']//(cfg['training']['per_device_train_batch_size']*cfg['training']['gradient_accumulation_steps'])
            training.append({k:t[k] for k in ('method','seed','optimizer_steps','actual_epoch','observed_sequences','actual_supervision_tokens_after_causal_shift','elapsed_seconds_including_batch_audit','peak_gpu_allocated_mib')})
    merged_count=0
    for r in rows(root/'analysis/all_predictions.jsonl'):
        key=(r['method'],r['seed'],r['problem_id']);want=identities.pop(key)
        assert hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()==want
        merged_count+=1
    assert not identities
    selected=[];by_method={r['condition']:r for r in recomputed}
    for family,method in [('base','base'),*selection['selected_conditions'].items()]:
        selected.append({'baseline':family,**by_method[method],'teacher_tokens':selection['teacher_lengths'].get(method)})
    result={'status':'complete','utc':datetime.now(timezone.utc).isoformat(),'job_id':os.environ.get('SLURM_JOB_ID'),
            'verified_marker_count':graph['verified_marker_count'],'verified_hash_bindings':graph['verified_hash_bindings'],'unique_files_hashed':graph['unique_files_hashed'],
            'complete_training_runs':len(training),'complete_evaluation_shards':shards,'evaluation_cells':len(recomputed),
            'test_questions':len(expected_ids),'unique_training_questions':len(train_ids),'prediction_records':merged_count,
            'duplicate_or_missing_predictions':0,'metrics_match_saved_analysis':True,'repeat_gate_passed':selection['repeat_gate_passed'],
            'selected_conditions':selected,'training':training,'formal_claim_allowed':False,
            'scope':'Pilot marker graph, direct external input bindings, adapters, training exposures, shard identity and saved metrics. No new grading or model inference.'}
    for name,value in [('audit.json',result),('hash_checks.json',checks),('metrics_recomputed.json',recomputed)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    files=[Path(config_path),Path(__file__),*out.glob('*.json')]
    (out/'COMPLETE.json').write_text(json.dumps({'status':'complete','hashes':{str(p.resolve()):digest(p) for p in files},'formal_claim_allowed':False},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('training','selected_conditions')}),flush=True)
