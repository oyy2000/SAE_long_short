"""Independent terminal audit of matched KD/SFT baseline predictions and adapters."""
from pathlib import Path
import math
from .experiment_io import read_json
from .records import read_jsonl
from .ncsu_reproduction import save,seal,verify
from .gsm8k_pilot_summary import audit_marker_graph
from .token_kd_baselines import paired_ids


def audit(config_path):
    review=read_json(config_path);launch=Path(review['launch_root'])
    verify(launch/'FROZEN.json');verify(launch/'SOURCES.json')
    if Path(__file__).resolve().parents[2]!=Path(review['code_root']):raise ValueError('Use frozen verification code')
    cfg=read_json(review['execution_config']);root=Path(cfg['result_root']);checkpoints=Path(cfg['checkpoint_root'])
    from .pilot_storage import configure
    configure(cfg,'analyze')
    out=Path(review['output_root']);out.mkdir(parents=True,exist_ok=False)
    markers=[root/'analysis/COMPLETE.json',root/'protocol/TEST_COMPLETE.json']
    markers.extend(sorted(root.glob('launch/*/SUBMISSION_VERIFIED.json')))
    graph=audit_marker_graph(markers,[root,checkpoints])
    reported=read_json(root/'analysis/metrics.json')['rows']
    summary_index={(r['cohort'],r['method'],r['arm'],r['seed']):r for r in reported}
    if len(summary_index)!=len(reported):raise ValueError('Duplicate aggregate metric cells')
    training_ids=paired_ids(list(read_jsonl(root/'sft/encoded/qwen3b_student/B1.jsonl')))
    if len(training_ids)!=cfg['train_questions']:raise ValueError('Incorrect training size')
    cells=[];training=[];records=0
    for method in cfg['baseline_matrix']:
        data=list(read_jsonl(root/'sft/encoded/qwen3b_student'/f'{method}.jsonl'))
        paired_ids(data,training_ids)
        expected_tokens=sum(sum(t!=-100 for t in r['labels'][1:]) for r in data)*cfg['training']['num_train_epochs']
        for seed in cfg['pilot_seeds']:
            for arm in ('sft','kd'):
                folder=root/'training'/method/arm/f'seed_{seed}'
                metrics=read_json(folder/'metrics.json')
                if set(metrics['problem_exposures'])!=set(training_ids) or any(v!=cfg['training']['num_train_epochs'] for v in metrics['problem_exposures'].values()):
                    raise ValueError('Missing or wrong training exposure')
                if metrics['actual_supervision_tokens_after_causal_shift']!=expected_tokens:raise ValueError('Supervision token mismatch')
                if metrics['optimizer_steps']!=math.ceil(len(data)/cfg['training']['gradient_accumulation_steps'])*cfg['training']['num_train_epochs']:
                    raise ValueError('Optimizer step mismatch')
                adapter=checkpoints/method/arm/f'seed_{seed}'
                for name in ('TRAIN_COMPLETE.json','adapter_model.safetensors','adapter_config.json'):
                    if not (adapter/name).is_file():raise ValueError('Missing complete adapter')
                training.append({'method':method,'arm':arm,'seed':seed,'target_tokens':expected_tokens,'optimizer_steps':metrics['optimizer_steps']})
    for cohort in cfg['evaluation_cohorts']:
        ids=paired_ids(list(read_jsonl(root/'inputs'/f'{cohort}.jsonl')))
        if set(ids)&set(training_ids):raise ValueError('Evaluation contamination')
        conditions=[('base','base',None)]+[(m,a,s) for m in cfg['baseline_matrix'] for a in ('sft','kd') for s in cfg['pilot_seeds']]
        for method,arm,seed in conditions:
            path=root/cohort/('base' if method=='base' else f'{method}/{arm}/seed_{seed}')
            rows=list(read_jsonl(path/'predictions.jsonl'));paired_ids(rows,ids)
            ratio=cfg['tokenskip_ratio'] if method=='B6' else None
            for r in rows:
                if (r['method'],r['arm'],r['seed'],r['cohort'],r['ratio'],r['max_new_tokens'])!=(method,arm,seed,cohort,ratio,cfg['evaluation_cap']):
                    raise ValueError('Prediction cell or decoding identity mismatch')
            result={'cohort':cohort,'method':method,'arm':arm,'seed':seed,'questions':len(rows),
                'accuracy':sum(bool(r['grade']['is_correct']) for r in rows)/len(rows),
                'mean_output_tokens':sum(r['output_tokens'] for r in rows)/len(rows),
                'cap_hit_rate':sum(bool(r['hit_max_new_tokens']) for r in rows)/len(rows)}
            report=summary_index.pop((cohort,method,arm,seed))
            for key in ('accuracy','mean_output_tokens','cap_hit_rate'):
                if not math.isclose(result[key],report[key],abs_tol=1e-12):raise ValueError('Aggregate metrics differ from predictions')
            records+=len(rows);cells.append(result)
    if summary_index:raise ValueError('Unexpected aggregate metric cells')
    expected=(len(cfg['baseline_matrix'])*len(cfg['pilot_seeds'])*2+1)*sum(len(list(read_jsonl(root/'inputs'/f'{c}.jsonl'))) for c in cfg['evaluation_cohorts'])
    if records!=expected:raise ValueError('Incomplete prediction matrix')
    save(out/'hash_checks.json',graph.pop('checks'))
    save(out/'metrics_recomputed.json',{'rows':cells})
    save(out/'audit.json',{**graph,'training_runs':len(training),'training':training,
        'prediction_records':records,'expected_prediction_records':expected,'duplicate_or_missing_records':0,
        'evaluation_cells':len(cells),'metrics_match_predictions':True,'formal_claim_allowed':False,
        'boundary':'Previously observed GSM8K cohorts; B7 historical Answer-associated SAE; MATH excluded'})
    seal(out/'COMPLETE.json',[launch/'FROZEN.json',root/'analysis/COMPLETE.json',*sorted(out.glob('*.json'))],formal_claim_allowed=False)
