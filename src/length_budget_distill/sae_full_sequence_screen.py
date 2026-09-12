"""Re-screen sealed SAE evidence without requiring an early-prefix sign.

Historical `confirmed` flags and completion markers remain unchanged. This is a
versioned reinterpretation of existing statistics, not a new SAE training run.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import platform
import sys
import importlib.metadata

from .experiment_io import read_json, write_json_exclusive
from .factorial import file_sha256, canonical_sha256
from .legacy_replication import marker_read, marker_write


def full_sequence_decision(candidate, selection):
    """Exactly the original confirmation rule minus the early-sign clause."""
    if selection['require_first64_direction']:
        raise ValueError('This protocol specifically removes the early-sign eligibility clause.')
    direction = candidate['direction']
    if direction not in ('short', 'long'):
        raise ValueError(f'Unknown feature direction: {direction}')
    expected = 1 if direction == 'short' else -1
    metric = candidate['metrics'][selection['primary_metric']]['test']
    effect = float(metric['paired_d'])
    p = float(candidate['confirmation_holm_p_value'])
    finite = math.isfinite(effect) and math.isfinite(p) and 0 <= p <= 1
    checks = {'finite_statistics': finite,
              'full_sequence_direction': finite and expected * effect > 0,
              'holm_significance': finite and p <= selection['confirmation_holm_alpha'],
              'effect_size': finite and abs(effect) >= selection['minimum_test_absolute_paired_d']}
    return {'feature_id': candidate['feature_id'], 'direction': direction,
            'original_confirmed': bool(candidate['confirmed']),
            'full_sequence_confirmed': all(checks.values()), 'checks': checks,
            'early_direction_diagnostic_only': candidate.get('first_64_direction_replicated'),
            'first64_used_for_eligibility': False}


def stable_features(summary, candidates_by_run, config):
    """Keep three-way mutual matching and both original numeric thresholds."""
    result = []
    decisions = {}
    selection = config['selection']
    seeds = config['seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('Three distinct SAE seeds are required.')
    for key, candidates in candidates_by_run.items():
        ids = [int(r['feature_id']) for r in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate candidate feature IDs.')
        decisions[key] = {int(r['feature_id']): full_sequence_decision(r, selection) for r in candidates}
    for condition in config['conditions']:
        pair_maps = {}
        for a,b in itertools.combinations(seeds, 2):
            eligible = {}
            for r in summary['pair_matches'][f'{condition}__{a}_{b}']:
                cosine, correlation = float(r['decoder_cosine']), float(r['test_trace_activation_correlation'])
                if (r['mutual_nearest_neighbor'] and r['same_direction']
                    and math.isfinite(cosine) and math.isfinite(correlation)
                    and cosine >= selection['minimum_decoder_cosine']
                    and correlation >= selection['minimum_activation_correlation']):
                    eligible[int(r['left_feature_id'])] = r
            pair_maps[(a,b)] = eligible
        a,b,c = seeds
        for fid, ab in pair_maps[(a,b)].items():
            ac = pair_maps[(a,c)].get(fid)
            bc = pair_maps[(b,c)].get(int(ab['right_feature_id']))
            if not ac or not bc or int(ac['right_feature_id']) != int(bc['right_feature_id']):
                continue
            mapped = {a:fid, b:int(ab['right_feature_id']), c:int(ac['right_feature_id'])}
            ds = [decisions[(condition,s)][f] for s,f in mapped.items()]
            if all(d['full_sequence_confirmed'] for d in ds) and len({d['direction'] for d in ds}) == 1:
                result.append({'condition': condition, 'direction': ds[0]['direction'],
                               'feature_ids_by_seed': mapped,
                               'all_originally_confirmed': all(d['original_confirmed'] for d in ds),
                               'first64_used_for_eligibility': False,
                               'pair_evidence': [ab,ac,bc]})
    return result, decisions


def prerequisites(config, source_root=None):
    parent = Path(source_root or config['parent_root'])
    paths = [('screen', parent/'sae/analysis/sae_summary.json'),
             ('screen', parent/'sae/SAE_TRAINING_AND_SCORING_COMPLETE'),
             ('generation', parent/'sae/corpus/mixed_trajectories.jsonl')]
    for condition in config['conditions']:
        for seed in config['seeds']:
            root = parent/f'sae/seed_{seed}'
            paths.append(('screen', root/'protocol/frozen_protocol.json'))
            paths.extend([('screen', root/f'condition_scores/{condition}/{name}') for name in
                          ['discovered_features.json', 'scoring_summary.json', 'CONDITION_SCORING_COMPLETE']])
            paths.extend([('generation', root/f'checkpoints/{condition}/sae_model.safetensors'),
                          ('generation', root/f'sae_training/{condition}/SAE_TRAINING_COMPLETE')])
    snapshot = Path(config['teacher']['snapshot_path'])
    paths.extend([('generation',snapshot/'config.json'), ('generation',snapshot/'tokenizer.json'),
                  ('generation',snapshot/'model.safetensors.index.json')])
    index = snapshot/'model.safetensors.index.json'
    if index.is_file():
        paths.extend(('generation',snapshot/f) for f in sorted(set(read_json(index)['weight_map'].values())))
    return [{'stage': stage, 'path': str(path), 'available': path.is_file()} for stage,path in paths]


def preflight(config, source_root=None):
    checks = prerequisites(config, source_root)
    versions = {}
    for package in ['torch','transformers','peft','datasets','trl','safetensors']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'status': 'ready' if all(r['available'] for r in checks) else 'blocked_missing_inputs',
            'node': platform.node(), 'python': sys.executable, 'versions': versions,
            'expected_runtime_python': config['python'], 'prerequisites': checks,
            'screen_ready': all(r['available'] for r in checks if r['stage']=='screen'),
            'generation_inputs_ready': all(r['available'] for r in checks if r['stage']=='generation'),
            'legacy_trl_available': versions['trl']=='0.9.6',
            'config_sha256': canonical_sha256(config),
            'intervention_launched': False, 'student_training_launched': False}


def rescreen(config, source_root=None):
    parent = Path(source_root or config['parent_root'])
    missing = [r['path'] for r in prerequisites(config, source_root) if r['stage']=='screen' and not r['available']]
    if missing:
        raise FileNotFoundError('Missing original SAE screening evidence: ' + ', '.join(missing))
    summary_path = parent/'sae/analysis/sae_summary.json'
    summary = read_json(summary_path)
    marker = marker_read(parent/'sae/SAE_TRAINING_AND_SCORING_COMPLETE')
    if marker.get('status') != 'complete' or marker.get('summary_sha256') != file_sha256(summary_path):
        raise ValueError('Parent SAE summary failed its registered hash check.')
    candidates, inputs = {}, [summary_path, parent/'sae/SAE_TRAINING_AND_SCORING_COMPLETE']
    for condition in config['conditions']:
        for seed in config['seeds']:
            score = parent/f'sae/seed_{seed}/condition_scores/{condition}'
            mark = marker_read(score/'CONDITION_SCORING_COMPLETE')
            metadata = read_json(score/'scoring_summary.json')
            if mark.get('status') != 'complete' or mark.get('summary_sha256') != file_sha256(score/'scoring_summary.json'):
                raise ValueError('Parent condition-score hash mismatch.')
            protocol_path = parent/f'sae/seed_{seed}/protocol/frozen_protocol.json'
            protocol = read_json(protocol_path)
            if metadata['config_hash'] != canonical_sha256(protocol):
                raise ValueError('Parent scoring protocol hash mismatch.')
            evaluation = protocol['sampling_ablation']['evaluation']
            for key in ['primary_metric','confirmation_holm_alpha','minimum_test_absolute_paired_d']:
                if config['selection'][key] != evaluation[key]:
                    raise ValueError(f'Only the early-direction criterion may change; mismatch in {key}.')
            feature_path = score/'discovered_features.json'
            evidence = [v for v in metadata['artifacts'].values() if Path(v['path']).name=='discovered_features.json']
            if len(evidence)!=1 or evidence[0]['sha256'] != file_sha256(feature_path):
                raise ValueError('Parent candidate statistics hash mismatch.')
            candidates[(condition,seed)] = read_json(feature_path)['candidates']
            inputs.extend([feature_path,score/'scoring_summary.json',score/'CONDITION_SCORING_COMPLETE',protocol_path])
    stable, decisions = stable_features(summary,candidates,config)
    out = Path(config['output_root'])/'screen'
    out.mkdir(parents=True,exist_ok=False)
    report = {'status':'complete','protocol':config,'stable_features':stable,
              'original_stable_feature_count':len(summary['stable_features']),
              'candidate_decisions':{f'{c}__{s}':list(ds.values()) for (c,s),ds in decisions.items()},
              'input_evidence':[{'path':str(p),'sha256':file_sha256(p)} for p in inputs],
              'source_sha256':file_sha256(Path(__file__)),
              'formal_claim_allowed':False,'intervention_launched':False,
              'interpretation':'Whole-sequence association only. Early differences remain diagnostics; causal and student gains untested.'}
    write_json_exclusive(out/'full_sequence_screen.json',report)
    marker_write(out/'FULL_SEQUENCE_SCREEN_COMPLETE',{'status':'complete',
        'summary_sha256':file_sha256(out/'full_sequence_screen.json'),'stable_feature_count':len(stable)})
    return report
