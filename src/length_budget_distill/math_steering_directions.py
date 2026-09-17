"""MATH calibration directions for the shared B3/B7 candidate decoder.

The dense control averages within each response and then equally over questions.
The historical SAE decoder direction retains its original BF16 construction;
it is transferred to MATH without relabeling it as a semantic reasoning feature.
"""
from pathlib import Path
import json
import logging
import shutil
import time

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import verify, save, seal, evidence, admission, teacher_bundle
from .baseline_reproduction import record_hardware, grade_prediction
from .sae_mechanism_readback import response_hidden_states, encode, unit

CODE = Path(__file__).resolve().parents[2]


def match_calibration_pairs(pairs, attempts, cfg):
    if len(pairs) != cfg['pairs'] or len({r['problem_id'] for r in pairs}) != len(pairs):
        raise ValueError('Calibration pairs are missing or duplicated')
    result = []
    for pair in pairs:
        if pair['question_role'] != 'calibration' or pair.get('reviewed_gold_override'):
            raise ValueError('Ineligible calibration role or unrepaired reasoning')
        if pair['verbose_teacher_revision'] != cfg['teacher']['revision'] or pair['verbose_teacher'] != cfg['teacher']['model_name']:
            raise ValueError('Calibration teacher differs')
        eligible = [r for r in attempts if r['problem_id'] == pair['problem_id'] and r['solution'] == pair['verbose']
                    and r['eligible_pair'] and r['is_correct'] and not r['hit_max_new_tokens']]
        if len(eligible) != 1:
            raise ValueError('Calibration pair does not identify one eligible sampled trace')
        attempt = eligible[0]
        if attempt['gold_answer'] != pair['answer'] or attempt['generated_tokens'] != pair['verbose_tokens']:
            raise ValueError('Calibration answer or token count differs')
        result.append({**pair, 'verbose_sampled_token_ids':attempt['sampled_token_ids'],
                       'verbose_candidate_index':attempt['candidate_index'], 'verbose_seed':attempt['seed']})
    return result


def prepare(config_path):
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    if root.exists(): raise FileExistsError(root)
    parent = Path(cfg['math_asc_root']); sae_parent = Path(cfg['sae_readback_root'])
    markers = [parent/'protocol/FROZEN.json', parent/'protocol/SOURCES.json', parent/'asc_calibration/COMPLETE.json',
               sae_parent/'protocol/FROZEN.json', sae_parent/'protocol/SOURCES.json']
    for marker in markers: verify(marker)
    pcfg = read_json(parent/'protocol/frozen_config.json'); scfg = read_json(sae_parent/'protocol/frozen_config.json')
    if pcfg['teacher'] != scfg['teacher']: raise ValueError('SAE and MATH teachers differ')
    cfg.update(teacher=pcfg['teacher'], sae=scfg['sae'], grading=pcfg['grading']['config'], code_root=str(root/'code'))
    if cfg['layer_index'] != cfg['sae']['layer_index']: raise ValueError('Dense and SAE must use the same block')
    pairs = match_calibration_pairs(list(read_jsonl(parent/'asc_calibration/pairs.jsonl')),
                                   list(read_jsonl(parent/'asc_calibration/attempts.jsonl')), cfg)
    for pair in pairs:
        for key in ('concise', 'verbose'):
            if not grade_prediction(pcfg, pair[key], pair['answer'], source=pair)['is_correct']:
                raise ValueError('Calibration answer no longer verifies')
    write_jsonl(root/'inputs/pairs.jsonl', pairs)
    save(root/'inputs/sae_hashes.json', evidence([Path(cfg['sae']['checkpoint_path'])]))
    save(root/'inputs/teacher_hashes.json', read_json(parent/'inputs/model_hashes.json')['teacher'])
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder, root/'code'/folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'), symlinks=True)
    seal(root/'protocol/SOURCES.json', [p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/FROZEN.json', [Path(config_path), *markers, root/'protocol/SOURCES.json',
         root/'protocol/frozen_config.json', *sorted((root/'inputs').glob('*'))], stage='MATH_direction_inputs', formal_training_ready=False)


def extract(config_path):
    import torch
    from safetensors.torch import load_file, save_file
    cfg = read_json(config_path); root = Path(cfg['result_root']); out = root/'directions'
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen direction source')
    for manifest in ('sae_hashes.json','teacher_hashes.json'):
        for path, digest in read_json(root/'inputs'/manifest).items():
            if file_sha256(path) != digest: raise ValueError('Changed direction model input: '+path)
    out.mkdir(parents=True, exist_ok=False); admission(cfg); record_hardware(out)
    model, tok = teacher_bundle(cfg); layer = model.model.layers[cfg['layer_index']]
    pairs = list(read_jsonl(root/'inputs/pairs.jsonl')); means = {}; records = []
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.monotonic()
    for pair in pairs:
        record = {'problem_id':pair['problem_id'], 'question_role':'calibration'}
        for name in ('concise','verbose'):
            prompt, response, _ = encode(tok, pair, pair[name])
            if name == 'verbose':
                raw = pair['verbose_sampled_token_ids']
                if not raw or raw[-1] != tok.eos_token_id or tok.eos_token_id in raw[:-1]:
                    raise ValueError('Verbose calibration lacks a valid sampled EOS boundary')
                if tok.decode(raw, skip_special_tokens=True) != pair[name]: raise ValueError('Verbose sampled text differs')
                response = raw[:-1]
                if len(response) != pair['verbose_tokens']: raise ValueError('Verbose token count differs')
            if not response or len(prompt)+len(response) > cfg['maximum_sequence_tokens']:
                raise ValueError('Calibration context is empty or exceeds the registered limit')
            hidden = response_hidden_states(model, layer, prompt+response, len(prompt))
            means[pair['problem_id']+'__'+name] = hidden.float().mean(0).cpu()
            record[name+'_tokens'] = len(response); record[name+'_token_sha256'] = canonical_sha256(response)
        records.append(record); logging.info('MATH direction paired responses %d/%d', len(records), len(pairs))
    differences = torch.stack([means[p['problem_id']+'__concise']-means[p['problem_id']+'__verbose'] for p in pairs])
    dense = differences.double().mean(0).float()
    decoder = load_file(cfg['sae']['checkpoint_path'], device='cpu')['decoder_weight'].to(torch.bfloat16)
    sparse = decoder[cfg['sae']['short_features']].float().sum(0)
    vectors = {'dense_reference_minus_generated':unit(dense), 'sae_short_8':unit(sparse)}
    for i, ids in enumerate(cfg['sae']['random_feature_sets']): vectors['random_sae_'+str(i+1)] = unit(decoder[ids].float().sum(0))
    if any(v.shape != (model.config.hidden_size,) or not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Nonfinite or incompatible steering vector')
    save_file(means, str(out/'paired_response_means.safetensors')); save_file(vectors, str(out/'directions.safetensors'))
    write_jsonl(out/'paired_measurements.jsonl', records)
    torch.cuda.synchronize()
    save(out/'summary.json', {'pairs':len(pairs), 'layer_index':cfg['layer_index'], 'directions':list(vectors),
        'dense_unscaled_norm':float(dense.norm()), 'sae_unscaled_norm':float(sparse.norm()),
        'dense_sae_cosine':float(torch.dot(vectors['dense_reference_minus_generated'],vectors['sae_short_8'])),
        'extraction_seconds':time.monotonic()-start, 'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'definition':'Per-question concise minus verbose response-state mean, then equal-weight question mean; unit normalization.',
        'verbose_tokens':'Actual sampled token IDs excluding EOS; concise reference tokenized separately after the identical native chat prompt.',
        'sae_construction':'Round historical decoder rows to BF16, sum selected rows in FP32, then normalize; no new feature discovery.',
        'claim_boundary':cfg['claim_boundary'], 'formal_training_ready':False})
    seal(out/'COMPLETE.json', [Path(config_path), root/'protocol/FROZEN.json', *sorted(out.glob('*'))],
         stage=cfg.get('direction_stage','MATH_dense_and_SAE_directions'), formal_training_ready=False)
