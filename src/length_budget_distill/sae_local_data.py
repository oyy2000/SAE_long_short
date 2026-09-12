"""Recover hash-bound rank data and prepare a new length-controlled SAE corpus.

Uses the existing trace encoder, chunk writer, split function and SAE trainer.
The recovered three-rank corpus is explicitly distinct from the historical raw pool.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256
from .sae_data import stable_question_split

ROOT = Path(__file__).resolve().parents[2]


def paths(config):
    return ROOT / config['result_root'], Path(config['runtime_root'])


def jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def evidence(path):
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def verify(item):
    if file_sha256(item['path']) != item['sha256']:
        raise ValueError(f"Changed artifact: {item['path']}")


def recover(config, config_path):
    import pyarrow as pa
    from transformers import AutoTokenizer
    from .sae_activations import encode_replayed_trace
    from .sae_clean_features import clean_positions

    root, runtime = paths(config)
    root.mkdir(parents=True, exist_ok=False)
    runtime.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    (root / 'protocol').mkdir()
    write_json_exclusive(root / 'protocol/frozen_protocol.json', config)
    expected = {value: key for key, value in config['rank_sha256'].items()}
    found = {}
    for pattern in config['recovery_globs']:
        for filename in sorted(glob.glob(pattern, recursive=True)):
            with open(filename, 'rb') as handle:
                rows = pa.ipc.open_stream(handle).read_all().to_pylist()
            if len(rows) != 881:
                continue
            payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in expected or expected[digest] in found:
                continue
            rank = expected[digest]
            destination = root / 'imported' / f'{rank}.jsonl'
            destination.parent.mkdir(exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(payload)
            found[rank] = dict(evidence(destination), arrow=evidence(Path(filename)))
    if set(found) != {'short', 'medium', 'long'}:
        raise RuntimeError(f'Missing hash-verifiable ranks: {set(expected.values()) - set(found)}')
    ranks = {rank: jsonl(item['path']) for rank, item in found.items()}
    ids = [{r['metadata']['problem_id'] for r in ranks[rank]} for rank in ('short', 'medium', 'long')]
    if not (ids[0] == ids[1] == ids[2] and len(ids[0]) == 881):
        raise ValueError('Rank question cohorts differ')
    split = config['question_split']
    assignments = stable_question_split(ids[0], train_count=split['train'], dev_count=split['dev'],
                                        test_count=split['test'], seed=split['seed'])
    teacher = Path(config['teacher']['snapshot_path'])
    if file_sha256(teacher / 'tokenizer.json') != config['teacher']['tokenizer_json_sha256']:
        raise ValueError('Teacher tokenizer mismatch')
    model_files = [teacher / n for n in sorted(set(read_json(teacher / 'model.safetensors.index.json')['weight_map'].values()))]
    model_evidence = [evidence(p) for p in model_files]
    tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
    corpus = []
    for rank in ('short', 'medium', 'long'):
        for row in ranks[rank]:
            meta = row['metadata']
            if not meta['is_correct'] or meta['teacher_model'] != config['teacher']['model_name']:
                raise ValueError('Unexpected source teacher or correctness')
            item = {'corpus_index': len(corpus), 'trace_id': row['id'], 'problem_id': meta['problem_id'],
                    'prompt': row['teacher_prompt'], 'student_prompt': row['prompt'],
                    'solution': row['completion'], 'gold_answer': meta['problem_metadata']['raw_answer'],
                    'analysis_length_label': rank, 'question_split': assignments[meta['problem_id']]}
            encoded = encode_replayed_trace(tokenizer, item, max_sequence_tokens=2048)
            if encoded['truncated_prompt_tokens']:
                raise ValueError('Prompt truncation not allowed in this experiment')
            token_ids, positions = clean_positions(tokenizer, item['solution'], config['clean_tokens'])
            if token_ids != encoded['solution_ids']:
                raise ValueError('Tokenizer alignment failure')
            item.update(solution_token_count=len(token_ids), clean_positions=positions,
                        input_ids=encoded['input_ids'], completion_start=encoded['completion_start'],
                        token_ids=token_ids)
            corpus.append(item)
    if len({r['trace_id'] for r in corpus}) != len(corpus):
        raise ValueError('Repeated selected trace IDs')
    write_jsonl(root / 'corpus.jsonl', corpus)
    write_json_exclusive(root / 'RECOVERY_COMPLETE.json', {
        'status': 'complete', 'formal_claim_allowed': False, 'ranks': found,
        'corpus': evidence(root / 'corpus.jsonl'), 'config': evidence(config_path),
        'source': evidence(Path(__file__)), 'teacher_files': model_evidence,
        'question_count': 881, 'trace_count': len(corpus),
        'note': config['claim_boundary']})
    print(json.dumps({'status': 'recovered', 'trace_count': len(corpus)}), flush=True)


def extract(config, shard):
    import torch
    from transformers import AutoModelForCausalLM
    from .sae_activations import ActivationChunkWriter, padded_batch

    root, runtime = paths(config)
    manifest = read_json(root / 'RECOVERY_COMPLETE.json')
    verify(manifest['corpus'])
    for item in manifest['teacher_files']:
        verify(item)
    corpus = jsonl(root / 'corpus.jsonl')
    settings = config['activation_extraction']
    rows = [r for r in corpus if r['corpus_index'] % settings['trajectory_shards'] == shard]
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(config['teacher']['snapshot_path'],
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    captured = []
    handle = model.model.layers[17].register_forward_hook(
        lambda module, inputs, output: captured.append((output[0] if isinstance(output, tuple) else output).detach()))
    writer = ActivationChunkWriter(runtime / 'activations' / f'shard_{shard}', 17, settings['chunk_tokens'])
    processed = []
    with torch.inference_mode():
        for start in range(0, len(rows), settings['batch_size']):
            batch = rows[start:start + settings['batch_size']]
            ids, mask = padded_batch(batch, model.config.pad_token_id or model.config.eos_token_id)
            captured.clear()
            # The backbone avoids allocating an unnecessary full vocabulary logit tensor.
            model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            if len(captured) != 1:
                raise RuntimeError('Expected exactly one layer capture')
            for i, row in enumerate(batch):
                begin = row['completion_start']
                end = begin + row['solution_token_count']
                writer.append(captured[0][i, begin:end], torch.tensor(row['token_ids']), trace_index=row['corpus_index'])
                processed.append(row['corpus_index'])
            if start % 100 == 0:
                print(json.dumps({'shard': shard, 'traces': len(processed), 'total': len(rows)}), flush=True)
    handle.remove()
    output = writer.finish()
    output.update(status='complete', shard=shard, trace_indices=processed,
                  corpus_sha256=manifest['corpus']['sha256'], config_hash=canonical_sha256(config),
                  source=evidence(Path(__file__)), formal_claim_allowed=False)
    write_json_exclusive(root / f'extraction_shard_{shard}.json', output)


def load_activations(config):
    import torch
    from safetensors.torch import load_file
    root, _ = paths(config)
    corpus = jsonl(root / 'corpus.jsonl')
    chunks = []
    trace_indices = []
    for shard in range(config['activation_extraction']['trajectory_shards']):
        manifest = read_json(root / f'extraction_shard_{shard}.json')
        if manifest['config_hash'] != canonical_sha256(config):
            raise ValueError('Extraction protocol mismatch')
        trace_indices.extend(manifest['trace_indices'])
        for item in manifest['chunks']:
            verify(item)
            chunks.append(load_file(item['path']))
    if sorted(trace_indices) != list(range(len(corpus))):
        raise ValueError('Missing or repeated extraction trace')
    data = {key: torch.cat([c[key] for c in chunks]) for key in chunks[0]}
    counts = torch.bincount(data['trace_indices'].long(), minlength=len(corpus))
    if counts.tolist() != [r['solution_token_count'] for r in corpus]:
        raise ValueError('Missing or repeated token records')
    return corpus, data


def sample(config):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from scipy.stats import spearmanr

    root, runtime = paths(config)
    corpus, data = load_activations(config)
    trace = data['trace_indices'].numpy()
    position = data['positions'].numpy()
    splits = np.array([r['question_split'] for r in corpus])[trace]
    lengths = np.array([r['solution_token_count'] for r in corpus])
    rng = np.random.default_rng(config['sampling']['seed'])
    train_indices = np.flatnonzero(splits == 'train')
    common_train = rng.choice(train_indices, config['sampling']['train_draws'], replace=True)
    values = data['activations'][torch.from_numpy(common_train)].float()
    mean = values.mean(0)
    scale = (3584 / (values - mean).square().sum(1).mean()).sqrt()
    normalizer = runtime / 'normalizer.safetensors'
    save_file({'mean': mean, 'scale': scale.reshape(1)}, str(normalizer))
    del values
    shared = {split: rng.choice(np.flatnonzero(splits == split), config['sampling']['eval_draws'], replace=True)
              for split in ('dev', 'test')}
    outputs = []
    for condition in config['sampling']['conditions']:
        destination = runtime / 'samples' / condition
        destination.mkdir(parents=True, exist_ok=False)
        if condition == 'full_token_uniform':
            indices = common_train
        else:
            support = train_indices if condition == 'full_trace_balanced' else np.flatnonzero((splits == 'train') & (position < 64))
            support_counts = np.bincount(trace[support], minlength=len(corpus))
            weights = 1.0 / support_counts[trace[support]]
            weights /= weights.sum()
            indices = np.random.default_rng(config['sampling']['seed'] + 1).choice(
                support, config['sampling']['train_draws'], replace=True, p=weights)
        selected = {'train': indices, **shared}
        samples = []
        for split, index in selected.items():
            filename = destination / f'{split}.safetensors'
            idx = torch.from_numpy(index)
            save_file({key: value[idx].contiguous() for key, value in data.items()}, str(filename))
            samples.append(dict(evidence(filename), split=split, token_count=len(index)))
        contributions = np.bincount(trace[indices], minlength=len(corpus))
        train_rows = np.array([r['question_split'] == 'train' for r in corpus])
        exposure = float(spearmanr(lengths[train_rows], contributions[train_rows]).statistic)
        manifest = {'config_hash': canonical_sha256(config), 'condition': condition,
                    'normalizer_shared_across_conditions': True, 'length_exposure_spearman': exposure,
                    'trace_contributions': contributions.tolist(),
                    'layers': [{'layer_index': 17, 'samples': samples, 'normalizer_path': str(normalizer),
                                'normalizer_sha256': file_sha256(normalizer)}]}
        write_json_exclusive(destination / 'sample_manifest.json', manifest)
        (destination / 'TOKEN_SAMPLES_COMPLETE').write_text(
            'status=complete\nmanifest_sha256=' + file_sha256(destination / 'sample_manifest.json') + '\n')
        outputs.append(dict(evidence(destination / 'sample_manifest.json'), condition=condition,
                            length_exposure_spearman=exposure))
    write_json_exclusive(root / 'SAMPLING_COMPLETE.json', {'status': 'complete', 'conditions': outputs,
                         'normalizer': evidence(normalizer), 'source': evidence(Path(__file__))})
