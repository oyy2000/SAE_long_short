"""Bind real student evaluations to the exact completed training cell."""
from pathlib import Path

from .experiment_io import read_json
from .factorial import file_sha256
from .ncsu_reproduction import verify


def evaluation_cell(student, method, seed, cfg):
    if student not in cfg['students'] or method not in ['base', *cfg['methods']]:
        raise ValueError('Unregistered student evaluation cell')
    if method == 'base':
        if seed is not None:
            raise ValueError('A base evaluation has no training seed')
        return f'{student}/base'
    if seed not in cfg['student_seeds']:
        raise ValueError('Unregistered student training seed')
    return f'{student}/{method}/seed_{seed}'


def evaluation_ratios(method, cfg):
    if method == 'base':
        return [None, *cfg['tokenskip_ratios']]
    if method == 'B6':
        return list(cfg['tokenskip_ratios'])
    if method not in cfg['methods']:
        raise ValueError('Unregistered evaluation method')
    return [None]


def validate_training_identity(marker, metrics, run, student, method, seed, spec):
    """A valid LoRA file alone cannot identify its source, method or seed."""
    if not marker.get('training_complete') or marker.get('synthetic_only'):
        raise ValueError('Evaluation requires completed real student training')
    if not metrics.get('training_complete') or metrics.get('synthetic_only'):
        raise ValueError('Synthetic or incomplete training metrics')
    if (metrics['student'], metrics['method'], metrics['seed']) != (student, method, seed):
        raise ValueError('Adapter training cell identity differs')
    if run['student']['model_name'] != spec['snapshot_path']:
        raise ValueError('Adapter base model differs')
    if run['training']['seed'] != seed or run['training']['data_seed'] != seed:
        raise ValueError('Adapter training seed differs')
    if metrics['optimizer_steps'] <= 0 or metrics['nonzero_lora_b_tensors'] <= 0:
        raise ValueError('Adapter has no completed updates')


def bound_student_model(student, method, seed, cfg):
    key = evaluation_cell(student, method, seed, cfg)
    interface = Path(cfg['sft_interface_root'])
    markers = [interface/'protocol/FROZEN.json', interface/'protocol/SOURCES.json']
    for marker in markers:
        verify(marker)
    parent = read_json(interface/'protocol/frozen_config.json')
    spec = cfg['students'][student]
    if spec != parent['students'][student]:
        raise ValueError('Student changed after its verified interface')
    inventory = read_json(interface/'inputs/model_hashes.json')[student]
    adapter = None
    if method != 'base':
        root = Path(cfg['sft_root'])
        parent_markers = [root/'protocol/FROZEN.json', root/'protocol/SOURCES.json']
        for marker in parent_markers:
            verify(marker)
        training = read_json(root/'protocol/frozen_config.json')
        if training['students'][student] != spec:
            raise ValueError('Training and evaluation student snapshots differ')
        if read_json(root/'inputs/model_hashes.json')[student] != inventory:
            raise ValueError('Training and evaluation model inventories differ')
        adapter = Path(training['checkpoint_root'])/key
        completion = root/'training'/key/'COMPLETE.json'
        verify(completion)
        marker = adapter/'TRAIN_COMPLETE.json'
        doc = verify(marker)
        required = [*parent_markers, root/'encoded'/student/(method+'.jsonl'),
                    root/'training'/key/'run_config.json', adapter/'training_metrics.json',
                    adapter/'adapter_model.safetensors', adapter/'adapter_config.json']
        if any(str(path.resolve()) not in doc['hashes'] for path in required):
            raise ValueError('Adapter marker omits required training provenance')
        if str(marker.resolve()) not in verify(completion)['hashes']:
            raise ValueError('Training result is not bound to this adapter')
        validate_training_identity(doc, read_json(adapter/'training_metrics.json'),
                                   read_json(root/'training'/key/'run_config.json'),
                                   student, method, seed, spec)
        markers += [*parent_markers, completion, marker]
    return {'cell': key, 'student': student, 'method': method, 'seed': seed,
            'model': spec, 'adapter_path': str(adapter) if adapter else None,
            'model_file_hashes': inventory, 'markers': [str(p) for p in markers]}


def verify_model_files(binding):
    for path, digest in binding['model_file_hashes'].items():
        if file_sha256(path) != digest:
            raise ValueError('Evaluation model/tokenizer file changed: '+path)
