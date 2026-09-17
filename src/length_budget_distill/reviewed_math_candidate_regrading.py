"""Migrate complete immutable candidate cohorts to reviewed MATH grading.

All raw generation fields, input order, token IDs, RNG streams and costs are
retained. Only grading/selection is recomputed, with parent-record hashes and
complete original shard manifests. No GPU generation is authorized by this view.
"""
from pathlib import Path
import logging
import shutil

from .experiment_io import read_json
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl
from .reviewed_math_cohort import ADDED_FIELDS, unique_index
from .unified_math_candidates import audit_candidate_grid, grade_candidate, select_candidates
from .typed_math_grading import grade_typed_response

CODE = Path(__file__).resolve().parents[2]
GRADE_FIELDS = {'is_correct', 'status', 'predicted_answer', 'answer_kind', 'component_count',
                'error', 'error_type', 'mode', 'final_box_count'}


def generation_provenance(cfg, root, cohort):
    """Resolve hardware to the verified original generation of a graded view."""
    root = Path(root)
    if not cfg.get('regrading_only'):
        return root, []
    parent = Path(cfg['generation_parent_root'])
    markers = [parent/'protocol/FROZEN.json', parent/'protocol/SOURCES.json',
               parent/'selection'/cohort/'COMPLETE.json']
    for marker in markers:
        verify(marker)
    pcfg = read_json(parent/'protocol/frozen_config.json')
    if pcfg.get('regrading_only') or cfg.get('grading_method') != 'reviewed_math_v1':
        raise ValueError('Expected a reviewed view of original generation')
    for key in ('teacher', 'generation', 'methods', 'shards', 'candidates_per_question',
                'candidate_seed_stride', 'selection_seed', 'decode_policy', 'selection_policy'):
        if cfg[key] != pcfg[key]:
            raise ValueError('Regraded view changed original generation settings: '+key)
    manifest = root/'selection'/cohort/'original_shards.json'
    view_marker = root/'selection'/cohort/'COMPLETE.json'
    view = verify(view_marker)
    if view['hashes'].get(str(manifest.resolve())) != file_sha256(manifest):
        raise ValueError('Original shard manifest is not bound by the graded view')
    entries = read_json(manifest)['shards']
    count = 1 if cohort == 'smoke' else pcfg['shards']
    if len(entries) != count:
        raise ValueError('Incomplete original generation manifest')
    for shard, entry in enumerate(entries):
        marker = parent/'generation'/cohort/f'shard_{shard:02d}'/'COMPLETE.json'
        if entry['shard'] != shard or Path(entry['marker']) != marker or entry['marker_sha256'] != file_sha256(marker):
            raise ValueError('Original generation shard identity changed')
        doc = verify(marker)
        hardware = marker.parent/'hardware.json'
        if entry['hashes'] != doc['hashes'] or doc['hashes'].get(str(hardware.resolve())) != file_sha256(hardware):
            raise ValueError('Original hardware evidence is not bound to its generation')
        markers.append(marker)
    return parent, [*markers, manifest, view_marker]


def regraded_record(original, assessment):
    if 'parent_record_sha256' in original or 'historical_grading' in original:
        raise ValueError('Expected original generation, not an already migrated record')
    return {**{k: v for k, v in original.items() if k not in GRADE_FIELDS}, **assessment,
            'parent_record_sha256': canonical_sha256(original),
            'historical_grading': {k: v for k, v in original.items() if k in GRADE_FIELDS}}


def migrate(config_path):
    from transformers import AutoTokenizer
    rcfg = read_json(config_path)
    if CODE != Path(rcfg['code_root']):
        raise ValueError('Use frozen candidate-regrading source')
    parent, root, asc, cohort = map(Path, (rcfg['parent_root'], rcfg['result_root'], rcfg['math_asc_root'], rcfg['cohort_root']))
    markers = [Path(rcfg['launch_root'])/'FROZEN.json', parent/'protocol/FROZEN.json', parent/'protocol/SOURCES.json',
               asc/'protocol/FROZEN.json', cohort/'COMPLETE.json']
    for marker in markers:
        verify(marker)
    old_cfg = read_json(parent/'protocol/frozen_config.json')
    asc_cfg = read_json(asc/'protocol/frozen_config.json')
    if old_cfg.get('grading_method', 'typed_math_v2') != 'typed_math_v2':
        raise ValueError('Migration expects the original v2 source scores')
    if old_cfg['teacher'] != asc_cfg['teacher'] or asc_cfg['grading']['method'] != 'reviewed_math_v1':
        raise ValueError('Teacher identity or revised grading method differs')
    cfg = {**old_cfg, 'result_root': str(root), 'code_root': str(root/'code'), 'math_asc_root': str(asc),
           'grading_method': 'reviewed_math_v1', 'grading': asc_cfg['grading']['config'],
           'cohort_overlay_root': str(cohort), 'generation_parent_root': str(parent),
           'regrading_only': True, 'allowed_generation_cohorts': rcfg['cohorts']}
    root.mkdir(parents=True, exist_ok=False)
    questions_by_id = unique_index([row for role in ['calibration', 'development', 'student_pool', 'dap_development_reserved']
                                    for row in read_jsonl(cohort/'cohorts'/f'{role}.jsonl')], 'reviewed cohort')
    question_groups = {}
    for name in ['smoke', 'development', 'student_pool']:
        originals = list(read_jsonl(parent/'inputs'/f'{name}.jsonl'))
        annotated = []
        for row in originals:
            source = questions_by_id[row['problem_id']]
            for key, value in source.items():
                if key not in ADDED_FIELDS and row[key] != value:
                    raise ValueError('Candidate input original source field changed: '+key)
            annotated.append({**row, **{k: source[k] for k in ADDED_FIELDS}})
        question_groups[name] = annotated
        write_jsonl(root/'inputs'/f'{name}.jsonl', annotated)
    shutil.copyfile(parent/'inputs/model_hashes.json', root/'inputs/model_hashes.json')
    for folder in ['src', 'scripts', 'configs', 'tests']:
        shutil.copytree(CODE/folder, root/'code'/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'), symlinks=True)
    seal(root/'protocol/SOURCES.json', [p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/FROZEN.json', [Path(config_path), *markers, root/'protocol/SOURCES.json',
                                     root/'protocol/frozen_config.json', *sorted((root/'inputs').iterdir())],
         stage='reviewed_candidate_regrading_protocol', new_generation_allowed=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'], local_files_only=True)
    completed, summaries = [], []
    for name in rcfg['cohorts']:
        original_selection = parent/'selection'/name
        verify(original_selection/'COMPLETE.json')
        originals = list(read_jsonl(original_selection/'predictions.jsonl'))
        questions = question_groups[name]
        if len(originals) != rcfg['expected_records'][name]:
            raise ValueError('Incomplete registered parent predictions')
        audit_candidate_grid(originals, questions, old_cfg, eos_token_id=tokenizer.eos_token_id)
        source_map = unique_index(questions, name)
        shard_manifest = []
        shard_markers = []
        for shard in range(1 if name == 'smoke' else old_cfg['shards']):
            path = parent/'generation'/name/f'shard_{shard:02d}'
            doc = verify(path/'COMPLETE.json')
            shard_markers.append(path/'COMPLETE.json')
            shard_manifest.append({'shard': shard, 'marker': str(path/'COMPLETE.json'),
                                   'marker_sha256': file_sha256(path/'COMPLETE.json'),
                                   'hashes': doc['hashes'], 'generation_unchanged': True})
        revised, changes = [], []
        for old in originals:
            source = source_map[old['problem_id']]
            if tokenizer.decode(old['token_ids'], skip_special_tokens=True) != old['response']:
                raise ValueError('Immutable sampled tokens no longer decode to stored text')
            prior_grade = grade_typed_response(old['response'], source, old_cfg['grading'])
            if any(old[k] != v for k, v in prior_grade.items()):
                raise ValueError('Original candidate grade differs from completed v2 evidence')
            assessment = grade_candidate(cfg, old['response'], source)
            new = regraded_record(old, assessment)
            revised.append(new)
            if old['is_correct'] != new['is_correct']:
                changes.append({'problem_id': old['problem_id'], 'condition': old['condition'],
                                'candidate_index': old['candidate_index'], 'parent_record_sha256': canonical_sha256(old),
                                'historical_grading': new['historical_grading'], 'reviewed_grading': assessment,
                                'question': source['question'], 'reviewed_answer': source['reviewed_answer'],
                                'response': old['response']})
        selected, coverage, common = select_candidates(revised, questions, cfg)
        original_summary = read_json(original_selection/'summary.json')
        out = root/'selection'/name
        out.mkdir(parents=True)
        write_jsonl(out/'predictions.jsonl', revised)
        write_jsonl(out/'grading_changes.jsonl', changes)
        save(out/'original_shards.json', {'shards': shard_manifest, 'complete_parent_shards_verified': True})
        for method, rows in selected.items():
            write_jsonl(out/f'{method}_full_support.jsonl', [dict(rows[pid], selected_baseline=method) for pid in sorted(rows)])
            write_jsonl(out/f'{method}_common_support.jsonl', [dict(rows[pid], selected_baseline=method) for pid in common])
        summary = {**original_summary, 'records': len(revised), 'coverage': coverage,
                   'common_support_ids': common, 'common_support_questions': len(common),
                   'grading_method': 'reviewed_math_v1', 'changed_correctness_labels': len(changes),
                   'original_generation_costs_retained': True, 'new_generation_calls': 0,
                   'formal_training_ready': False, 'generation_parent_root': str(parent)}
        save(out/'summary.json', summary)
        if len(changes) != rcfg['expected_correctness_changes'][name]:
            raise ValueError('Unexpected score changes require review before migration completion')
        completed.append((out, [original_selection/'COMPLETE.json', *shard_markers]))
        summaries.append({'cohort': name, 'questions': len(questions), 'records': len(revised),
                          'correctness_changes': len(changes), 'common_support_questions': len(common)})
    for out, parents in completed:
        seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', *parents, *sorted(out.iterdir())],
             stage='reviewed_candidate_selection_from_immutable_generation', formal_training_ready=False)
    save(root/'summary.json', {'status': 'registered_candidate_cohorts_regraded', 'cohorts': summaries,
                             'generation_calls': 0, 'training_release': False})
    seal(root/'COMPLETE.json', [root/'protocol/FROZEN.json', root/'summary.json',
                              *[out/'COMPLETE.json' for out, _ in completed]],
         stage='reviewed_candidate_cohort_migration', training_release=False)
    logging.info('Migrated complete candidate cohorts: %s', summaries)
