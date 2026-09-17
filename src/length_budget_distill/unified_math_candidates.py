"""Common-teacher mathematical candidate generation and selection.

Reuse the paired cached decoder and earliest-index exact-text deduplication.
Development outputs are never student supervision. Full student-pool generation
requires its own explicit frozen-cohort authorization in the execution config.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import logging
import math
import random
import shutil
import time

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify, admission, teacher_bundle
from .baseline_reproduction import record_hardware, grade_prediction
from .ncsu_multi_answer import unique_correct
from .sae_norm_intervention import NoInterventionController, generate_condition_raw, generation_stream_seed
from .math_baseline_calibration import validate_roles

CODE = Path(__file__).resolve().parents[2]


class FixedVectorController(NoInterventionController):
    """Decode-position addition without running the SAE encoder at inference.

    Absolute mode uses the same BF16 addition as ASC ResidualAddition. Relative
    mode matches the norm-intervention formula. Both preserve earlier states and
    completed rows, while retaining inexpensive perturbation-norm diagnostics.
    """
    def __init__(self, layer, vector, kind, strength):
        super().__init__(vector.device)
        if kind not in ('absolute_vector','relative_vector') or strength < 0:
            raise ValueError('Invalid fixed-vector intervention')
        if vector.ndim != 1 or not vector.isfinite().all(): raise ValueError('Invalid vector')
        self.layer, self.vector, self.kind, self.strength = layer, vector, kind, strength
        if kind == 'relative_vector':
            if vector.norm() == 0: raise ValueError('Relative direction cannot be zero')
            self.vector = vector.float()/vector.float().norm()

    def begin(self, spec, batch_size):
        import torch
        super().begin(spec,batch_size)
        self.modified = torch.zeros(batch_size,device=self.device,dtype=torch.int64)
        self.maximum = torch.zeros(batch_size,device=self.device)
        self._handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output,tuple) else output
        flat = hidden[:,-1,:]; changed = hidden.clone()
        if self.kind == 'absolute_vector':
            delta = (self.vector*self.strength).to(flat)*self.live[:,None]
            changed[:,-1,:] = flat+delta
        else:
            delta = self.vector*self.strength*flat.float().norm(dim=-1,keepdim=True)*self.live[:,None]
            changed[:,-1,:] = (flat.float()+delta).to(flat.dtype)
        ratio = (changed[:,-1,:].float()-flat.float()).norm(dim=-1)/flat.float().norm(dim=-1).clamp_min(1e-12)
        self.modified += (ratio>0).long()
        self.maximum = self.maximum.maximum(ratio)
        return (changed,*output[1:]) if isinstance(output,tuple) else changed

    def end(self):
        self._handle.remove();self._handle=None
        return [{'modified_positions':n,'max_delta_to_hidden_norm_fraction':r}
                for n,r in zip(self.modified.tolist(),self.maximum.tolist())]


def settings_for_candidate(cfg, candidate):
    if not 0 <= candidate < cfg['candidates_per_question']: raise ValueError('Unknown candidate index')
    return {**cfg['generation'],'seed':cfg['generation']['seed']+cfg['candidate_seed_stride']*candidate}


def grade_candidate(cfg, response, source):
    method = cfg.get('grading_method', 'typed_math_v2')
    if 'reviewed_answer' in source and method != 'reviewed_math_v1':
        raise ValueError('Reviewed candidate source requires its declared grading backend')
    return grade_prediction({'grading': {'method': method, 'config': cfg['grading']}},
                            response, source['answer'], source=source)


def require_same_grading_method(*configs):
    if len({cfg.get('grading_method', 'typed_math_v2') for cfg in configs}) != 1:
        raise ValueError('Candidate grading backends differ')


def audit_candidate_grid(rows, questions, cfg, *, eos_token_id=None):
    sources = {r['problem_id']:r for r in questions}
    expected = {(pid,method,i) for pid in sources for method in cfg['methods'] for i in range(cfg['candidates_per_question'])}
    observed = set()
    for row in rows:
        key = (row['problem_id'],row['condition'],row['candidate_index'])
        if key not in expected or key in observed: raise ValueError('Unexpected or duplicate candidate key')
        observed.add(key);source=sources[row['problem_id']]
        if row['question'] != source['question'] or row['gold_answer'] != source['answer']:
            raise ValueError('Candidate source identity differs')
        seed = generation_stream_seed(settings_for_candidate(cfg,row['candidate_index'])['seed'],row['problem_id'])
        if row['seed'] != seed: raise ValueError('Unpaired or changed candidate random stream')
        if eos_token_id is not None:
            tokens=row['token_ids'];cap=cfg['generation']['max_new_tokens'];ended=bool(tokens and tokens[-1]==eos_token_id)
            if not tokens or len(tokens)>cap or (eos_token_id in tokens[:-1]): raise ValueError('Invalid token/EOS boundary')
            if not ended and len(tokens)!=cap: raise ValueError('Unfinished short candidate')
            if row['generated_tokens']!=len(tokens)-int(ended) or row['output_token_count']!=len(tokens) or row['hit_max_new_tokens']==ended:
                raise ValueError('Invalid candidate token/cap accounting')
    if observed != expected: raise ValueError('Missing candidate records')


def select_candidates(rows, questions, cfg):
    """Keep full-pool coverage separate from the available-arm common support."""
    audit_candidate_grid(rows,questions,cfg)
    grouped = {method:defaultdict(list) for method in cfg['methods']}
    for row in rows: grouped[row['condition']][row['problem_id']].append(row)
    chosen = {};coverage = {}
    for method, group in grouped.items():
        unique = {q['problem_id']:unique_correct(group[q['problem_id']]) for q in questions}
        chosen[method] = {pid:values[0] for pid,values in unique.items() if values}
        coverage[method] = {'questions':len(questions),'eligible_questions':len(chosen[method]),
            'unique_correct_histogram':dict(Counter(len(v) for v in unique.values())),
            'zero_support_ids':[pid for pid,v in unique.items() if not v]}
        if method == 'B1':
            chosen['B0']={pid:random.Random(int(canonical_sha256([cfg['selection_seed'],pid])[:16],16)).choice(values)
                          for pid,values in unique.items() if values}
            coverage['B0']={**coverage[method],'same_raw_pool_as':'B1'}
    common = sorted(set.intersection(*(set(v) for v in chosen.values())))
    return chosen,coverage,common


def validate_raw_student_transfer(cfg, development_cfg):
    """Allow the already tested raw source stage without tuning on student IDs."""
    require_same_grading_method(cfg, development_cfg)
    if cfg.get('student_generation_stage') != 'raw_B0_B1_B2_sources':
        raise ValueError('Student generation needs a separately registered stage')
    if set(cfg['methods']) != {'B1','B2'}:
        raise ValueError('Raw source registration covers B0/B1/B2 only')
    for key in ('teacher','grading','generation','candidates_per_question','candidate_seed_stride',
                'selection_seed','methods','decode_policy','selection_policy'):
        if cfg[key] != development_cfg[key]:
            raise ValueError('Student source protocol changed after development: '+key)


def selected_student_methods(points, development_cfg):
    """Only transfer the actual admissible development choices, with all details."""
    if not points['all_families_have_admissible_point']:
        raise ValueError('At least one steering family has no admissible development point')
    families = {'B3','B4','B7'}
    if set(points['selected_conditions']) != families or set(points['selected_method_specs']) != families:
        raise ValueError('The selected steering families are incomplete')
    methods = {}
    for family in sorted(families):
        condition = points['selected_conditions'][family]
        if not condition or not condition.startswith(family+'__') or condition not in development_cfg['methods']:
            raise ValueError('Selected condition is outside the registered development sweep')
        spec = points['selected_method_specs'][family]
        if spec != development_cfg['methods'][condition]:
            raise ValueError('Selected direction, layer, prompt or strength differs from the sweep')
        methods[family] = dict(spec)
    return methods


def bind_steered_student_transfer(cfg, roles, dev, bindings):
    """Resolve development choices only after their complete analysis is sealed."""
    from .unified_steering_analysis import operating_points
    analysis = Path(cfg['development_selection_root'])
    raw = Path(cfg['raw_student_candidate_root'])
    sweep = Path(cfg['steered_development_candidate_root'])
    for root, relatives in [(analysis, ('COMPLETE.json',)),
                           (raw, ('protocol/FROZEN.json','protocol/SOURCES.json','selection/smoke/COMPLETE.json')),
                           (sweep, ('protocol/FROZEN.json','protocol/SOURCES.json','selection/development/COMPLETE.json'))]:
        for relative in relatives:
            marker = root/relative; verify(marker); bindings.append(marker)
    acfg = read_json(Path(cfg['development_selection_config']))
    if Path(acfg['result_root']) != analysis or Path(acfg['steered_root']) != sweep:
        raise ValueError('Development selection config points to different evidence')
    # The config itself must be bound by the completed analysis.
    if str(Path(cfg['development_selection_config']).resolve()) not in read_json(analysis/'COMPLETE.json')['hashes']:
        raise ValueError('Selection config is not bound by the completed analysis')
    bindings.append(Path(cfg['development_selection_config']))
    scfg = read_json(sweep/'protocol/frozen_config.json'); rcfg = read_json(raw/'protocol/frozen_config.json')
    require_same_grading_method(cfg, scfg, rcfg)
    points = read_json(analysis/'operating_points.json')
    assessments = {r['condition']:r for r in read_jsonl(analysis/'condition_metrics.jsonl')}
    reference = read_json(analysis/'summary.json')['reference']
    recomputed, _ = operating_points(assessments, reference, scfg['operating_point_selection'])
    if {k:v['condition'] if v else None for k,v in recomputed.items()} != points['selected_conditions']:
        raise ValueError('Development choices do not follow the registered selection rule')
    if cfg.get('methods'):
        raise ValueError('Steered student methods must be resolved from complete development evidence')
    cfg['methods'] = selected_student_methods(points, scfg)
    cfg['selected_development_conditions'] = points['selected_conditions']
    for key in ('teacher','grading','generation','candidates_per_question','candidate_seed_stride',
                'selection_seed','decode_policy','selection_policy'):
        if cfg[key] != scfg[key] or cfg[key] != rcfg[key]:
            raise ValueError('Steering student generation changed a shared setting: '+key)
    if cfg['shards'] != rcfg['shards'] or cfg['runtime']['routes']['h100'] != rcfg['runtime']['routes']['h100']:
        raise ValueError('Steered student shards and H100 route must match the raw stage')
    if cfg['registered_gpu_route'] != 'h100' or set(cfg['allowed_generation_cohorts']) != {'smoke','student_pool'}:
        raise ValueError('Steered student stage must use its registered cohorts and H100 route')
    for name, expected in [('student_pool',roles['student_pool']),('development',dev)]:
        if list(read_jsonl(raw/'inputs'/f'{name}.jsonl')) != expected:
            raise ValueError('Steered and raw student/development question order differs')
    if list(read_jsonl(sweep/'inputs/development.jsonl')) != dev:
        raise ValueError('Selection sweep uses different development questions')
    bindings.extend([analysis/'operating_points.json', analysis/'condition_metrics.jsonl',
                     analysis/'summary.json', raw/'inputs/student_pool.jsonl'])


def prepare(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root']);parent=Path(cfg['math_asc_root'])
    if root.exists(): raise FileExistsError(root)
    bindings=[parent/'protocol/FROZEN.json',parent/'protocol/SOURCES.json']
    for marker in bindings:verify(marker)
    pcfg=read_json(parent/'protocol/frozen_config.json');cohorts=Path(pcfg['math_cohorts']);verify(cohorts/'COMPLETE.json')
    bindings.append(cohorts/'COMPLETE.json')
    cfg.update(teacher=pcfg['teacher'],grading=pcfg['grading']['config'],
               grading_method=pcfg['grading']['method'],code_root=str(root/'code'))
    roles={name:list(read_jsonl(cohorts/'cohorts'/f'{name}.jsonl')) for name in
           ('calibration','development','student_pool','dap_development_reserved')}
    validate_roles(*(roles[k] for k in ('calibration','development','student_pool','dap_development_reserved')))
    dev=list(read_jsonl(parent/'inputs/sources.jsonl'))
    if len(dev)!=cfg['development_questions'] or len(roles['student_pool'])!=cfg['student_pool_questions']:
        raise ValueError('Candidate cohort size differs')
    if not set(r['problem_id'] for r in dev)<=set(r['problem_id'] for r in roles['development']):
        raise ValueError('Development questions outside their frozen role')
    if 'student_pool' in cfg['allowed_generation_cohorts']:
        if cfg.get('student_generation_stage') == 'selected_B3_B4_B7':
            bind_steered_student_transfer(cfg,roles,dev,bindings)
        else:
            reference = Path(cfg['raw_development_candidate_root'])
            for relative in ('protocol/FROZEN.json','protocol/SOURCES.json','selection/development/COMPLETE.json'):
                marker = reference/relative; verify(marker); bindings.append(marker)
            validate_raw_student_transfer(cfg,read_json(reference/'protocol/frozen_config.json'))
            if list(read_jsonl(reference/'inputs/student_pool.jsonl')) != roles['student_pool']:
                raise ValueError('Raw student question pool changed after development')
            if list(read_jsonl(reference/'inputs/development.jsonl')) != dev:
                raise ValueError('Raw development question pool differs')
    if cfg.get('reference_candidate_root'):
        reference = Path(cfg['reference_candidate_root'])
        for relative in ('protocol/FROZEN.json','protocol/SOURCES.json','selection/development/COMPLETE.json'):
            marker = reference/relative; verify(marker); bindings.append(marker)
        rcfg = read_json(reference/'protocol/frozen_config.json')
        require_same_grading_method(cfg, rcfg)
        for key in ('teacher','grading','generation','candidates_per_question','candidate_seed_stride','selection_seed'):
            if cfg[key] != rcfg[key]: raise ValueError('Steered and reference candidate settings differ: '+key)
        previous = {q['problem_id']:q for q in read_jsonl(reference/'inputs/development.jsonl')}
        if {q['problem_id']:q for q in dev} != previous:
            raise ValueError('Steered and reference development questions differ')
        if any(spec['question_template'] != rcfg['methods']['B1']['question_template'] for spec in cfg['methods'].values()):
            raise ValueError('Steered methods must retain the B1 question template')
    smoke=sorted(dev,key=lambda r:canonical_sha256([cfg['smoke_subset_seed'],r['problem_id']]))[:cfg['smoke_questions']]
    for name,rows in [('smoke',smoke),('development',dev),('student_pool',roles['student_pool'])]:
        write_jsonl(root/'inputs'/f'{name}.jsonl',rows)
    if cfg['generation']['repetition_penalty']!=1. or cfg['generation']['add_special_tokens']:
        raise ValueError('Current decoder requires unit repetition penalty and already rendered special tokens')
    for name,spec in cfg['methods'].items():
        spec['question_template'].format(question='template validation')
        if spec['kind'] not in ('unmodified','absolute_vector','relative_vector'): raise ValueError('Unknown method kind')
        if spec['kind']!='unmodified':
            verify(spec['vector_marker']);bindings.extend([Path(spec['vector_marker']),Path(spec['vector_file'])])
    model_hashes=read_json(parent/'inputs/model_hashes.json')['teacher']
    for path,digest in model_hashes.items():
        if file_sha256(path)!=digest:raise ValueError('Changed teacher model input')
    save(root/'inputs/model_hashes.json',model_hashes)
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/FROZEN.json',[Path(config_path),root/'protocol/SOURCES.json',root/'protocol/frozen_config.json',
        *bindings,*sorted((root/'inputs').glob('*'))],stage='unified_mathematics_candidate_inputs',formal_training_ready=False)


def load(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if CODE!=Path(cfg['code_root']): raise ValueError('Use frozen candidate-generation source')
    return cfg


def generate(cfg, cohort, shard):
    if cfg.get('regrading_only'):
        raise ValueError('This protocol only regrades immutable parent candidates')
    import torch
    from safetensors.torch import load_file
    if cohort not in cfg['allowed_generation_cohorts']:raise ValueError('Cohort is not registered for generation')
    if not 0 <= shard < cfg['shards'] or (cohort=='smoke' and shard!=0):raise ValueError('Invalid shard')
    root=Path(cfg['result_root']);out=root/'generation'/cohort/f'shard_{shard:02d}'
    if cohort!='smoke':verify(root/'generation/smoke/shard_00/COMPLETE.json')
    questions=list(read_jsonl(root/'inputs'/f'{cohort}.jsonl'))
    if cohort!='smoke':questions=[q for i,q in enumerate(questions) if i%cfg['shards']==shard]
    out.mkdir(parents=True,exist_ok=False);admission(cfg);record_hardware(out);model,tok=teacher_bundle(cfg)
    if cfg.get('registered_gpu_route') == 'h100' and 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('This student-pool direct-generation stage is registered on H100')
    if model.generation_config.forced_eos_token_id is not None:raise ValueError('Forced EOS is not supported by this decoder')
    controllers={}
    for method,spec in cfg['methods'].items():
        controllers[method]=None
        if spec['kind']!='unmodified':
            vector=load_file(spec['vector_file'])[spec['vector_key']].to(model.device)
            controllers[method]=FixedVectorController(model.model.layers[spec['layer_index']],vector,spec['kind'],spec['strength'])
    torch.cuda.reset_peak_memory_stats();rows=[];batches=[];start=time.monotonic();replay_seconds=0.
    with (out/'predictions.jsonl').open('x') as handle:
        for offset in range(0,len(questions),cfg['generation']['batch_size']):
            batch=questions[offset:offset+cfg['generation']['batch_size']]
            for candidate in range(cfg['candidates_per_question']):
                settings=settings_for_candidate(cfg,candidate)
                for method,spec in cfg['methods'].items():
                    prompts=[{**q,'prompt':spec['question_template'].format(question=q['question'])} for q in batch]
                    torch.cuda.synchronize();before=time.monotonic()
                    outputs=generate_condition_raw(model,tok,controllers[method],{'name':method},prompts,settings)
                    torch.cuda.synchronize();elapsed=time.monotonic()-before
                    batch_id=f'{offset:04d}__{candidate}__{method}'
                    batches.append({'batch_id':batch_id,'condition':method,'candidate_index':candidate,
                                    'questions':len(batch),'generation_wall_seconds':elapsed})
                    if cohort=='smoke' and offset==0 and candidate==0:
                        before=time.monotonic();replay=generate_condition_raw(model,tok,controllers[method],{'name':method},prompts,settings)
                        torch.cuda.synchronize();replay_seconds+=time.monotonic()-before
                        if [r['token_ids'] for r in replay]!=[r['token_ids'] for r in outputs]:raise ValueError('Candidate replay differs')
                    for source,result in zip(prompts,outputs):
                        assessment=grade_candidate(cfg,result['response'],source)
                        row={**result,**assessment,'question':source['question'],'gold_answer':source['answer'],
                            'dataset':source['dataset'],'question_role':source['question_role'],'cohort':cohort,
                            'candidate_index':candidate,'teacher_prompt':source['prompt'],'method_spec':spec,
                            'batch_id':batch_id,'amortized_generation_wall_seconds':elapsed/len(batch)}
                        if spec['kind']=='relative_vector' and result['diagnostics']['max_delta_to_hidden_norm_fraction']>spec['strength']+cfg['norm_rounding_tolerance']:
                            raise ValueError('Actual vector dose exceeds the rounding tolerance')
                        rows.append(row);handle.write(json.dumps(row)+'\n')
                    handle.flush();logging.info('Math candidates %s shard=%d records=%d batch=%d candidate=%d method=%s',cohort,shard,len(rows),offset,candidate,method)
    audit_candidate_grid(rows,questions,cfg,eos_token_id=tok.eos_token_id)
    write_jsonl(out/'batches.jsonl',batches)
    save(out/'summary.json',{'questions':len(questions),'records':len(rows),'cohort':cohort,'shard':shard,
        'generation_batch_seconds':sum(r['generation_wall_seconds'] for r in batches),'replay_seconds':replay_seconds,
        'total_seconds':time.monotonic()-start,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'teacher_model':cfg['teacher']['model_name'],'teacher_revision':cfg['teacher']['revision'],
        'generation_settings':cfg['generation'],'no_sae_encoder_during_decoding':True,'formal_training_ready':False})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*sorted(out.glob('*'))],stage='unified_mathematics_candidates',formal_training_ready=False)


def merge(cfg,cohort):
    from transformers import AutoTokenizer
    from .sae_generation_analysis import audit_batches
    root=Path(cfg['result_root']);out=root/'selection'/cohort
    if out.exists():raise FileExistsError(out)
    if cohort not in cfg['allowed_generation_cohorts']:raise ValueError('Unregistered cohort')
    questions=list(read_jsonl(root/'inputs'/f'{cohort}.jsonl'));tok=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    rows=[];markers=[];costs=defaultdict(float)
    for shard in range(1 if cohort=='smoke' else cfg['shards']):
        path=root/'generation'/cohort/f'shard_{shard:02d}';marker=path/'COMPLETE.json';verify(marker);markers.append(marker)
        part=list(read_jsonl(path/'predictions.jsonl'))
        expected=questions if cohort=='smoke' else [q for i,q in enumerate(questions) if i%cfg['shards']==shard]
        audit_candidate_grid(part,expected,cfg,eos_token_id=tok.eos_token_id)
        for method,seconds in audit_batches(part,list(read_jsonl(path/'batches.jsonl'))).items():costs[method]+=seconds
        source_map={q['problem_id']:q for q in expected}
        for row in part:
            if row['response']!=tok.decode(row['token_ids'],skip_special_tokens=True):raise ValueError('Sampled text changed')
            grade=grade_candidate(cfg,row['response'],source_map[row['problem_id']])
            if any(row[k]!=v for k,v in grade.items()):raise ValueError('Candidate grading differs')
        rows.extend(part)
    selected,coverage,common=select_candidates(rows,questions,cfg);out.mkdir(parents=True)
    write_jsonl(out/'predictions.jsonl',rows)
    for method,values in selected.items():
        write_jsonl(out/f'{method}_full_support.jsonl',[dict(values[pid],selected_baseline=method) for pid in sorted(values)])
        write_jsonl(out/f'{method}_common_support.jsonl',[dict(values[pid],selected_baseline=method) for pid in common])
    save(out/'summary.json',{'cohort':cohort,'questions':len(questions),'records':len(rows),'coverage':coverage,
        'common_support_questions':len(common),'common_support_ids':common,'available_methods':list(selected),
        'generation_batch_seconds_by_method':dict(costs),'B0_and_B1_share_the_same_generation_cost':True,
        'formal_training_ready':False,'all_B0_through_B7_support_ready':False,
        'development_records_must_not_train_students':cohort!='student_pool'})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*markers,*sorted(out.glob('*'))],stage='unified_mathematics_candidate_selection',formal_training_ready=False)
