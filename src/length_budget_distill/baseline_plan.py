"""Compile the staged reference SFT matrix without submitting jobs."""
from pathlib import Path
from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256


def compile_matrix(cfg):
    models=cfg['models'];rows=[]
    for module,spec in cfg['matrix'].items():
        teachers=[models.get(t,t) for t in spec['teachers']]
        students=[]
        for value in spec['students']:
            resolved=models.get(value,value)
            students.extend(resolved if isinstance(resolved,list) else [resolved])
        start=len(rows)
        for teacher in teachers:
            for condition in spec['conditions']:
                if condition not in cfg['baselines']:raise ValueError('Unknown baseline')
                for student in students:
                    for seed in cfg['student_seeds']:
                        rows.append({'module':module,'teacher':teacher,'condition':condition,'student':student,'seed':seed,'budget':spec['budget'],'status':'planned_not_submitted'})
        if len(rows)-start!=spec['expected_sft_runs']:raise ValueError('Module count mismatch: '+module)
    ids=[(r['teacher'],r['condition'],r['student'],r['seed'],r['budget']) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate SFT cells')
    if len(rows)!=cfg['total_reference_sft_runs_per_complete_scale']:raise ValueError('Total count mismatch')
    return rows


def write_matrix(config_path,project):
    project=Path(project);path=Path(config_path);cfg=read_json(path)
    rows=compile_matrix(cfg)
    out=project/cfg['outputs']['result_root']/'preparation/reference_matrix_v1'
    out.mkdir(parents=True,exist_ok=False)
    write_json_exclusive(out/'matrix.json',{'experiment':cfg['experiment_name'],'scope':'one complete scale; excludes pilot/fidelity checks, SAE fits, and additional compression-point SFT','runs':rows,'total_runs':len(rows),'config_sha256':file_sha256(path),'config_hash':canonical_sha256(cfg)})
    write_json_exclusive(out/'COMPLETE.json',{'status':'matrix_preparation_complete','experiment_execution_complete':False,'submitted_jobs':0,'matrix_sha256':file_sha256(out/'matrix.json'),'source_sha256':file_sha256(Path(__file__))})
    return out
