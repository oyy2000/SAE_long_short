"""Audited screening and figures for fixed-support clean SAE features."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256, read_key_value_marker
from .sae_clean_features import load_clean_protocol
from .sae_feature_analysis import paired_feature_statistics, holm_adjust


def residualize_from_discovery(values, nuisance, discovery):
    design=np.column_stack([np.ones(len(nuisance)),nuisance])
    coefficients=np.linalg.lstsq(design[discovery],values[discovery],rcond=None)[0]
    return values-design@coefficients,coefficients


def analyze_clean_features(path,project):
    config,rows=load_clean_protocol(path);root=project/config['outputs']['result_root']
    count=len(rows);width=config['clean_tokens']['count'];k=config['parent_sae']['k'];feature_count=config['parent_sae']['feature_count']
    all_indices=np.zeros((count,width,k),np.int32);all_values=np.zeros((count,width,k),np.float32);seen=np.zeros(count,np.uint8)
    evidence=[];sources=set()
    for shard in range(config['scoring']['shards']):
        directory=root/f'score_shards/shard_{shard:02d}';mp=directory/'manifest.json';m=read_json(mp);mark=read_key_value_marker(directory/'SCORE_SHARD_COMPLETE')
        if mark.get('status')!='complete' or mark['manifest_sha256']!=file_sha256(mp):raise ValueError('Incomplete or mismatched score shard')
        if m['config_hash']!=canonical_sha256(config) or m['data_sha256']!=file_sha256(m['data_path']):raise ValueError('Score input hash mismatch')
        with np.load(m['data_path']) as data:
            idx=data['row_indices']
            if len(np.unique(idx))!=len(idx) or np.any(seen[idx]):raise ValueError('Duplicate trace in scores')
            all_indices[idx]=data['indices'];all_values[idx]=data['values'];seen[idx]=1
        sources.add(m['source_sha256']);evidence.append({'manifest_path':str(mp),'manifest_sha256':file_sha256(mp),'data_path':m['data_path'],'data_sha256':m['data_sha256']})
    if not np.all(seen==1) or len(sources)!=1:raise ValueError('Missing traces or inconsistent scoring source')
    dense=np.zeros((count,feature_count),np.float32);early=np.zeros_like(dense)
    rr=np.broadcast_to(np.arange(count)[:,None,None],all_values.shape)
    np.add.at(dense,(rr.ravel(),all_indices.ravel()),all_values.ravel())
    n=config['clean_tokens']['early_count']
    np.add.at(early,(rr[:,:n].ravel(),all_indices[:,:n].ravel()),all_values[:,:n].ravel())
    dense/=width;early/=n
    splits=np.array([r['question_split'] for r in rows]);dev=splits=='dev';test=splits=='test'
    nuisance=np.array([r['nuisance'] for r in rows]);residual,coefficients=residualize_from_discovery(dense,nuisance,dev)
    statistics={}
    for split,mask in [('dev',dev),('test',test)]:
        subset=[r for r,keep in zip(rows,mask) if keep]
        statistics[split]={}
        for name,array in [('clean',dense),('early',early),('adjusted',residual)]:
            statistics[split][name]=paired_feature_statistics(array[mask],subset,frequency_values=dense[mask])
    gate=config['feature_gate'];d=statistics['dev'];t=statistics['test']
    eligible=(d['clean']['bh_q_value']<=gate['max_dev_bh_q']) & (d['adjusted']['bh_q_value']<=gate['max_dev_bh_q'])
    eligible&=(np.abs(d['clean']['paired_d'])>=gate['min_dev_abs_paired_d']) & (np.abs(d['adjusted']['paired_d'])>=gate['min_dev_abs_paired_d'])
    eligible&=(d['clean']['paired_d']*d['adjusted']['paired_d']>0) & (d['clean']['paired_d']*d['early']['paired_d']>0)
    eligible&=(d['early']['p_value']<=gate['max_early_p']) & (d['clean']['prevalence']>=gate['minimum_prevalence'])
    candidates=[]
    ranking=np.minimum(np.abs(d['clean']['paired_d']),np.abs(d['adjusted']['paired_d']))
    for sign,direction in [(1,'short'),(-1,'long')]:
        pool=np.flatnonzero(eligible & (sign*d['clean']['paired_d']>0))
        ranked=sorted(pool,key=lambda fid:(-ranking[fid],int(fid)))[:gate['candidate_count_per_direction']]
        candidates.extend({'feature_id':int(fid),'direction':direction,'discovery_score':float(ranking[fid])} for fid in ranked)
    candidate_ids=[r['feature_id'] for r in candidates]
    corrected=holm_adjust([max(t['clean']['p_value'][fid],t['adjusted']['p_value'][fid]) for fid in candidate_ids])
    token_ids=np.array([r['token_ids'] for r in rows])
    for slot,candidate in enumerate(candidates):
        fid=candidate['feature_id'];sign=1 if candidate['direction']=='short' else -1
        mass=(all_values*(all_indices==fid)).sum(axis=-1)
        shares=[]
        for mask in (dev,test):
            totals=np.bincount(token_ids[mask].ravel(),weights=mass[mask].ravel())
            shares.append(float(totals.max()/max(totals.sum(),1e-12)))
        candidate.update({'confirmation_holm_p':float(corrected[slot]),'maximum_top_token_mass_share':max(shares),
            'dev_clean_d':float(d['clean']['paired_d'][fid]),'dev_adjusted_d':float(d['adjusted']['paired_d'][fid]),
            'test_clean_d':float(t['clean']['paired_d'][fid]),'test_adjusted_d':float(t['adjusted']['paired_d'][fid]),'test_early_d':float(t['early']['paired_d'][fid]),'test_early_p':float(t['early']['p_value'][fid])})
        reasons=[]
        if corrected[slot]>gate['max_confirmation_holm_p']:reasons.append('confirmation_holm')
        if any(sign*t[metric]['paired_d'][fid]<gate['min_confirmation_abs_paired_d'] for metric in ('clean','adjusted')):reasons.append('confirmation_direction_or_effect')
        if sign*t['early']['paired_d'][fid]<=0 or t['early']['p_value'][fid]>gate['max_early_p']:reasons.append('early_replication')
        if max(shares)>gate['maximum_top_token_mass_share']:reasons.append('single_token_concentration')
        candidate['rejection_reasons']=reasons;candidate['passed']=not reasons
    passed=sorted([r for r in candidates if r['passed']],key=lambda r:(-r['discovery_score'],r['feature_id']))
    selected=passed[:gate['max_selected_features']]
    output=root/'feature_gate';output.mkdir(exist_ok=False)
    out_csv=output/'all_feature_statistics.csv'
    fields=['feature_id']+[f'{s}_{metric}_{value}' for s in ('dev','test') for metric in ('clean','adjusted','early') for value in ('effect','paired_d','p_value','bh_q_value','prevalence')]
    with out_csv.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for fid in range(feature_count):
            row={'feature_id':fid}
            for s in ('dev','test'):
                for metric in ('clean','adjusted','early'):
                    for value in ('effect','paired_d','p_value','bh_q_value','prevalence'):row[f'{s}_{metric}_{value}']=float(statistics[s][metric][value][fid])
            writer.writerow(row)
    coefficient_path=output/'discovery_nuisance_coefficients.npy';np.save(coefficient_path,coefficients)
    selection={'status':'complete','gate_passed':bool(selected),'candidate_count':len(candidates),'passed_count':len(passed),'candidates':candidates,'selected':selected,'config_hash':canonical_sha256(config),'formal_claim_allowed':False}
    selection_path=output/'selected_features.json';write_json_exclusive(selection_path,selection)
    figures=plot_clean_feature_gate(config,statistics,candidates,project)
    old= config['old_selected_features']['short_feature_ids']
    report=output/'feature_gate_report_zh.md'
    lines=['# 固定推理 token 的 SAE 特征重筛选','',f'发现集：{config["cohort_audit"]["dev"]}；确认集：{config["cohort_audit"]["test"]}。',f'每条轨迹固定 {width} 个 token，全部 {feature_count} 个特征完成重评分。',f'发现集预筛通过 {int(eligible.sum())} 个；注册候选 {len(candidates)} 个；最终通过 {len(passed)} 个。','', '| 原短特征 | 确认集清理后 d | 确认集校正后 d | 清理后 p |','|---|---:|---:|---:|']
    for fid in old:lines.append(f'| {fid} | {t["clean"]["paired_d"][fid]:.4f} | {t["adjusted"]["paired_d"][fid]:.4f} | {t["clean"]["p_value"][fid]:.4g} |')
    lines+=['','入选特征：'+(', '.join(str(r['feature_id']) for r in selected) if selected else '无。按预设门槛停止目标干预与学生蒸馏。'),'','开发与确认题目均为旧池探索性复用；这里的确认是划分内复现，不是独立新样本的机制证据。','原始与协变量校正统计均按题聚合；未因阴性结果调整阈值。','']
    report.write_text('\n'.join(lines))
    artifacts=[out_csv,coefficient_path,selection_path,report,*figures]
    manifest={'status':'passed','gate_passed':bool(selected),'config_hash':canonical_sha256(config),'config_sha256':file_sha256(path),'eligible_trace_count':count,'scored_trace_count':int(seen.sum()),'scored_token_count':count*width,'duplicate_trace_count':0,'missing_trace_count':0,'feature_count':feature_count,'inputs':evidence,'source_sha256':file_sha256(Path(__file__)),'artifacts':{str(p):file_sha256(p) for p in artifacts},'formal_claim_allowed':False}
    mp=output/'feature_gate_manifest.json';write_json_exclusive(mp,manifest)
    (output/'FEATURE_GATE_COMPLETE').write_text(f'status=passed\ngate_passed={str(bool(selected)).lower()}\nmanifest_sha256={file_sha256(mp)}\n')
    if not selected:
        decision={'status':'complete','decision':'stop_no_clean_replicated_feature','completed_stages':['clean_feature_screen'],'not_triggered_stages':['target_engagement','independent_generation','student_sft'],'feature_gate_manifest_sha256':file_sha256(mp),'formal_claim_allowed':False}
        dp=root/'final_decision.json';write_json_exclusive(dp,decision)
        (root/'CLEAN_CAUSAL_PILOT_COMPLETE').write_text(f'status=complete\ndecision_sha256={file_sha256(dp)}\nformal_claim_allowed=false\n')
    print(json.dumps(selection),flush=True)


def plot_clean_feature_gate(config,statistics,candidates,project):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=project/config['outputs']['figure_root'];out.mkdir(parents=True,exist_ok=False)
    old_path=Path(config['old_selected_features']['selection_source_path'])
    old={r['feature_id']:r for r in read_json(old_path)['candidates']}
    ids=config['old_selected_features']['short_feature_ids']
    x=np.arange(len(ids));figure,axes=plt.subplots(1,2,figsize=(12,4.7))
    for offset,label,color,array in [(-.25,'Original full-trace','#4472C4',[old[f]['metrics']['token_mean_activation']['test']['paired_d'] for f in ids]),(0,'Fixed clean 64 tokens','#70AD47',[statistics['test']['clean']['paired_d'][f] for f in ids]),(.25,'Clean + nuisance adjustment','#ED7D31',[statistics['test']['adjusted']['paired_d'][f] for f in ids])]:
        axes[0].bar(x+offset,array,width=.24,label=label,color=color)
    axes[0].set_xticks(x,[str(f) for f in ids]);axes[0].set_ylabel('Confirmation paired d (short minus long)');axes[0].set_xlabel('Previously selected short feature');axes[0].legend(fontsize=8);axes[0].axhline(0,color='gray',linewidth=.8)
    if candidates:
        for passed,color,label in [(False,'#999999','Rejected'),(True,'#4472C4','Passed')]:
            r=[r for r in candidates if r['passed']]
            if not passed:r=[r for r in candidates if not r['passed']]
            axes[1].scatter([v['dev_adjusted_d'] for v in r],[v['test_adjusted_d'] for v in r],color=color,label=label)
        axes[1].legend()
    else:axes[1].text(.5,.5,'No feature passed the discovery gate',ha='center',va='center',transform=axes[1].transAxes)
    axes[1].axhline(0,color='gray',linewidth=.8);axes[1].axvline(0,color='gray',linewidth=.8);axes[1].set_xlabel('Discovery adjusted paired d');axes[1].set_ylabel('Confirmation adjusted paired d')
    for axis in axes:axis.grid(axis='y',alpha=.25)
    figure.suptitle('SAE length association after removing answer / position / denominator confounds')
    figure.tight_layout();paths=[out/'clean_feature_gate.png',out/'clean_feature_gate.pdf']
    for p in paths:figure.savefig(p,dpi=220,bbox_inches='tight')
    plt.close(figure);return paths
