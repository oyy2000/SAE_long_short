"""Audit full state readbacks and decide whether controlled generation can run."""
import csv
import json
from pathlib import Path
import numpy as np
from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256, read_key_value_marker
from .sae_clean_features import load_clean_protocol


def analyze_engagement(path,project,version,finalize_failure=False):
    config,eligible=load_clean_protocol(path);root=project/config['outputs']['result_root'];suffix='' if version==1 else f'_v{version}'
    pp=root/f'engagement_protocol{suffix}/protocol.json';protocol=read_json(pp)
    if read_key_value_marker(pp.parent/'ENGAGEMENT_PROTOCOL_FROZEN')['protocol_sha256']!=file_sha256(pp):raise ValueError('Protocol changed')
    expected_ids={f'{r["trace_id"]}:slot_{slot}' for r in eligible if r['question_split']=='test' for slot in config['scoring']['readback_positions']}
    expected={(state,strength) for state in expected_ids for strength in config['engagement']['signed_strengths']}
    summaries=[];inputs=[]
    for slot,fid in enumerate(protocol['feature_ids']):
        d=root/f'engagement_shards{suffix}/feature_{slot:02d}';mp=d/'manifest.json';m=read_json(mp)
        if read_key_value_marker(d/'ENGAGEMENT_SHARD_COMPLETE')['manifest_sha256']!=file_sha256(mp):raise ValueError('Readback manifest changed')
        if m['protocol_sha256']!=file_sha256(pp) or m['config_hash']!=canonical_sha256(config) or m['source_sha256']!=protocol['source_sha256']:raise ValueError('Readback provenance mismatch')
        rp=d/'state_readbacks.jsonl'
        if file_sha256(rp)!=m['state_readbacks_sha256']:raise ValueError('State records changed')
        rows=[json.loads(l) for l in rp.open()];keys=[(r['state_id'],r['strength']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Duplicate or missing readback states')
        if any(r['feature_id']!=fid for r in rows):raise ValueError('Unexpected readback feature')
        for s in m['results']:
            subset=[r for r in rows if r['strength']==s['strength']]
            before=np.array([r['activation_before'] for r in subset]);after=np.array([r['activation_after'] for r in subset]);other=np.array([r['off_target_l2'] for r in subset])
            checks={'mean_target_before':float(before.mean()),'mean_target_after':float(after.mean()),'active_state_fraction':float((before>0).mean()),'relative_target_change':float((after.sum()-before.sum())/max(before.sum(),1e-12)),'off_target_to_target_l2_ratio':float(np.linalg.norm(other)/max(np.linalg.norm(after-before),1e-12))}
            for key,value in checks.items():
                if not np.isclose(value,s[key],rtol=1e-5,atol=1e-7):raise ValueError('State/summary mismatch: '+key)
            summaries.append(s)
        inputs.append({'manifest_path':str(mp),'manifest_sha256':file_sha256(mp),'records_path':str(rp),'records_sha256':file_sha256(rp)})
    primary=protocol['matching'][str(protocol['primary_feature_id'])];matching=True
    for fid in protocol['feature_ids'][1:]:
        candidate=protocol['matching'][str(fid)]
        for key in ('frequency','conditional_amplitude'):
            ratio=candidate[key]/primary[key]
            matching &= 1/protocol['maximum_control_fold_difference']<=ratio<=protocol['maximum_control_fold_difference']
    passed=bool(matching and all(s['passed'] for s in summaries))
    output=root/f'engagement_analysis_v{version}';output.mkdir(exist_ok=False)
    csv_path=output/'engagement_metrics.csv'
    with csv_path.open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    figure_paths=plot_engagement(summaries,config,project,version)
    report=output/'engagement_report_zh.md'
    lines=[f'# 目标激活读回 v{version}','',f'每个特征 {len(expected_ids)} 个确认集状态，正负两个方向；共 {len(expected)*len(protocol["feature_ids"])} 条读回记录。',f'全部状态覆盖与哈希审计通过。阶段门槛：{"通过" if passed else "未通过"}。','', '| 特征 | 方向 | 激活覆盖率 | 目标激活变化 | 非目标/目标 L2 比值 | 通过 |','|---|---:|---:|---:|---:|---|']
    for s in summaries:lines.append(f'| {s["feature_id"]} | {s["strength"]:+.0f} | {100*s["active_state_fraction"]:.2f}% | {100*s["relative_target_change"]:+.2f}% | {s["off_target_to_target_l2_ratio"]:.3f} | {s["passed"]} |')
    lines+=['','使用 BF16 隐藏状态写回，FP32 SAE 重新编码；只测量实际激活状态，未用空白或 EOS 后位置稀释诊断。','两个随机对照均需通过同一读回门槛，且 dev 频率与条件激活幅度须在主特征两倍范围内。','该阶段是冻结状态上的操作检验，不等于生成时有效覆盖或特征的语义因果效应。','']
    if version==2:lines+=['v1 中对照 882 因匹配支持集与读回位置不一致失败。v2 仅用 dev 的实际读回位置重新匹配，并增加 dev 最低覆盖筛选；确认门槛未放宽，原始失败记录保留。','']
    report.write_text('\n'.join(lines))
    mp=output/'engagement_analysis_manifest.json';m={'status':'passed','gate_passed':passed,'version':version,'config_hash':canonical_sha256(config),'protocol_sha256':file_sha256(pp),'control_matching_passed':bool(matching),'input_evidence':inputs,'state_records':len(expected)*len(protocol['feature_ids']),'missing_records':0,'duplicate_records':0,'artifacts':{str(p):file_sha256(p) for p in [csv_path,report,*figure_paths]},'source_sha256':file_sha256(Path(__file__)),'formal_claim_allowed':False};write_json_exclusive(mp,m)
    (output/'ENGAGEMENT_GATE_COMPLETE').write_text(f'status=passed\ngate_passed={str(passed).lower()}\nmanifest_sha256={file_sha256(mp)}\n')
    if finalize_failure and not passed:
        dp=root/'final_decision.json';decision={'status':'complete','decision':'stop_target_or_control_engagement_failed','completed_stages':['clean_feature_screen','target_engagement'],'not_triggered_stages':['independent_generation','student_sft'],'feature_gate_manifest_sha256':file_sha256(root/'feature_gate/feature_gate_manifest.json'),'engagement_manifest_sha256':file_sha256(mp),'formal_claim_allowed':False};write_json_exclusive(dp,decision);(root/'CLEAN_CAUSAL_PILOT_COMPLETE').write_text(f'status=complete\ndecision_sha256={file_sha256(dp)}\nformal_claim_allowed=false\n')
    print(json.dumps({'status':'passed','version':version,'engagement_gate_passed':passed,'report':str(report)}),flush=True)


def plot_engagement(rows,config,project,version):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(13,4.5));labels=[f'{r["feature_id"]}\n{r["strength"]:+.0f}' for r in rows];colors=['#4472C4' if r['role']=='primary' else '#ED7D31' for r in rows]
    for ax,key,factor,label in zip(axes,['relative_target_change','active_state_fraction','off_target_to_target_l2_ratio'],[100,100,1],['Target activation change (%)','Active-state coverage (%)','Off-target / target change (L2)']):
        ax.bar(labels,[factor*r[key] for r in rows],color=colors);ax.set_ylabel(label);ax.grid(axis='y',alpha=.25)
    axes[0].axhline(0,color='gray',linewidth=.8);axes[1].axhline(100*config['engagement']['minimum_engaged_state_fraction'],color='#C00000',linestyle='--');axes[2].axhline(config['engagement']['maximum_nontarget_change_ratio'],color='#C00000',linestyle='--')
    figure.suptitle(f'Target engagement v{version}: primary feature and matched random controls');figure.tight_layout();out=project/config['outputs']['figure_root'];paths=[out/f'target_engagement_v{version}.png',out/f'target_engagement_v{version}.pdf']
    for p in paths:figure.savefig(p,dpi=220,bbox_inches='tight')
    plt.close(figure);return paths
