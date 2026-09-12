"""Versioned presentation review; preserve the completed experiment's artifacts."""
from __future__ import annotations
import copy
import re
from pathlib import Path
from .experiment_io import read_json,write_json_exclusive
from .sae_local_data import ROOT,paths,evidence,verify
from .sae_local_finalize import outcome_plots


def review(config):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    root,_=paths(config)
    marker=root/'EXPERIMENT_COMPLETE.json';completed=read_json(marker)
    verify(completed['audit']);verify(completed['report'])
    audit=read_json(completed['audit']['path'])
    for item in audit['outputs']:verify(item)
    revised=copy.deepcopy(config)
    revised['figure_root']=str(Path(config['figure_root'])/'presentation_v2')
    (ROOT/revised['figure_root']).mkdir(exist_ok=False)
    original=Axes.legend
    def legend_with_clearance(axis,*args,**kwargs):
        if axis.get_xlabel()=='Paired accuracy difference (percentage points)':
            axis.set_ylim(-.4,2.7)
            kwargs['loc']='upper right'
        return original(axis,*args,**kwargs)
    Axes.legend=legend_with_clearance
    try:
        # Reuse the frozen plot implementation. Only add empty legend headroom.
        figures=outcome_plots(revised,read_json(root/'student_followup/STUDENT_COMPLETE.json'))
    finally:Axes.legend=original
    report=ROOT/'docs/phase6_local_length_controlled_strength_report_zh_v2.md'
    text=Path(completed['report']['path']).read_text()
    text=text.replace('../figures/phase6_local_length_controlled_strength_v1/05_student_accuracy.png',
                      '../figures/phase6_local_length_controlled_strength_v1/presentation_v2/05_student_accuracy.png')
    text=text.replace('等目标 token下','等目标 token 下')
    text+='\n展示复核：图 5 的图例增加留白；此展示版本对应原实验完成标记。参见[展示复核记录](../results/phase6_local_length_controlled_strength_v1/exploratory/PRESENTATION_REVIEW_COMPLETE.json)。\n'
    with report.open('x') as handle:handle.write(text)
    links=[]
    output=root/'PRESENTATION_REVIEW_COMPLETE.json'
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if link.startswith(('http:','https:','#')):continue
        target=(report.parent/link.split('#')[0]).resolve()
        if target==output:continue
        if not target.exists():raise ValueError(f'Broken presentation link: {link}')
        links.append(str(target))
    write_json_exclusive(output,{'status':'complete','scope':'presentation review only',
        'parent_completion':evidence(marker),'report':evidence(report),
        'figures':[evidence(p) for p in figures],'source':evidence(Path(__file__)),
        'change':'Add empty vertical headroom for the comparison legend; reuse existing metric-driven plotting logic.',
        'report_link_checks':links,'formal_claim_allowed':False})
    print(report,flush=True)
