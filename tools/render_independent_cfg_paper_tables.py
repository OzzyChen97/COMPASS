"""Generate independent-CFG manuscript tables directly from audited CSV files."""
import csv
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS=['t2mgpt','satopp','motiongpt','motiongpt_satopp']
LABELS=['T2M-GPT','T2M-GPT + SATO++','MotionGPT','MotionGPT + SATO++']


def render(dataset):
    path=ROOT/f'ablation/{dataset}_independent_cfg13_20260907/{dataset}_summary.csv'
    rows=[r for r in csv.DictReader(path.open()) if r['lookup']=='fallback_lower']
    assert len(rows)==12
    expected=7679 if dataset=='motionx' else 4380
    assert all(int(r['samples'])==expected for r in rows)
    index={(r['model'],r['level']):r for r in rows};assert len(index)==12
    lines=[f'% Source: {path.relative_to(ROOT)}',f'% SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}',
           '% Percent; one generation seed (123), averaged over 20 random candidate partitions.',
           r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll*{9}{c}@{}}',r'\toprule',
           r'Method & Encoder & \multicolumn{3}{c}{P1} & \multicolumn{3}{c}{P2} & \multicolumn{3}{c}{P3} \\',
           r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
           r'& & R@1 & R@2 & R@3 & R@1 & R@2 & R@3 & R@1 & R@2 & R@3 \\',r'\midrule']
    for i,m in enumerate(MODELS):
        if i==2:lines.append(r'\midrule')
        cells=[LABELS[i], 'CLIP' if i<2 else 'T5']
        family=MODELS[:2] if i<2 else MODELS[2:]
        for level in ['P1','P2','P3']:
            for key in ['R1','R2','R3']:
                value=float(index[m,level][key]);cell=f'{100*value:.2f}'
                if value==max(float(index[other,level][key]) for other in family):cell=r'\textbf{'+cell+'}'
                cells.append(cell)
        lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular*}'])
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    for dataset in ['motionx','humanml']:
        out=ROOT/f'paper_tip/tables/{dataset}_independent_cfg13.tex'
        with out.open('x') as f:f.write(render(dataset))
        print(out)