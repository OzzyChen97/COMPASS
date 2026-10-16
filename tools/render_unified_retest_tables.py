"""Validate complete unified result rows and render their sole CSV/LaTeX source."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_motionx_random_retrieval import digest,GROUP_SEEDS
BASE=ROOT/'ablation/unified_retest_cfg13_20260909'
NAMES={'t2mgpt':'T2M-GPT','satopp':'T2M-GPT+SATO++','motiongpt':'MotionGPT','motiongpt_satopp':'MotionGPT+SATO++',
       'synonym':'Synonym augmentation','no_preserve':'w/o preserve','no_attribute':'w/o attribution','no_teacher':'w/o teacher','no_alter':'w/o alter'}
METRICS=['FID','FID_P','FID_D','Matching_P','R1_P','R2_P','R3_P','Diversity_O','Diversity_real']


def load(path,n):
    s=json.loads((path/'summary.json').read_text());r=json.loads((path/'run.json').read_text())
    assert s['samples']==n and not s['smoke'] and s['cfg_scale']==1.3
    assert r['seed']==123 and r['group_seeds']==GROUP_SEEDS
    assert digest(r['source']['checkpoint'])==s['checkpoint_sha256']
    z=np.load(path/'embeddings.npz',allow_pickle=False)
    assert len(z['motion_id'])==n and len(set(z['motion_id']))==n
    parts=json.loads((path/'partitions.json').read_text())
    for cond,suffix in [('original','O'),('perturbed','P')]:
        rows=[x for x in parts if x['condition']==cond]
        assert [x['group_seed'] for x in rows]==GROUP_SEEDS
        np.testing.assert_allclose(np.mean([x['R'] for x in rows],axis=0),[s[f'R{k}_{suffix}'] for k in (1,2,3)],rtol=0,atol=1e-12)
    audits=[]
    for file in sorted((path/'shards').glob('batch_*.json')):
        a=json.loads(file.read_text())
        for cond in ('original','perturbed'):
            assert all(x['generation_text']==x['retrieval_text'] for x in a[cond])
        audits.extend(x['motion_id'] for x in a['perturbed'])
    assert sorted(audits)==sorted(z['motion_id'])
    s['result_path']=str(path);s['summary_sha256']=digest(path/'summary.json')
    return s,z


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-encoder-fid',type=Path)
    args=parser.parse_args()
    fid_index={}
    if args.source_encoder_fid:
        assert json.loads((args.source_encoder_fid.parent/'complete.json').read_text())['all_fid_recomputed']
        fid_rows=json.loads(args.source_encoder_fid.read_text())
        fid_index={(r['dataset'],r['model']):r for r in fid_rows}
        assert len(fid_index)==18
    groups={};allrows=[]
    for dataset,n,models in [('humanml',4380,list(NAMES)[:4]),('motionx',7679,list(NAMES)[:4]),('kit',734,['t2mgpt','motiongpt','satopp']),('components',4380,['t2mgpt','synonym','satopp','no_preserve','no_attribute','no_teacher','no_alter']),('transfer',734,['t2mgpt','satopp'])]:
        rows=[];ref=None;ids=None;ruler=None;pair=None
        for name in models:
            if dataset=='kit':path=BASE/'remaining/kit/evaluations'/name
            elif dataset=='components':path=BASE/('runs/humanml' if name in ('t2mgpt','satopp') else 'remaining/component_evaluation')/name
            else:path=BASE/'runs'/dataset/name
            s,z=load(path,n)
            if ids is None:ids=z['motion_id'];ruler=s['ruler_sha256'];pair=s['pair_sha256'];ref=z.get('real')
            else:
                assert np.array_equal(ids,z['motion_id']) and ruler==s['ruler_sha256'] and pair==s['pair_sha256']
                if ref is not None:assert np.array_equal(ref,z['real'])
            row=dict(s,table=dataset,row=name,label=NAMES[name])
            if fid_index:
                f=fid_index[('humanml' if dataset=='components' else dataset,name)]
                assert all(f[k]==s[k] for k in ('samples','pair_sha256','ruler_sha256'))
                for metric in ('FID','FID_P','FID_D'):
                    if metric in f:row[metric]=f[metric]
                row.update(fid_result_path=f['result_path'],fid_checkpoint_sha256=f['checkpoint_sha256'],
                    fid_source_summary_sha256=digest(Path(f['result_path'])/'summary.json'),
                    column_protocol='FID variants: source encoders; all other metrics: independently adapted encoders')
            rows.append(row);allrows.append(row)
        groups[dataset]=rows
    out=args.source_encoder_fid.parent/'paper_tables' if fid_index else BASE/'consolidated'
    out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(allrows,indent=2)+'\n')
    fields=['table','row','label','samples']+METRICS+['paired_embedding_L2','first_token_agreement','checkpoint_sha256','ruler_sha256','pair_sha256','result_path','summary_sha256']
    if fid_index:fields+=['fid_result_path','fid_checkpoint_sha256','fid_source_summary_sha256','column_protocol']
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(allrows)
    for group,rows in groups.items():
        keys=METRICS[:-1] if group!='transfer' else ['FID_D','Matching_P','R1_P','R2_P','R3_P','paired_embedding_L2','first_token_agreement']
        labels={'FID':'FID $\\downarrow$','FID_P':'FID$_P$ $\\downarrow$','FID_D':'FID$_D$ $\\downarrow$','Matching_P':'Matching$_P$ $\\downarrow$','Diversity_O':'Div.$_O$',
            'R1_P':'R@1$_P$','R2_P':'R@2$_P$','R3_P':'R@3$_P$','paired_embedding_L2':'Paired $\\ell_2$','first_token_agreement':'First token (\\%)'}
        family_layout=group in ('humanml','motionx')
        if family_layout:
            lines=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{8}{c}@{}}',r'\toprule',
                r'Method & \multicolumn{3}{c}{Source encoder} & \multicolumn{5}{c}{Adapted encoder} \\',
                r'\cmidrule(lr){2-4}\cmidrule(lr){5-9}',
                r' & FID $\downarrow$ & FID$_P$ $\downarrow$ & FID$_D$ $\downarrow$ & Matching$_P$ $\downarrow$ & R@1$_P$ $\uparrow$ & R@2$_P$ $\uparrow$ & R@3$_P$ $\uparrow$ & Div.$_O$ \\',r'\midrule']
        else:
            lines=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l'+'r'*len(keys)+'@{}}','\\toprule','Method & '+' & '.join(labels[k] for k in keys)+r' \\',r'\midrule']
        def score(row,key):
            if key=='Diversity_O':return abs(row[key]-row['Diversity_real'])
            return -row[key] if key.startswith('R') else row[key]
        for i,row in enumerate(rows):
            if family_layout and i==2:lines.append(r'\midrule')
            peers=rows[(i//2)*2:(i//2)*2+2] if family_layout else rows
            values=[]
            for k in keys:
                value=f"{row[k]*100:.2f}" if k.startswith('R') or k=='first_token_agreement' else f"{row[k]:.3f}"
                if (family_layout or group in ('kit','components')) and score(row,k)==min(score(other,k) for other in peers):
                    value=r'\textbf{'+value+'}'
                values.append(value)
            prefix=[row['label']]
            lines.append(' & '.join(prefix+values)+r' \\')
        lines+=['\\bottomrule','\\end{tabular*}']
        (ROOT/f'paper_tip/tables/unified_{group}.tex').write_text('\n'.join(lines)+'\n')
    (out/'audit.json').write_text(json.dumps(dict(complete=True,table_rows=len(allrows),unique_runs=18,checks=['sample coverage','shared references','shared pair/ruler hashes','checkpoint hashes','retrieval partition averages','same generation/retrieval text']),indent=2)+'\n')
    print('UNIFIED_TABLES_VERIFIED',len(allrows))


if __name__=='__main__':main()