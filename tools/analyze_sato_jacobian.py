#!/usr/bin/env python3
"""Summarize direct Jacobian estimates and plot honest paired diagnostics."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-sato-jacobian')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def paired_summary(values, bootstrap=2000, seed=123):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite paired metric; inspect raw records')
    rng=np.random.default_rng(seed)
    draws=values[rng.integers(len(values),size=(bootstrap,len(values)))].mean(axis=1)
    return {'mean':float(values.mean()),'median':float(np.median(values)),
            'mean_ci95':np.percentile(draws,[2.5,97.5]).tolist(),
            'fraction_below_one':float((values<1).mean()),
            'q10_q90':np.percentile(values,[10,90]).tolist(),
            'min':float(values.min()),'max':float(values.max()),'n_groups':len(values)}


def create_summary(source, audit_path, bootstrap):
    manifest=json.loads((source/'manifest.json').read_text())
    pairs={m:sorted(load_rows(source/f'{m}_jacobian.jsonl'),key=lambda r:r['group']) for m in ('baseline','satopp')}
    for m in pairs:
        complete=json.loads((source/f'{m}_complete.json').read_text())
        if len(pairs[m])!=complete['samples'] or complete['selection_digest']!=manifest['selection_digest']:
            raise ValueError('Incomplete extraction')
    if len(pairs['baseline'])!=len(pairs['satopp']):raise ValueError('Unmatched rows')
    records=[]
    for b,s in zip(pairs['baseline'],pairs['satopp']):
        if any(b[k]!=s[k] for k in ('group','input_sha256','target_sha256','positions','input_shape','output_shape')):
            raise ValueError('Model input, prefix or direction identities differ')
        f2b,f2s=b['frobenius']['squared_norm'],s['frobenius']['squared_norm']
        sb,ss=b['spectral']['norm_estimate'],s['spectral']['norm_estimate']
        record={'group':b['group'],'motion_id':b['motion_id'],
                'frobenius_squared_ratio':f2s/f2b,'frobenius_ratio':np.sqrt(f2s/f2b),
                'spectral_ratio':ss/sb,'hidden_norm_ratio':s['base_hidden_norm']/b['base_hidden_norm'],
                'scaled_frobenius_ratio':np.sqrt(s['frozen_scale_frobenius_squared']/b['frozen_scale_frobenius_squared']),
                'scaled_spectral_ratio':s['frozen_scale_spectral']/b['frozen_scale_spectral']}
        for kind in ('high','low','random'):
            gains={m:[r['gain'] for r in row['directional'][kind]] for m,row in [('baseline',b),('satopp',s)]}
            if any(x<=0 for x in gains['baseline']):raise ValueError('Zero baseline direction gain: ratio undefined')
            ratios=np.asarray(gains['satopp'])/np.asarray(gains['baseline'])
            record[f'{kind}_direction_ratio']=float(ratios.mean())
            record[f'{kind}_scaled_direction_ratio']=float(ratios.mean()/record['hidden_norm_ratio'])
        records.append(record)
    names=[k for k in records[0] if k not in ('group','motion_id')]
    statistics={k:paired_summary([r[k] for r in records],bootstrap) for k in names}
    quality={}
    for model,rows in pairs.items():
        errors={}
        for row in rows:
            for kind in ('high','low','random'):
                for direction in row['directional'][kind]:
                    for fd in direction['finite_difference_checks']:
                        errors.setdefault(str(fd['epsilon']),[]).append(fd['jvp_relative_error'])
        quality[model]={
            'spectral_converged_groups':sum(r['spectral']['best_converged'] for r in rows),
            'spectral_residual_median':float(np.median([r['spectral']['best_residual'] for r in rows])),
            'spectral_residual_max':float(max(r['spectral']['best_residual'] for r in rows)),
            'power_iterations_median':float(np.median([len(run['trace']) for r in rows for run in r['spectral']['restarts']])),
            'hutchinson_relative_probe_se_median':float(np.median([r['frobenius']['probe_standard_error']/r['frobenius']['squared_norm'] for r in rows])),
            'finite_difference_relative_error':{e:{'median':float(np.median(v)),'p90':float(np.percentile(v,90)),'max':float(max(v))} for e,v in errors.items()},
            'frobenius_squared_mean':float(np.mean([r['frobenius']['squared_norm'] for r in rows])),
            'spectral_mean':float(np.mean([r['spectral']['norm_estimate'] for r in rows])),
            'hidden_norm_mean':float(np.mean([r['base_hidden_norm'] for r in rows]))}
    audit=json.loads(audit_path.read_text())
    return {'schema':'sato-jacobian-analysis/v1','manifest':manifest,'statistics':statistics,'quality':quality,
            'paired_rows':records,'bootstrap_repetitions':bootstrap,
            'bootstrap_unit':'one source-motion group; all three directions are averaged within each group before resampling',
            'ratio_estimand':'mean of within-group matched ratios, not a ratio of population means',
            'gradient_audit':{k:v['attribution_total'] for k,v in audit['audits'].items()},
            'interpretation_limits':[
                'Raw Frobenius squared-norm ratio corresponds to q^2 under ideal uniform contraction, not q.',
                'Finite-difference relative hidden response is not a spectral norm estimate.',
                'The same baseline-defined word positions and directions are used for both models.',
                'Power iteration is an estimate, not a certified operator-norm bound.',
                'Hutchinson Monte Carlo uncertainty is recorded separately; motion bootstrap conditions on the sampled probes.',
                'Jacobian differences across checkpoints do not attribute those differences causally to the attribution loss.',
                'The current manuscript provides no Theorems 1/4/5 or explicit R_rob definition; no theorem verification or robustness-gradient alignment experiment is claimed.']}, pairs


def ratio_panel(ax, summary, keys, labels, color):
    rng=np.random.default_rng(7)
    for i,key in enumerate(keys):
        values=np.asarray([r[key] for r in summary['paired_rows']])
        ax.scatter(i+rng.uniform(-.11,.11,len(values)),values,color=color,alpha=.35,s=17,edgecolors='none')
        st=summary['statistics'][key]
        lo,hi=st['mean_ci95']
        ax.errorbar(i,st['mean'],yerr=[[max(0,st['mean']-lo)],[max(0,hi-st['mean'])]],fmt='D',color=color,ms=5,capsize=4,lw=1.6)
    ax.axhline(1,color='#555555',linestyle='--',lw=1,label='No change: q = 1')
    ax.axhline(.75,color='#999999',linestyle=':',lw=1,label='Proposed reference: q = 0.75')
    ax.set_xticks(range(len(keys)),labels)
    ax.set_xlim(-.45,len(keys)-.55)
    ax.set_ylabel('SATO++ / T2M-GPT (paired ratio)')
    ax.set_ylim(bottom=0)
    ax.spines[['top','right']].set_visible(False)


def ratio_axis_limits(summary, keys):
    """Keep shared axes wide enough for every group, including ratios above 1."""
    maximum=max(1.,max(row[key] for row in summary['paired_rows'] for key in keys))
    return 0.,maximum*1.10


def plots(summary,pairs,out,only=None):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    n=len(summary['paired_rows'])
    def save(fig,name):
        if only is None or name in only:
            for ext in ('pdf','png'):fig.savefig(out/f'{name}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.7),sharey=True)
    ratio_panel(axes[0],summary,['frobenius_ratio','spectral_ratio'],['Frobenius norm','Spectral estimate'],'#2166ac')
    ratio_panel(axes[1],summary,['scaled_frobenius_ratio','scaled_spectral_ratio'],['Scaled Frobenius','Scaled spectral'],'#737373')
    axes[0].set_title('(a) Raw Jacobian J')
    axes[1].set_title('(b) J / fixed basepoint hidden norm')
    axes[0].set_ylim(*ratio_axis_limits(summary,('frobenius_ratio','spectral_ratio','scaled_frobenius_ratio','scaled_spectral_ratio')))
    axes[0].legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('Direct local Jacobian norm comparison',fontsize=12)
    fig.tight_layout(rect=[0,.055,1,1])
    fig.text(.5,.005,f'HumanML3D test: {n} source-motion groups; dots = groups, diamonds = means, bars = 95% motion-group bootstrap intervals.',ha='center',fontsize=8)
    save(fig,'jacobian_norm_ratios')

    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),sharey=True)
    ratio_panel(axes[0],summary,[f'{k}_direction_ratio' for k in ('high','low','random')],['High attribution','Low attribution','Random'],'#2166ac')
    ratio_panel(axes[1],summary,[f'{k}_scaled_direction_ratio' for k in ('high','low','random')],['High attribution','Low attribution','Random'],'#737373')
    axes[0].set_title('(a) Raw directional gain\n' + r'$\dfrac{\|J\delta\|_2}{\|\delta\|_2}$', pad=12, fontsize=12)
    axes[1].set_title('(b) Gain scaled by fixed basepoint hidden norm\n' + r'$\dfrac{\|J\delta\|_2}{\|\delta\|_2\,\|G_K(H)\|_F}$', pad=12, fontsize=12)
    axes[0].set_ylim(*ratio_axis_limits(summary,('high_direction_ratio','low_direction_ratio','random_direction_ratio','high_scaled_direction_ratio','low_scaled_direction_ratio','random_scaled_direction_ratio')))
    axes[0].set_xlabel('Shared baseline-ranked word position')
    axes[1].set_xlabel('Shared baseline-ranked word position')
    axes[0].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=8,loc='upper right')
    fig.suptitle('Matched-direction Jacobian-vector products',fontsize=12)
    fig.tight_layout(rect=[0,.055,1,1])
    fig.text(.5,.005,f'{n} test motion groups; exactly matched words and directions; three directions averaged per group; all groups retained.',ha='center',fontsize=8)
    save(fig,'matched_direction_jvp')

    fig,axes=plt.subplots(1,2,figsize=(10.3,3.2),sharey=True)
    for ax,(model,rows) in zip(axes,pairs.items()):
        for row in rows:
            for run in row['spectral']['restarts']:
                ax.plot(np.arange(1,len(run['trace'])+1),np.array(run['trace'])/row['spectral']['norm_estimate'],color='#2166ac',alpha=.15,lw=.9)
        ax.axhline(1,color='#555555',linestyle='--',lw=1)
        ax.set_title('T2M-GPT' if model=='baseline' else 'SATO++')
        ax.set_xlabel('Power iteration')
        ax.set_ylabel('Singular-value estimate / best final estimate')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Power-iteration convergence (all groups and restarts)',fontsize=12)
    fig.tight_layout()
    save(fig,'power_iteration_convergence')

    variants=summary['gradient_audit']
    keys=list(variants['create_graph_false_actual']['layers'])
    def order(key):
        if key=='text_encoder':return (0,0)
        if key=='conditioning_and_motion_embeddings':return (1,0)
        if key.startswith('trans_base.blocks.'):return (2,int(key.rsplit('.',1)[1]))
        if key.startswith('trans_head.blocks.'):return (3,int(key.rsplit('.',1)[1]))
        return (4,{'final_layernorm':0,'generation_head':1,'projection_head':2,'predictor_head':3}[key])
    keys=sorted(keys,key=order)
    norms=np.full((2,len(keys)),np.nan)
    for i,variant in enumerate(['create_graph_false_actual','create_graph_true_diagnostic']):
        for j,key in enumerate(keys):
            row=variants[variant]['layers'][key]
            if row['grad_l2_norm'] is not None and row['grad_l2_norm']>0:norms[i,j]=row['grad_l2_norm']
    valid=norms[np.isfinite(norms)]
    fig,ax=plt.subplots(figsize=(11.4,2.9))
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#eeeeee')
    image=ax.imshow(norms,aspect='auto',cmap=cmap,norm=LogNorm(vmin=float(valid.min()),vmax=float(valid.max())))
    for i,variant in enumerate(['create_graph_false_actual','create_graph_true_diagnostic']):
        for j,key in enumerate(keys):
            row=variants[variant]['layers'][key]
            if row['status']=='frozen':ax.text(j,i,'F',ha='center',va='center',fontsize=9,color='#555555')
            elif row['status']=='disconnected':ax.text(j,i,'x',ha='center',va='center',fontsize=9,color='#555555')
            elif row['status']=='connected_zero':ax.text(j,i,'0',ha='center',va='center',fontsize=9,color='#555555')
    labels=[]
    for key in keys:
        if key.startswith('trans_base.blocks.'):labels.append('B'+str(int(key.rsplit('.',1)[1])+1))
        elif key.startswith('trans_head.blocks.'):labels.append('B'+str(int(key.rsplit('.',1)[1])+10))
        else:labels.append({'text_encoder':'Text','conditioning_and_motion_embeddings':'Inputs','final_layernorm':'Norm','generation_head':'Gen','projection_head':'Proj','predictor_head':'Pred'}[key])
    ax.set_xticks(range(len(keys)),labels,rotation=45,ha='right')
    ax.set_yticks([0,1],['Checkpoint configuration','Graph-enabled diagnostic'])
    ax.set_xlabel('Text encoder, AR blocks B1–B18, generation head, projection and predictor')
    fig.colorbar(image,ax=ax,pad=.015,fraction=.025,label='Gradient L2 (log scale)')
    ax.set_title('Where does the weighted attribution objective backpropagate?')
    fig.tight_layout(rect=[0,.09,1,1])
    fig.text(.5,.01,'F = frozen; x = disconnected (not a small gradient). Fixed real batch: 2 preserving + 1 altering pair. Diagnostic row is not retraining.',ha='center',fontsize=8)
    save(fig,'attribution_gradient_reachability')


def run(args):
    source=Path(args.input);out=source/'analysis';out.mkdir(exist_ok=True)
    summary,pairs=create_summary(source,Path(args.gradient_audit),args.bootstrap)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (out/'paired_ratios.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary['paired_rows'][0]));writer.writeheader();writer.writerows(summary['paired_rows'])
    with (out/'attribution_gradient_layers.csv').open('w') as handle:
        writer=csv.writer(handle);writer.writerow(['configuration','module','status','gradient_l2','sgd_gradient_l2','connected_tensors','disconnected_tensors'])
        for variant,audit in summary['gradient_audit'].items():
            for key,row in audit['layers'].items():writer.writerow([variant,key,row['status'],row['grad_l2_norm'],row['sgd_update_gradient_l2'],row['connected_tensors'],row['disconnected_tensors']])
    plots(summary,pairs,out)
    print(json.dumps({'statistics':summary['statistics'],'quality':summary['quality']},indent=2))
    print('JACOBIAN_ANALYSIS_COMPLETE '+str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--gradient-audit',required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    run(p.parse_args())