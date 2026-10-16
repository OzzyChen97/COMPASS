"""Lock fixed mixed-caption assets for the unified manuscript retest."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_motionx_random_retrieval import digest,GROUP_SEEDS
from tools import train_motionx_text_adaptation_pilot as pilot


def balanced_mix(level_rows,seed=123):
    maps={}
    for level in ('P1','P2','P3'):
        rows=level_rows[level];mapping={r['motion_id']:r for r in rows}
        if len(mapping)!=len(rows):raise ValueError('Duplicate motion IDs')
        if any(r.get('relation')!='preserve' or r.get('level')!=level for r in rows):
            raise ValueError('Invalid rewrite relation or level')
        maps[level]=mapping
    ids=sorted(maps['P1'])
    if any(set(m)!=set(ids) for m in maps.values()):raise ValueError('Incomplete level triples')
    normalize=lambda text:' '.join(text.split()).lower()
    for mid in ids:
        if len({normalize(maps[q][mid]['text_orig']) for q in maps})!=1:
            raise ValueError(f'Original captions differ across levels: {mid}')
    order=np.random.default_rng(seed).permutation(len(ids))
    assignment={ids[int(index)]:('P1','P2','P3')[i%3] for i,index in enumerate(order)}
    return [dict(maps[assignment[mid]][mid]) for mid in ids]


def read_pairs(path):
    rows=json.loads(Path(path).read_text())
    return rows['pairs'] if isinstance(rows,dict) else rows


def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    datasets={};models={}
    for ds,n,root,pattern in [
        ('humanml',4380,ROOT/'dataset/HumanML3D','test_perb_table1_level_{}.json'),
        ('motionx',7679,pilot.DATASET_ROOT,'test_perb_gemini31_max75_{}.json')]:
        pair_root=ROOT/'dataset/Motion-X-322' if ds=='motionx' else root
        paths={q:pair_root/pattern.format(q.lower()) for q in ['P1','P2','P3']}
        mixed=balanced_mix({q:read_pairs(path) for q,path in paths.items()})
        assert len(mixed)==n
        ids={r['motion_id'] for r in mixed};test=pilot.load_split_ids(root,'test')
        forbidden=pilot.load_split_ids(root,'train')|pilot.load_split_ids(root,'val')
        assert ids<=test and not ids&forbidden
        out=a.output/f'{ds}_mixed.json';write_json(out,mixed)
        datasets[ds]=dict(pair_file=str(out.resolve()),pair_sha256=digest(out),samples=n,
            level_counts=dict(Counter(r['level'] for r in mixed)),source_hashes={str(path):digest(path) for path in paths.values()},
            assignment_seed=123,assignment='permuted sorted IDs, round-robin P1/P2/P3, no outcome selection')
    # KIT's historical 734 count is the STRICT resolution of an 830-row file.
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from utils.word_vectorizer import WordVectorizer
    vec=WordVectorizer(str(ROOT/'glove'),'our_vab')
    kit=ROOT/'dataset/KIT-ML'
    paths={'kit':kit/'test_perb_gemini31_fullmotion_mixed_full.json',
           'transfer':kit/'_deprecated_gpt54mini_20260726/test_perb_deterministic_p2p3.json'}
    for ds,path in paths.items():
        rows=read_pairs(path)
        loader=CaptionLockedDATALoader('kit',True,32,vec,pair_data=rows,num_workers=0,length_policy='strict')
        names={entry['motion_id'] for entry in loader.dataset.entries}
        selected=[r for r in rows if r['motion_id'] in names]
        assert len(names)==len(selected)==734
        test=pilot.load_split_ids(kit,'test');forbidden=pilot.load_split_ids(kit,'train')|pilot.load_split_ids(kit,'val')
        assert names<=test and not names&forbidden
        out=a.output/f'{ds}_mixed.json';write_json(out,sorted(selected,key=lambda r:r['motion_id']))
        datasets[ds]=dict(pair_file=str(out.resolve()),pair_sha256=digest(out),samples=734,
            source_file=str(path),source_sha256=digest(path),resolution=loader.dataset.resolution_stats,
            level_counts=dict(Counter(r['level'] for r in selected)))
    mx=json.loads((ROOT/'ablation/motionx_independent_cfg13_20260907/protocol.json').read_text())['models']
    for name,row in mx.items():
        assert digest(row['checkpoint'])==row['sha256'];models['motionx/'+name]=row
    for name in ['t2mgpt','satopp','motiongpt','motiongpt_satopp']:
        row=json.loads((ROOT/f'ablation/humanml_independent_cfg13_20260907/main/{name}/manifest.json').read_text())
        assert row['frozen_verified'] and digest(row['checkpoint'])==row['checkpoint_sha256']
        models['humanml/'+name]=dict(checkpoint=row['checkpoint'],sha256=row['checkpoint_sha256'],
            encoder='T5' if name.startswith('motiongpt') else 'CLIP',config=row['source'].get('config'))
    protocol=dict(version='unified-mixed-cfg13-v1',generation_seed=123,cfg_scale=1.3,
        group_seeds=GROUP_SEEDS,group_size=32,drop_tail=False,query='same text_pair used for generation',
        statistics='one generation; mean R@ across partitions; whole-mixture FID, not mean level FID',
        datasets=datasets,ready_models=models,
        pending=['KIT generator text adaptations','KIT retrieval text adaptation and validation','component text adaptations'],
        table_row_reuse={'ablation/t2mgpt':'humanml/t2mgpt','ablation/satopp':'humanml/satopp',
                        'transfer/t2mgpt':'humanml/t2mgpt','transfer/satopp':'humanml/satopp'})
    write_json(a.output/'manifest.json',protocol)
    print('UNIFIED_MIXED_ASSETS_LOCKED', {k:v['samples'] for k,v in datasets.items()},flush=True)


if __name__=='__main__':main()