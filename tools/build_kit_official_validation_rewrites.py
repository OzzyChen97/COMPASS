"""Build KIT official-validation rewrites using the existing approved recipe.

Uses the existing recipe's paired provider configuration; never forwards an
unrelated environment credential to a different provider. No test text is read.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import hashlib
import json
from pathlib import Path
import sys
import time
import re
from urllib.parse import urlsplit, urlunsplit

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset.HumanML3D import build_full_pairs as recipe
from tools import train_motionx_text_adaptation_pilot as pilot


def read_api_configuration(path):
    """Read only the API-labelled entries; never return the Overleaf token."""
    text=Path(path).read_text()
    def field(label):
        match=re.search(r'^\s*'+label+r'\s*:\s*\n[ \t]*([^\r\n]+)',text,re.I|re.M)
        if not match:raise ValueError('Missing labelled API configuration field')
        return match.group(1).strip()
    key=field('api key');url=field('api url');parts=urlsplit(url)
    if not key or parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Invalid API configuration')
    path=parts.path.rstrip('/')
    if not path:path='/v1'
    return key,urlunsplit((parts.scheme,parts.netloc,path,'',''))


def parse_response_json(text):
    text=text.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text)


def constrained_request(caption):
    n=len(caption.split())
    return json.dumps({'inputs':[{'index':0,'caption':caption}],
        'constraints':f'Every paraphrase, including P3, must contain between {max(1,math.ceil(.3*n))} and {3*n} whitespace-separated words. Keep the action explicit and preserve every semantic detail. Narrative framing must fit this length limit.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--credentials-file',type=Path)
    p.add_argument('--model',default='gemini-3.1-flash-lite')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.workers<1 or a.limit<0:p.error('Invalid worker count or limit')
    if a.credentials_file:
        from openai import OpenAI
        api_key,base_url=read_api_configuration(a.credentials_file)
        def make_client():return OpenAI(api_key=api_key,base_url=base_url,timeout=90.,max_retries=0)
    else:
        base_url=recipe.BASE_URL
        def make_client():return recipe.get_client().with_options(timeout=90.,max_retries=0)
    if a.resume and not a.output.is_dir():p.error('Resume directory does not exist')
    a.output.mkdir(parents=True,exist_ok=a.resume);raw=a.output/'per_motion';raw.mkdir(exist_ok=a.resume)
    root=ROOT/'dataset/KIT-ML';val=pilot.load_split_ids(root,'val')
    assert not val&(pilot.load_split_ids(root,'train')|pilot.load_split_ids(root,'test'))
    selected=[];skipped={}
    for mid in sorted(val):
        motion=np.load(root/'new_joint_vecs'/f'{mid}.npy',mmap_mode='r')
        caps=sorted(pilot.full_motion_captions(root/'texts'/f'{mid}.txt'))
        if len(motion)<24 or len(motion)>=200 or not caps:
            skipped[mid]=dict(frames=len(motion),full_captions=len(caps));continue
        i=int(hashlib.sha256(f'123:{mid}'.encode()).hexdigest(),16)%len(caps)
        selected.append((mid,caps[i]))
    if a.limit:selected=selected[:a.limit]
    metadata=dict(split='official_val',official_motion_count=len(val),eligible=len(selected),skipped=skipped,
                  smoke=bool(a.limit),model=a.model,filter_model=a.model,base_url=base_url,
                  recipe_sha256=pilot.sha256_file(ROOT/'dataset/HumanML3D/build_full_pairs.py'),
                  usage='validation only; exclude all rows from evaluator/generator training',
                  selected_captions_sha256=pilot.canonical_sha256(selected))
    if a.resume:
        previous=json.loads((a.output/'manifest.json').read_text())
        for key,value in metadata.items():
            if previous.get(key)!=value:raise ValueError('Resume protocol does not match existing validation assets')
        if previous.get('complete'):
            assert pilot.sha256_file(a.output/'val_pairs.json')==previous['output_sha256']
            print('KIT_OFFICIAL_VALIDATION_ALREADY_COMPLETE',flush=True);return
    else:
        (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def one(item):
        mid,caption=item
        cached=raw/f'{mid}.json'
        if a.resume and cached.exists():
            rows=json.loads(cached.read_text())['selected']
            assert len(rows)==3 and {r['level'] for r in rows}=={'P1','P2','P3'}
            assert all(r['motion_id']==mid and r['text_orig']==caption and r['relation']=='preserve' for r in rows)
            return rows
        client=make_client()
        accepted_by_level={};attempt_history=[]
        for attempt in range(5):
            try:
                response=client.chat.completions.create(model=a.model,temperature=.7,max_tokens=2000,
                    response_format={'type':'json_object'},messages=[{'role':'system','content':recipe.PRESERVING_PROMPT},
                    {'role':'user','content':constrained_request(caption)}])
                result=parse_response_json(response.choices[0].message.content)['results'];assert len(result)==1
                candidates=[]
                for level in ['P1','P2','P3']:
                    for phrase in result[0][level]:
                        if isinstance(phrase,str) and recipe.basic_filter(caption,phrase):
                            candidates.append(dict(motion_id=mid,text_orig=caption,text_pair=phrase.strip(),relation='preserve',level=level))
                judged=client.chat.completions.create(model=a.model,temperature=0.,max_tokens=600,
                    response_format={'type':'json_object'},messages=[{'role':'system','content':recipe.FILTER_SYSTEM_PROMPT},
                    {'role':'user','content':json.dumps(candidates)}])
                flags=parse_response_json(judged.choices[0].message.content)['results']
                assert len(flags)==len(candidates) and all(type(x) is bool for x in flags)
                attempt_history.append(dict(raw_response=result,candidates=candidates,accepted=flags))
                for row,ok in zip(candidates,flags):
                    if ok:accepted_by_level.setdefault(row['level'],row)
                if set(accepted_by_level)!={'P1','P2','P3'}:
                    print('KIT_LEVELS_PENDING',mid,sorted({'P1','P2','P3'}-set(accepted_by_level)),flush=True)
                    raise ValueError('Incomplete semantic-filtered levels')
                rows=[accepted_by_level[level] for level in ['P1','P2','P3']]
                (raw/f'{mid}.json').write_text(json.dumps(dict(attempt_history=attempt_history,selected=rows),indent=2)+'\n')
                print('KIT_VALIDATION_REWRITTEN',mid,flush=True);return rows
            except Exception as exc:
                # Do not print API error payloads or credential-bearing requests.
                kind=type(exc).__name__
                if kind in ['AuthenticationError','PermissionDeniedError']:
                    raise RuntimeError('Configured rewrite-provider credential was rejected') from None
                print('KIT_REWRITE_RETRY',mid,attempt+1,kind,flush=True)
                if attempt==4:
                    (a.output/f'failed_{mid}.json').write_text(json.dumps(dict(attempt_history=attempt_history,error_type=kind),indent=2)+'\n')
                    raise RuntimeError(f'Validation rewrite failed for {mid}: {kind}') from None
                time.sleep(2*(attempt+1))
    rows=[];failures={}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,item):item[0] for item in selected}
        for future in as_completed(futures):
            try:rows.extend(future.result())
            except Exception as exc:
                failures[futures[future]]=type(exc).__name__
                print('KIT_MOTION_FAILED_CONTINUING',futures[future],type(exc).__name__,flush=True)
    if failures:
        metadata.update(complete=False,completed_motions=len(rows)//3,failed_motions=failures)
        (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        raise RuntimeError(f'{len(failures)} validation motions remain incomplete; accepted triples are retained for resume')
    rows.sort(key=lambda row:(row['motion_id'],row['level']))
    assert len(rows)==len(selected)*3
    for i,row in enumerate(rows):row['pair_id']=i+1
    out=a.output/'val_pairs.json';out.write_text(json.dumps(rows,indent=2)+'\n')
    metadata.update(complete=True,pairs=len(rows),output_sha256=pilot.sha256_file(out))
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('KIT_OFFICIAL_VALIDATION_READY',len(selected),len(rows),flush=True)


if __name__=='__main__':main()