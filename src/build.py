import pandas as pd, numpy as np, json, glob, os, re, yaml
D='data/'
v=pd.read_csv(D+'paichecker/labelled_data/swe-bench-analysis.csv').dropna(subset=['instance_id']).reset_index(drop=True)
base=v[['repo','instance_id','patch','test_patch','problem_statement','hints_text','created_at','version','FAIL_TO_PASS','PASS_TO_PASS','difficulty','label']].copy()
# response matrix from the live experiments repo
rows={};meta={}
for f in sorted(glob.glob(D+'experiments/evaluation/verified/*/results/results.json')):
    name=f.split('/')[-3]; r=json.load(open(f)); res=set(r.get('resolved',[]))
    rows[name]=base.instance_id.isin(res).astype(int).values
    m={}
    mp=os.path.dirname(os.path.dirname(f))+'/metadata.yaml'
    if os.path.exists(mp):
        try: y=yaml.safe_load(open(mp)); m={'name':(y.get('info') or {}).get('name'),'tags':y.get('tags')}
        except Exception as e: m={'err':str(e)}
    meta[name]=m; 
    extra=res-set(base.instance_id)
    if extra: print('extra ids',name,len(extra))
cost={};calls={}
for f in sorted(glob.glob(D+'experiments/evaluation/verified/*/per_instance_details.json')):
    name=f.split('/')[-2]
    if name in rows: continue
    d=json.load(open(f))
    rows[name]=np.array([int(bool(d.get(i,{}).get('resolved',False))) for i in base.instance_id])
    cost[name]=[d.get(i,{}).get('cost',np.nan) for i in base.instance_id]
    calls[name]=[d.get(i,{}).get('api_calls',np.nan) for i in base.instance_id]
    mp=os.path.dirname(f)+'/metadata.yaml'
    y=yaml.safe_load(open(mp)); meta[name]={'name':(y.get('info') or {}).get('name'),'tags':y.get('tags'),'bash_only':True}
pd.DataFrame(cost,index=base.instance_id).to_csv('out/bash_cost.csv');pd.DataFrame(calls,index=base.instance_id).to_csv('out/bash_calls.csv')
R=pd.DataFrame(rows,index=base.instance_id)
R.to_csv('out/response_matrix.csv'); json.dump(meta,open('out/sub_meta.json','w'),default=str)
base.to_pickle('out/base.pkl')
print(R.shape, R.sum().describe().round(1).to_dict())
# compare to paichecker's copy
common=[c for c in R.columns if c in v.columns]
diff=sum(((v[c]=='yes').astype(int).values!=R[c].values).sum() for c in common)
print('common subs',len(common),'cell diffs',diff,'new subs',[c for c in R.columns if c not in v.columns], 'missing', [c for c in v.columns[13:-2] if c not in R.columns])
