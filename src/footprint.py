import pandas as pd, numpy as np, json, glob, re, sys
sys.path.insert(0,'src')
F=pd.read_pickle('out/feat.pkl'); R=pd.read_csv('out/response_matrix.csv',index_col=0)
import importlib.util
def parse_files(p):
    out={}
    cur=None
    for ln in (p or '').split('\n'):
        if ln.startswith('diff --git'):
            m=re.match(r'diff --git a/(.*?) b/(.*)',ln); cur=m.group(2) if m else None
            if cur: out[cur]=0
        elif cur and (ln.startswith('+') or ln.startswith('-')) and not ln.startswith(('+++','---')): out[cur]+=1
    return out
TEST=re.compile(r'(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_tests?\.py$|conftest\.py|reproduce|repro_|\.(rst|md|txt|cfg|toml|ini|json|yml|yaml)$',re.I)
gold={r.instance_id:set(r.g_srcpaths) for r in F.itertuples()}; gch=dict(zip(F.instance_id,F.g_churn))
rec=[]
for d in glob.glob('data/patchdiff/data/tool_results/*/'):
    name=d.rstrip('/').split('/')[-1]; col=[c for c in R.columns if c.startswith(name[:20])][0]
    for ln in open(d+'all_preds.jsonl'):
        j=json.loads(ln); i=j['instance_id']
        if i not in gold or R.loc[i,col]!=1: continue
        fs=parse_files(j.get('model_patch') or ''); src={k:v for k,v in fs.items() if not TEST.search(k) and k.endswith('.py')}
        a=set(src); g=gold[i]
        rec.append(dict(instance_id=i,tool=col,n_a=len(a),subset=int(len(a)>0 and a<g),same=int(a==g),disjoint=int(len(a)>0 and not(a&g)),superset=int(a>g),churn_ratio=sum(src.values())/max(gch[i],1)))
A=pd.DataFrame(rec); A.to_csv('out/agent_footprint.csv',index=False)
print(len(A),A[['subset','same','disjoint','superset']].mean().round(3).to_dict(), A.churn_ratio.describe().round(2).to_dict())
I=A.groupby('instance_id').agg(n_pass=('tool','size'),any_subset=('subset','max'),any_disjoint=('disjoint','max'),min_churn_ratio=('churn_ratio','min'))
F=F.merge(I,left_on='instance_id',right_index=True,how='left')
mf=F[F.g_files>1]
print('multi-file gold',len(mf),'with >=1 passing agent patch',mf.n_pass.notna().sum(),'passing with strict subset of gold files',int(mf.any_subset.sum()))
print('instances w/ passing patch',F.n_pass.notna().sum(),'any disjoint',int(F.any_disjoint.sum()))
F.to_pickle('out/feat2.pkl')
