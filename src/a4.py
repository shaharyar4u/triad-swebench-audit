import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
F=pd.read_pickle('out/feat3.pkl'); R=pd.read_csv('out/response_matrix.csv',index_col=0); R=R.loc[F.index,R.sum()>=1]; OUT={}
# ---- replicate PAIChecker re-ranking on its 131 agents + null
pc=pd.read_csv('data/paichecker/labelled_data/swe-bench-analysis.csv',nrows=1).columns[13:-2]
R131=R[[c for c in pc]]; mis=F.S_any.values==1
def rk(M): return M.mean().rank(ascending=False,method='min')
def rk1(M): return M.mean().rank(ascending=False,method='first')
full=rk(R131); obs=(full!=rk(R131[~mis])).mean(); obs1=(rk1(R131)!=rk1(R131[~mis])).mean()
rng=np.random.default_rng(7); nul=[];nul1=[];mx=[]
for _ in range(5000):
    idx=rng.choice(500,432,replace=False); S=R131.iloc[idx]; nul.append((full!=rk(S)).mean()); nul1.append((rk1(R131)!=rk1(S)).mean()); mx.append((full-rk(S)).abs().max())
nul=np.array(nul); obsmax=(full-rk(R131[~mis])).abs().max()
OUT['pai_replication']=dict(n_agents=R131.shape[1],observed_changed=round(float(obs),3),observed_changed_first=round(float(obs1),3),null_mean=round(float(nul.mean()),3),null_lo=round(float(np.quantile(nul,.025)),3),null_hi=round(float(np.quantile(nul,.975)),3),p_ge=float((nul>=obs).mean()),null_first_mean=round(float(np.mean(nul1)),3),obs_max=int(obsmax),null_max_mean=round(float(np.mean(mx)),2),p_max=float((np.array(mx)>=obsmax).mean()),
    score_shift_obs=round(float(100*(R131[~mis].mean()-R131.mean()).mean()),2))
print(OUT['pai_replication'])
# pass-rate-matched null: remove random instances matched on pass-rate decile (so removal is equally 'hard')
dec=pd.qcut(F.pass_rate.rank(method='first'),10,labels=False).values; d=F.defect.values==1; fullA=rk(R); obsA=(fullA!=rk(R[~d])).mean(); tauA=stats.kendalltau(R.mean(),R[~d].mean())[0]
cnt=pd.Series(dec[d]).value_counts(); nm=[];nt=[]
for _ in range(2000):
    drop=np.concatenate([rng.choice(np.where(dec==k)[0],c,replace=False) for k,c in cnt.items()]); keep=np.setdiff1d(np.arange(500),drop); nm.append((fullA!=rk(R.iloc[keep])).mean()); nt.append(stats.kendalltau(R.mean(),R.iloc[keep].mean())[0])
OUT['matched_null']=dict(obs_changed=round(float(obsA),3),null_changed=round(float(np.mean(nm)),3),p=float((np.array(nm)>=obsA).mean()),obs_tau=round(float(tauA),4),null_tau=round(float(np.mean(nt)),4),p_tau=float((np.array(nt)<=tauA).mean()))
print(OUT['matched_null'])
# ---- triage: can cheap signals find defects?
static=['g_files','g_hunks','g_churn','g_units','g_dirs','t_files','t_add','t_asserts','t_lit_asserts','t_f2p','t_p2p','t_newtests','ps_words','ps_has_code','hints_words']
resp=['pass_rate','irc','disc','bdiff']
F['irc']=F.irc.fillna(0)
Xs=np.log1p(F[static].astype(float)); Xr=F[resp].astype(float)
diff=pd.get_dummies(F.difficulty,dtype=float)
sets={'static':Xs,'response':Xr,'human effort label':diff,'static+response':pd.concat([Xs,Xr],axis=1),'all':pd.concat([Xs,Xr,diff],axis=1)}
targets={'PAIChecker misaligned':F.S_any,'Issue (I)':F.I,'Patch label (SC or DP)':((F.P_scope==1)|(F.P_defect==1)).astype(int),'UTBoost insufficient tests':F.T_utboost,'Any axis':F.defect}
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=0); tri={}
for tn,y in targets.items():
    tri[tn]={'n_pos':int(y.sum())}
    for sn,Xm in sets.items():
        aucs=[];aps=[]
        for tr,te in cv.split(Xm,y):
            mdl=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000,class_weight='balanced')).fit(Xm.iloc[tr],y.iloc[tr]); p=mdl.predict_proba(Xm.iloc[te])[:,1]; aucs.append(roc_auc_score(y.iloc[te],p)); aps.append(average_precision_score(y.iloc[te],p))
        tri[tn][sn]=dict(auc=round(float(np.mean(aucs)),3),sd=round(float(np.std(aucs)),3),ap=round(float(np.mean(aps)),3))
    aucs=[]
    Xm=sets['all']
    for tr,te in list(cv.split(Xm,y))[:15]:
        g=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0).fit(Xm.iloc[tr],y.iloc[tr]); aucs.append(roc_auc_score(y.iloc[te],g.predict_proba(Xm.iloc[te])[:,1]))
    tri[tn]['all_gbm']=dict(auc=round(float(np.mean(aucs)),3))
OUT['triage']=tri
print(pd.DataFrame({t:{s:(v['auc'] if isinstance(v,dict) else v) for s,v in d.items()} for t,d in tri.items()}).T.to_string())
# review budget: how many of the PAIChecker-misaligned are found reviewing top-k by the 'all' model (out-of-fold)
y=F.S_any; Xm=sets['all']; p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000,class_weight='balanced')),Xm,y,cv=5,method='predict_proba')[:,1]
o=np.argsort(-p); OUT['budget']={int(k):round(float(y.values[o[:k]].sum()/y.sum()),3) for k in [50,100,150,250]}
print(OUT['budget'])
json.dump(OUT,open('out/res_a4.json','w'),indent=1,default=float)
# list footprint cases for manual check
A=pd.read_csv('out/agent_footprint.csv'); 
print(A[A.subset==1].groupby('instance_id').tool.count().to_string()); print(A[A.disjoint==1].groupby('instance_id').tool.count().to_string())
