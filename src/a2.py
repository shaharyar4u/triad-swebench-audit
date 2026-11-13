import pandas as pd, numpy as np, json, warnings, itertools; warnings.filterwarnings('ignore')
from scipy import stats, optimize
import statsmodels.api as sm
OUT={}
F=pd.read_pickle('out/feat2.pkl').set_index('instance_id'); R=pd.read_csv('out/response_matrix.csv',index_col=0)
R=R.loc[F.index, R.sum()>=1]; n,m=R.shape; X=R.values.astype(float)
meta=json.load(open('out/sub_meta.json'))
for c in ['any_subset','any_disjoint']: F[c]=F[c].fillna(0).astype(int)
F['has_agent']=F.n_pass.notna().astype(int)
# ---- artefact-centred axes
F['I']=((F.S_IS==1)|(F.S_FP==1)).astype(int)
F['T_lax']=((F.T_utboost==1)|(F.any_disjoint==1)).astype(int); F['T_strict']=F.S_UL
F['Tx']=((F.T_lax==1)|(F.T_strict==1)).astype(int)
F['P_scope']=F.S_SC; F['P_defect']=F.S_DP; F['P_unneeded']=((F.any_subset==1)|(F.g_nonfunc_hunks>0)).astype(int)
F['Px']=((F.P_scope==1)|(F.P_defect==1)|(F.P_unneeded==1)).astype(int)
F['k']=F.I+F.Tx+F.Px; F['defect']=(F.k>0).astype(int); F['clean']=1-F.defect
F['pass_rate']=R.mean(1).values; F['n_solved']=R.sum(1).values
OUT['n_inst']=n; OUT['n_sub']=m
OUT['prev']={k:int(F[k].sum()) for k in ['I','Tx','Px','T_lax','T_strict','T_utboost','any_disjoint','P_scope','P_defect','P_unneeded','any_subset','S_any','defect']}
OUT['prev']['nonfunc']=int((F.g_nonfunc_hunks>0).sum()); OUT['kdist']={int(k):int(v) for k,v in F.k.value_counts().sort_index().items()}
def assoc(x,y):
    t=pd.crosstab(x,y).reindex(index=[0,1],columns=[0,1],fill_value=0).values; o,p=stats.fisher_exact(t); n11=t[1,1]
    po=(t[0,0]+t[1,1])/t.sum(); pe=(t[0].sum()*t[:,0].sum()+t[1].sum()*t[:,1].sum())/t.sum()**2
    return dict(both=int(n11),OR=round(float(o),2),p=float(p),jaccard=round(n11/max(t[1,0]+t[0,1]+n11,1),3),kappa=round((po-pe)/(1-pe),3))
OUT['overlap']={f'{u}-{v}':assoc(F[u],F[v]) for u,v in itertools.combinations(['I','Tx','Px'],2)}
OUT['audit_overlap']={'PAIChecker-UTBoost':assoc(F.S_any,F.T_utboost),'PAIChecker-agentDisjoint':assoc(F.S_any,F.any_disjoint),'PAIChecker-agentSubset':assoc(F.S_any,F.any_subset),'UTBoost-agentDisjoint':assoc(F.T_utboost,F.any_disjoint)}
# venn regions
OUT['venn']={''.join(k):int(((F.I==k[0])&(F.Tx==k[1])&(F.Px==k[2])).sum()) for k in itertools.product([0,1],repeat=3)} if False else {f'I{a}T{b}P{c}':int(((F.I==a)&(F.Tx==b)&(F.Px==c)).sum()) for a,b,c in itertools.product([0,1],repeat=3)}
# footprint headline
mf=F[F.g_files>1]; OUT['footprint']=dict(multi_file=len(mf),multi_with_pass=int(mf.has_agent.sum()),multi_subset=int(mf.any_subset.sum()),inst_with_pass=int(F.has_agent.sum()),disjoint=int(F.any_disjoint.sum()))
# ---- item statistics
irc=[]
for i in range(n):
    x=X[i]; rest=(X.sum(0)-x)/(n-1); irc.append(np.nan if x.std()==0 else np.corrcoef(x,rest)[0,1])
F['irc']=irc
def nll(par):
    a=np.exp(par[:n]); b=par[n:2*n]; th=par[2*n:]
    z=a[:,None]*(th[None,:]-b[:,None]); 
    ll=(X*z-np.logaddexp(0,z)).sum()
    pr=-(par[:n]**2).sum()/(2*0.5**2)-(b**2).sum()/(2*3**2)-(th**2).sum()/2
    p=1/(1+np.exp(-z)); e=X-p
    ga=(e*(th[None,:]-b[:,None])).sum(1)*a-par[:n]/0.25; gb=(-e*a[:,None]).sum(1)-b/9; gt=(e*a[:,None]).sum(0)-th
    return -(ll+pr), -np.concatenate([ga,gb,gt])
x0=np.concatenate([np.zeros(n),-stats.norm.ppf(np.clip(X.mean(1),.01,.99)),stats.zscore(X.mean(0))])
r=optimize.minimize(nll,x0,jac=True,method='L-BFGS-B',options={'maxiter':3000})
a=np.exp(r.x[:n]); b=r.x[n:2*n]; th=r.x[2*n:]; z=a[:,None]*(th[None,:]-b[:,None])
F['disc']=a; F['bdiff']=b; theta=pd.Series(th,index=R.columns)
OUT['irt']=dict(converged=bool(r.success),acc=round(float(((z>0)==(X>.5)).mean()),3),rho_disc_irc=round(float(stats.spearmanr(F.disc,F.irc,nan_policy='omit')[0]),3),rho_theta_score=round(float(stats.spearmanr(th,X.mean(0))[0]),4),disc_q=[round(float(q),2) for q in np.quantile(a,[.1,.5,.9])])
def cliff(x,y):
    u=stats.mannwhitneyu(x,y); return round(2*u.statistic/(len(x)*len(y))-1,3),float(u.pvalue)
clean=F[F.clean==1]; rows=[]
groups={'Clean':clean,'Issue (I)':F[F.I==1],'Test-lax':F[F.T_lax==1],'Patch (P)':F[F.Px==1],'P: scope creep':F[F.P_scope==1],'P: defective gold':F[F.P_defect==1],'P: unneeded change':F[F.P_unneeded==1],'Exactly one axis':F[F.k==1],'Two or more axes':F[F.k>=2],'Any defect':F[F.defect==1],'All':F}
for nm,s in groups.items():
    d,p=cliff(s.pass_rate,clean.pass_rate); d2,p2=cliff(s.disc,clean.disc)
    rows.append(dict(group=nm,n=len(s),pass_rate=round(s.pass_rate.mean(),3),never=int((s.n_solved==0).sum()),irc=round(s.irc.median(),3),disc=round(s.disc.median(),2),b=round(s.bdiff.median(),2),d_pass=d,p_pass=p,d_disc=d2,p_disc=p2))
T1=pd.DataFrame(rows); OUT['groups']=rows; print(T1.to_string())
# at-risk comparison for T_lax (instances with >=1 passing patch among the 3 tools)
ar=F[F.has_agent==1]; OUT['Tlax_atrisk']=dict(n_atrisk=len(ar),n_T=int(ar.T_lax.sum()),pass_T=round(ar[ar.T_lax==1].pass_rate.mean(),3),pass_notT=round(ar[(ar.T_lax==0)&(ar.clean==1)].pass_rate.mean(),3),cliff=cliff(ar[ar.T_lax==1].pass_rate,ar[(ar.T_lax==0)&(ar.clean==1)].pass_rate))
ns=F[F.n_solved==0]; OUT['never']=dict(n=len(ns),I=int(ns.I.sum()),T=int(ns.Tx.sum()),P=int(ns.Px.sum()),any=int(ns.defect.sum()),S_any=int(ns.S_any.sum()),base_rate=round(F.defect.mean(),3),fisher=assoc((F.n_solved==0).astype(int),F.defect))
# ---- regression: does defect status add to size + human effort + repo?
F['logit']=np.log((F.n_solved+.5)/(m-F.n_solved+.5)); F['lchurn']=np.log1p(F.g_churn); F['multi']=(F.g_files>1).astype(int)
dm=pd.get_dummies(F.difficulty,drop_first=False,dtype=float).drop(columns=['<15 min fix']); rp=pd.get_dummies(F.repo,dtype=float); rp=rp.drop(columns=['django/django'])
def ols(cols,extra=True):
    Xd=F[cols].astype(float)
    if extra: Xd=pd.concat([Xd,F[['lchurn','multi']],np.log1p(F[['t_f2p','ps_words']]),dm,rp],axis=1)
    mod=sm.OLS(F.logit,sm.add_constant(Xd)).fit(cov_type='HC3'); return mod
m0=ols([],True); m1=ols(['I','T_lax','P_scope','P_defect','P_unneeded'],True); m1u=ols(['I','T_lax','P_scope','P_defect','P_unneeded'],False)
OUT['reg']={'r2_controls':round(m0.rsquared,3),'r2_full':round(m1.rsquared,3),'coef':{k:dict(beta=round(m1.params[k],3),lo=round(m1.conf_int().loc[k,0],3),hi=round(m1.conf_int().loc[k,1],3),p=float(m1.pvalues[k]),OR=round(float(np.exp(m1.params[k])),2)) for k in ['I','T_lax','P_scope','P_defect','P_unneeded','lchurn','multi']},'coef_unadj':{k:dict(beta=round(m1u.params[k],3),p=float(m1u.pvalues[k])) for k in ['I','T_lax','P_scope','P_defect','P_unneeded']}}
print(json.dumps(OUT['reg'],indent=0)[:1500])
m2=ols(['I','T_lax','Px'],True); F['IxP']=F.I*F.Px; m3=ols(['I','T_lax','Px','IxP'],True); OUT['reg']['interaction_IxP']=dict(beta=round(m3.params['IxP'],3),p=float(m3.pvalues['IxP']))
F.to_pickle('out/feat3.pkl'); theta.to_csv('out/theta.csv'); json.dump(OUT,open('out/res_a2.json','w'),indent=1,default=float)
print({k:OUT[k] for k in ['prev','kdist','overlap','audit_overlap','venn','footprint','irt','Tlax_atrisk','never']})
