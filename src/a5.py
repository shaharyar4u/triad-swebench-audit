import pandas as pd, numpy as np, json, glob, warnings, itertools; warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.stats.multitest import multipletests
F=pd.read_pickle('out/feat3.pkl'); R=pd.read_csv('out/response_matrix.csv',index_col=0); R=R.loc[F.index,R.sum()>=1]; X=R.values.astype(float); n,m=X.shape; OUT={}
rng=np.random.default_rng(11)
def ranks(s): return stats.rankdata(-s,method='min')
full_s=X.mean(0); full_r=ranks(full_s)
def stat(keep):
    s=X[keep].mean(0); return (ranks(s)!=full_r).mean(), stats.kendalltau(full_s,s)[0], s
# 1 BH on group tests
a2=json.load(open('out/res_a2.json')); g=[r for r in a2['groups'] if r['group'] not in('Clean','All')]
pp=[r['p_pass'] for r in g]+[r['p_disc'] for r in g]; adj=multipletests(pp,method='fdr_bh')[1]
OUT['bh']={r['group']:dict(q_pass=round(float(adj[i]),4),q_disc=round(float(adj[i+len(g)]),4)) for i,r in enumerate(g)}
# 2 expected co-occurrence under independence (permutation)
I,T,P=F.I.values,F.Tx.values,F.Px.values; obs=int(((I+T+P)>=2).sum()); sim=[]
for _ in range(20000): sim.append(((rng.permutation(I)+rng.permutation(T)+P)>=2).sum())
sim=np.array(sim); OUT['cooccur']=dict(observed=obs,expected=round(float(sim.mean()),1),lo=int(np.quantile(sim,.025)),hi=int(np.quantile(sim,.975)),p=float((sim>=obs).mean()))
# 4 per-axis removal nulls
def nulltest(mask,B=2000,matched=False):
    d=int(mask.sum()); keep=~mask; oc,ot,_=stat(keep); dec=pd.qcut(F.pass_rate.rank(method='first'),10,labels=False).values; cnt=pd.Series(dec[mask]).value_counts()
    nc=[];nt=[]
    for _ in range(B):
        if matched: drop=np.concatenate([rng.choice(np.where(dec==k)[0],c,replace=False) for k,c in cnt.items()])
        else: drop=rng.choice(n,d,replace=False)
        k=np.ones(n,bool); k[drop]=False; c,t,_=stat(k); nc.append(c); nt.append(t)
    nc=np.array(nc); nt=np.array(nt)
    return dict(removed=d,changed=round(float(oc),3),null_changed=round(float(nc.mean()),3),p_changed=float((nc>=oc).mean()),tau=round(float(ot),4),null_tau=round(float(nt.mean()),4),null_tau_lo=round(float(np.quantile(nt,.025)),4),p_tau=float((nt<=ot).mean()))
masks={'Issue axis':F.I.values==1,'Test axis':F.Tx.values==1,'Patch axis':F.Px.values==1,'PAIChecker misaligned':F.S_any.values==1,'Any axis':F.defect.values==1}
OUT['axis_nulls']={k:dict(random=nulltest(v,1000),matched=nulltest(v,1000,True)) for k,v in masks.items()}
print(pd.DataFrame({k:v['random'] for k,v in OUT['axis_nulls'].items()}).T.to_string())
# 3 power: planted differential bias
def power(delta,sims=120,B=150,d=131):
    hits=0
    for _ in range(sims):
        D=rng.choice(n,d,replace=False); fav=rng.random(m)<0.5; Y=X.copy()
        sub=Y[np.ix_(D,np.where(fav)[0])]; flip=(rng.random(sub.shape)<delta)&(sub==0); sub[flip]=1; Y[np.ix_(D,np.where(fav)[0])]=sub
        sub=Y[np.ix_(D,np.where(~fav)[0])]; flip=(rng.random(sub.shape)<delta)&(sub==1); sub[flip]=0; Y[np.ix_(D,np.where(~fav)[0])]=sub
        fs=Y.mean(0); k=np.ones(n,bool); k[D]=False; ot=stats.kendalltau(fs,Y[k].mean(0))[0]
        nt=[]
        for _ in range(B):
            kk=np.ones(n,bool); kk[rng.choice(n,d,replace=False)]=False; nt.append(stats.kendalltau(fs,Y[kk].mean(0))[0])
        hits+= (np.mean(np.array(nt)<=ot)<0.05)
    return hits/sims
OUT['power']={str(dl):round(power(dl),3) for dl in [0.0,0.05,0.10,0.15,0.25]}; print('power',OUT['power'])
# 5 per-system reliance vs matched null
d=F.defect.values==1; dec=pd.qcut(F.pass_rate.rank(method='first'),10,labels=False).values; cnt=pd.Series(dec[d]).value_counts(); obsd=X[~d].mean(0)-full_s; nd=[]
for _ in range(3000):
    drop=np.concatenate([rng.choice(np.where(dec==k)[0],c,replace=False) for k,c in cnt.items()]); k=np.ones(n,bool); k[drop]=False; nd.append(X[k].mean(0)-full_s)
nd=np.array(nd); p=np.array([(np.abs(nd[:,j]-nd[:,j].mean())>=abs(obsd[j]-nd[:,j].mean())).mean() for j in range(m)]); q=multipletests(p,method='fdr_bh')[1]
OUT['reliance']=dict(n_sig_raw=int((p<.05).sum()),expected_by_chance=round(.05*m,1),n_sig_bh=int((q<.05).sum()),min_q=round(float(q.min()),3))
# 6 KR-20
def kr20(M): k=M.shape[0]; t=M.sum(0); pq=(M.mean(1)*(1-M.mean(1))).sum(); return k/(k-1)*(1-pq/t.var(ddof=0))
top=np.argsort(-full_s)[:30]
OUT['kr20']=dict(full=round(kr20(X),4),clean=round(kr20(X[~d]),4),defect=round(kr20(X[d]),4),defect_sb500=None,top30_full=round(kr20(X[:,top]),3),top30_clean=round(kr20(X[~d][:,top]),3),top30_defect=round(kr20(X[d][:,top]),3))
def sb(r,k): return k*r/(1+(k-1)*r)
OUT['kr20']['top30_clean_sb500']=round(sb(OUT['kr20']['top30_clean'],500/369),3); OUT['kr20']['top30_defect_sb500']=round(sb(OUT['kr20']['top30_defect'],500/131),3)
# 7 local dependence Q3
a=F.disc.values; b=F.bdiff.values; th=pd.read_csv('out/theta.csv',index_col=0).iloc[:,0].reindex(R.columns).values
Pm=1/(1+np.exp(-a[:,None]*(th[None,:]-b[:,None]))); E=X-Pm; ok=E.std(1)>0; C=np.corrcoef(E[ok]); dd=d[ok]; iu=np.triu_indices(ok.sum(),1)
both=dd[iu[0]]&dd[iu[1]]; neither=(~dd[iu[0]])&(~dd[iu[1]]); q3=C[iu]
obsq=q3[both].mean()-q3[neither].mean(); nq=[]
for _ in range(500):
    pd_=rng.permutation(dd); bb=pd_[iu[0]]&pd_[iu[1]]; nn=(~pd_[iu[0]])&(~pd_[iu[1]]); nq.append(q3[bb].mean()-q3[nn].mean())
OUT['q3']=dict(mean_defect_pairs=round(float(q3[both].mean()),4),mean_clean_pairs=round(float(q3[neither].mean()),4),diff=round(float(obsq),4),p=float((np.abs(nq)>=abs(obsq)).mean()))
# same repo confound note: within-repo Q3
# 9 recent systems only
recent=np.array([c[:6]>='202507' for c in R.columns]); Xs=X[:,recent]; fs=Xs.mean(0); ot=stats.kendalltau(fs,Xs[~d].mean(0))[0]; oc=(ranks(fs)!=ranks(Xs[~d].mean(0))).mean(); nt=[];nc=[]
for _ in range(2000):
    k=np.ones(n,bool); k[rng.choice(n,131,replace=False)]=False; s=Xs[k].mean(0); nt.append(stats.kendalltau(fs,s)[0]); nc.append((ranks(fs)!=ranks(s)).mean())
OUT['recent']=dict(n_sys=int(recent.sum()),tau=round(float(ot),4),null_tau=round(float(np.mean(nt)),4),p_tau=float((np.array(nt)<=ot).mean()),changed=round(float(oc),3),null_changed=round(float(np.mean(nc)),3))
# 11 footprint per tool
A=pd.read_csv('out/agent_footprint.csv'); OUT['fp_tool']={t[9:30]:dict(n=int(len(gp)),same=round(gp.same.mean(),3),superset=round(gp.superset.mean(),3),subset=round(gp.subset.mean(),3),disjoint=round(gp.disjoint.mean(),3)) for t,gp in A.groupby('tool')}
# 8 multilingual
L=pd.read_csv('data/paichecker/labelled_data/swe_bench_multilingual.csv').set_index('instance_id'); rows={};cost={}
for f in sorted(glob.glob('data/experiments/evaluation/multilingual/*/per_instance_details.json')):
    dj=json.load(open(f)); nm=f.split('/')[-2]; rows[nm]=[int(bool(dj.get(i,{}).get('resolved',False))) for i in L.index]; cost[nm]=[dj.get(i,{}).get('cost',np.nan) for i in L.index]
M=pd.DataFrame(rows,index=L.index); Cm=pd.DataFrame(cost,index=L.index); M=M.loc[:,M.sum()>0]; Cm=Cm[M.columns]; cov=[ (pd.Series([i in json.load(open(f)) for i in L.index]).mean()) for f in sorted(glob.glob('data/experiments/evaluation/multilingual/*/per_instance_details.json'))[:1]]
mis=(L.human_label!='No Misalignment').values; pr=M.mean(1).values; u=stats.mannwhitneyu(pr[mis],pr[~mis]); Xm=M.values.astype(float); fs=Xm.mean(0)
oc=(ranks(fs)!=ranks(Xm[~mis].mean(0))).mean(); ot=stats.kendalltau(fs,Xm[~mis].mean(0))[0]; nc=[];nt=[]
for _ in range(5000):
    k=np.ones(len(L),bool); k[rng.choice(len(L),int(mis.sum()),replace=False)]=False; s=Xm[k].mean(0); nc.append((ranks(fs)!=ranks(s)).mean()); nt.append(stats.kendalltau(fs,s)[0])
lab=L.human_label
OUT['multilingual']=dict(n_inst=len(L),n_sys=M.shape[1],id_coverage=round(float(cov[0]),3),misaligned=int(mis.sum()),pass_mis=round(float(pr[mis].mean()),3),pass_al=round(float(pr[~mis].mean()),3),cliff=round(float(2*u.statistic/(mis.sum()*(~mis).sum())-1),3),p=float(u.pvalue),never=int((M.sum(1)==0).sum()),never_mis=int(((M.sum(1)==0).values&mis).sum()),
  pass_SC=round(float(pr[lab.str.contains('SC').values].mean()),3),pass_DP=round(float(pr[lab.str.contains('DP').values].mean()),3),pass_IS=round(float(pr[lab.str.contains('IS').values].mean()),3),
  changed=round(float(oc),3),null_changed=round(float(np.mean(nc)),3),p_changed=float((np.array(nc)>=oc).mean()),tau=round(float(ot),3),null_tau=round(float(np.mean(nt)),3),p_tau=float((np.array(nt)<=ot).mean()),
  score_shift_pp=round(float(100*(Xm[~mis].mean(0)-fs).mean()),2),cost_share_mis=round(float(np.nansum(Cm.values[mis])/np.nansum(Cm.values)),3),share_inst=round(float(mis.mean()),3),best=round(float(fs.max()),3))
print({k:OUT[k] for k in ['bh','cooccur','reliance','kr20','q3','recent','fp_tool','multilingual']})
json.dump(OUT,open('out/res_a5.json','w'),indent=1,default=float)
