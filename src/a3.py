import pandas as pd, numpy as np, json, warnings, re; warnings.filterwarnings('ignore')
from scipy import stats
F=pd.read_pickle('out/feat3.pkl'); R=pd.read_csv('out/response_matrix.csv',index_col=0); R=R.loc[F.index,R.sum()>=1]
meta=json.load(open('out/sub_meta.json')); n,m=R.shape; OUT={}
score=R.mean(); order=score.sort_values(ascending=False)
date=pd.to_datetime(pd.Series({c:c[:8] for c in R.columns}),format='%Y%m%d'); bash=pd.Series({c:bool(meta.get(c,{}).get('bash_only')) or 'mini-v' in c for c in R.columns})
defect=F.defect.values==1; clean=~defect
def fis(a,b):
    t=pd.crosstab(pd.Series(a),pd.Series(b)).reindex(index=[False,True],columns=[False,True],fill_value=0).values; o,p=stats.fisher_exact(t); return round(float(o),2),float(p)
# ---- frontier informativeness
fr={}
for K in [5,10,20,30]:
    top=order.index[:K]; s=R[top].sum(1).values
    allp=s==K; allf=s==0; inf=~(allp|allf)
    fr[K]=dict(all_pass=int(allp.sum()),all_fail=int(allf.sum()),informative=int(inf.sum()),
        defect_share_inf=round(defect[inf].mean(),3),defect_share_allfail=round(defect[allf].mean(),3),defect_share_allpass=round(defect[allp].mean(),3),
        fisher_inf_or_fail_vs_pass=fis(~allp,defect),
        I_fail=int(F.I.values[allf].sum()),P_fail=int(F.Px.values[allf].sum()),T_fail=int(F.Tx.values[allf].sum()),
        Tlax_share_allpass=round(F.T_lax.values[allp].mean(),3),Tlax_share_rest=round(F.T_lax.values[~allp].mean(),3))
OUT['frontier']=fr; print(json.dumps(fr,indent=0)[:1400])
top10=order.index[:10]; s10=R[top10].sum(1).values
# headroom ceiling: for best system, composition of its failures
best=order.index[0]; fail=R[best].values==0
OUT['best']=dict(name=best,score=round(score[best],3),fails=int(fail.sum()),fails_defect=int(defect[fail].sum()),fails_I=int(F.I.values[fail].sum()),fails_P=int(F.Px.values[fail].sum()),fails_Plabel=int(((F.P_scope==1)|(F.P_defect==1)).values[fail].sum()),fails_PAI=int(F.S_any.values[fail].sum()))
# ---- per-submission subset scores
sub=pd.DataFrame({'score':score,'clean':R[clean].mean(),'defect':R[defect].mean(),'I':R[F.I.values==1].mean(),'Tlax':R[F.T_lax.values==1].mean(),'Plab':R[((F.P_scope==1)|(F.P_defect==1)).values].mean(),'date':date,'bash':bash})
sub['gap']=sub.clean-sub.defect; sub['theta']=pd.read_csv('out/theta.csv',index_col=0).iloc[:,0]
OUT['gap']=dict(mean_clean=round(sub.clean.mean(),3),mean_defect=round(sub.defect.mean(),3),mean_gap=round(sub.gap.mean(),3),rho_gap_score=[round(float(x),3) for x in stats.spearmanr(sub.gap,sub.score)],shift_pp=dict(mean=round(100*(sub.clean-sub.score).mean(),2),min=round(100*(sub.clean-sub.score).min(),2),max=round(100*(sub.clean-sub.score).max(),2)))
# relative risk on defect vs clean for weak vs strong systems
q=pd.qcut(sub.score,4,labels=['Q1','Q2','Q3','Q4']); OUT['quartile']={k:dict(n=int(len(g)),score=round(g.score.mean(),3),clean=round(g.clean.mean(),3),I=round(g.I.mean(),3),Plab=round(g.Plab.mean(),3),Tlax=round(g.Tlax.mean(),3),ratio_I=round(g.I.mean()/g.clean.mean(),3),ratio_P=round(g.Plab.mean()/g.clean.mean(),3)) for k,g in sub.groupby(q)}
print(pd.DataFrame(OUT['quartile']).T.to_string())
# temporal within bash-only (fixed scaffold)
bo=sub[sub.bash].sort_values('date'); OUT['bash']=dict(n=len(bo),rho_I_ratio_date=[round(float(x),3) for x in stats.spearmanr((bo.I/bo.clean),bo.date.astype('int64'))],rho_P_ratio_date=[round(float(x),3) for x in stats.spearmanr((bo.Plab/bo.clean),bo.date.astype('int64'))],rho_P_ratio_score=[round(float(x),3) for x in stats.spearmanr((bo.Plab/bo.clean),bo.score)],rho_I_ratio_score=[round(float(x),3) for x in stats.spearmanr((bo.I/bo.clean),bo.score)])
OUT['all_ratio']=dict(rho_P_ratio_score=[round(float(x),3) for x in stats.spearmanr((sub.Plab/sub.clean),sub.score)],rho_I_ratio_score=[round(float(x),3) for x in stats.spearmanr((sub.I/sub.clean),sub.score)])
# ---- leaderboard re-ranking
r_full=score.rank(ascending=False,method='min'); r_clean=sub.clean.rank(ascending=False,method='min'); d=(r_full-r_clean)
# drop-only-deflating (I + P labels) variant
keep2=~((F.I==1)|(F.P_scope==1)|(F.P_defect==1)).values; r_k2=R[keep2].mean().rank(ascending=False,method='min')
OUT['rerank']=dict(n_clean=int(clean.sum()),changed=int((d!=0).sum()),share=round((d!=0).mean(),3),max_up=int(d.max()),max_down=int(d.min()),mean_abs=round(d.abs().mean(),2),kendall=round(float(stats.kendalltau(score,sub.clean)[0]),4),
   top10_changed=int((d[top10]!=0).sum()),top10_set_overlap=len(set(top10)&set(sub.clean.sort_values(ascending=False).index[:10])),
   top30_mean_abs=round(d[order.index[:30]].abs().mean(),2),rest_mean_abs=round(d[order.index[30:]].abs().mean(),2),kendall_top30=round(float(stats.kendalltau(score[order.index[:30]],sub.clean[order.index[:30]])[0]),3),
   paichecker_only=dict(changed=int((r_full!=R[F.S_any.values==0].mean().rank(ascending=False,method='min')).sum()),kendall_top30=round(float(stats.kendalltau(score[order.index[:30]],R[F.S_any.values==0].mean()[order.index[:30]])[0]),3)))
# random-removal null: remove 131 random instances
rng=np.random.default_rng(1); nullk=[];nullk30=[];nullch=[]
for _ in range(2000):
    idx=rng.choice(n,clean.sum(),replace=False); s=R.iloc[idx].mean(); nullk.append(stats.kendalltau(score,s)[0]); nullk30.append(stats.kendalltau(score[order.index[:30]],s[order.index[:30]])[0]); nullch.append((r_full!=s.rank(ascending=False,method='min')).mean())
OUT['rerank']['null']=dict(kendall_mean=round(float(np.mean(nullk)),4),kendall_p=float((np.array(nullk)<=OUT['rerank']['kendall']).mean()),k30_mean=round(float(np.mean(nullk30)),3),k30_lo=round(float(np.quantile(nullk30,.025)),3),k30_p=float((np.array(nullk30)<=OUT['rerank']['kendall_top30']).mean()),changed_mean=round(float(np.mean(nullch)),3))
print(OUT['rerank']); print(OUT['gap'],OUT['bash'],OUT['all_ratio'],OUT['best'])
# ---- cost (bash-only)
C=pd.read_csv('out/bash_cost.csv',index_col=0).loc[F.index]; A=pd.read_csv('out/bash_calls.csv',index_col=0).loc[F.index]
C=C[[c for c in C.columns if c in R.columns]]; A=A[C.columns]
tot=C.sum().sum(); OUT['cost']=dict(n_runs=C.shape[1],total_usd=round(float(tot),0),share_defect=round(float(C[defect].sum().sum()/tot),3),share_inst_defect=round(defect.mean(),3),
  mean_cost_clean=round(float(np.nanmean(C[clean].values)),3),mean_cost_defect=round(float(np.nanmean(C[defect].values)),3),
  mean_calls_clean=round(float(np.nanmean(A[clean].values)),1),mean_calls_defect=round(float(np.nanmean(A[defect].values)),1),
  mean_calls_I=round(float(np.nanmean(A[F.I.values==1].values)),1),mean_calls_Plab=round(float(np.nanmean(A[((F.P_scope==1)|(F.P_defect==1)).values].values)),1),mean_calls_Tlax=round(float(np.nanmean(A[F.T_lax.values==1].values)),1),
  unresolved_cost_share_defect=round(float((C[defect]*(1-R.loc[F.index[defect],C.columns])).sum().sum()/(C*(1-R[C.columns])).sum().sum()),3),
  mw_calls=float(stats.mannwhitneyu(A[defect].mean(1),A[clean].mean(1)).pvalue))
print(OUT['cost'])
sub.to_csv('out/submissions.csv'); json.dump(OUT,open('out/res_a3.json','w'),indent=1,default=float)
