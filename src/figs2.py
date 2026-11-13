import pandas as pd, numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
F=pd.read_pickle('out/feat3.pkl'); R=pd.read_csv('out/response_matrix.csv',index_col=0); R=R.loc[F.index,R.sum()>=1]; S=pd.read_csv('out/submissions.csv',index_col=0)
a2=json.load(open('out/res_a2.json'));a3=json.load(open('out/res_a3.json'));a4=json.load(open('out/res_a4.json'));a5=json.load(open('out/res_a5.json'))
plt.rcParams.update({'font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':200,'savefig.bbox':'tight'})
B='#2b6cb0';G='#9aa0a6';O='#c05621';K='#333'
# Fig1 overview
fig,ax=plt.subplots(figsize=(8.4,3.5)); ax.axis('off')
def box(x,y,w,h,t,c='#eef2f7'): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02',fc=c,ec='#555',lw=.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
box(.0,.70,.2,.24,'SWE-bench Verified\n500 tasks: issue,\ntests, gold patch'); box(.0,.38,.2,.24,'Prior audits\nPAIChecker (68)\nUTBoost (26)'); box(.0,.06,.2,.24,'Leaderboard\n174 systems\n877 passing patches')
box(.29,.70,.2,.24,'Issue axis (I)\n20 instances','#e6f0fa'); box(.29,.38,.2,.24,'Test axis (T)\n52 instances','#e6f0fa'); box(.29,.06,.2,.24,'Patch axis (P)\n81 instances','#e6f0fa')
box(.58,.55,.18,.3,'Defect cards\n131 flagged\n369 clean','#fdf1e3'); box(.58,.12,.18,.3,'Item statistics\npass rate, 2PL,\nregression','#fdf1e3')
box(.84,.70,.16,.24,'RQ1 overlap\nRQ2 footprint'); box(.84,.38,.16,.24,'RQ3 item\nbehaviour'); box(.84,.06,.16,.24,'RQ4 null-calibrated\nranking\nRQ5 cost, triage')
for y in (.82,.5,.18):
    ax.annotate('',xy=(.29,y),xytext=(.2,y),arrowprops=dict(arrowstyle='->',color='#555',lw=.8)); ax.annotate('',xy=(.58,.7 if y>.4 else .27),xytext=(.49,y),arrowprops=dict(arrowstyle='->',color='#555',lw=.8))
for y,yy in ((.82,.7),(.5,.4),(.18,.27)): ax.annotate('',xy=(.84,y),xytext=(.76,yy),arrowprops=dict(arrowstyle='->',color='#555',lw=.8))
ax.set_xlim(-.02,1.02); ax.set_ylim(0,1); fig.savefig('out/F1_overview.png'); plt.close()
# Fig2 regions
reg=[('Patch only',63),('Test only',37),('Issue only',9),('Test + patch',11),('Issue + patch',7),('Issue + test',4),('All three',0)]
fig,ax=plt.subplots(figsize=(6.2,2.9)); ax.barh([r[0] for r in reg][::-1],[r[1] for r in reg][::-1],color=([B]*4+[G]*3)); 
for i,(k,v) in enumerate(reg[::-1]): ax.text(v+.8,i,str(v),va='center')
ax.set_xlabel('Instances (of 500)'); ax.set_title('109 flagged instances carry one axis; 22 carry two; none carries three',loc='left',fontsize=9.5); fig.savefig('out/F2_regions.png'); plt.close()
# Fig3 footprint: relation between agent and gold file sets, per tool
A=pd.read_csv('out/agent_footprint.csv'); fig,ax=plt.subplots(figsize=(6.4,2.6))
names={'20241029_OpenHands-CodeAct-2.1-sonnet-20241022':'OpenHands','20241221_codestory_midwit_claude-3-5-sonnet_swe-search':'CodeStory','20250110_learn_by_interact_claude3.5':'LearnByInteract'}
left=np.zeros(3); tl=[names[t] for t in sorted(names)]
for col,c,lab in [('same',G,'same files'),('superset','#c9ced6','superset'),('subset',B,'strict subset'),('disjoint',O,'disjoint')]:
    v=[100*A[A.tool==t][col].mean() for t in sorted(names)]; ax.barh(tl,v,left=left,color=c,label=lab); left+=v
ax.set_xlabel('% of passing patches'); ax.legend(frameon=False,fontsize=7.5,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.25)); ax.set_xlim(0,100)
fig.savefig('out/F3_footprint.png'); plt.close()
# Fig4 pass rate + discrimination by group
gr=[('Clean',F[F.clean==1]),('Issue',F[F.I==1]),('Test, lax',F[F.T_lax==1]),('Scope creep',F[F.P_scope==1]),('Defective gold',F[F.P_defect==1]),('Unneeded change',F[F.P_unneeded==1])]
fig,axs=plt.subplots(1,2,figsize=(8.4,3)); 
for ax,col,yl in ((axs[0],'pass_rate','Pass rate over 174 systems'),(axs[1],'disc','2PL discrimination (a)')):
    bp=ax.boxplot([g[1][col].values for g in gr],tick_labels=[f"{g[0]}\n(n={len(g[1])})" for g in gr],patch_artist=True,widths=.6,medianprops=dict(color=K),flierprops=dict(markersize=2))
    for i,p in enumerate(bp['boxes']): p.set_facecolor(G if i==0 else '#cfe0f3'); p.set_edgecolor('#555')
    ax.set_ylabel(yl); ax.tick_params(axis='x',labelsize=6.5)
fig.savefig('out/F4_items.png'); plt.close()
# Fig5 forest
co=a2['reg']['coef']; un=a2['reg']['coef_unadj']; ks=[('I','Issue'),('P_scope','Scope creep'),('P_defect','Defective gold'),('P_unneeded','Unneeded change'),('T_lax','Test, lax')]
fig,ax=plt.subplots(figsize=(6.2,2.8))
for i,(k,l) in enumerate(ks[::-1]):
    ax.plot([co[k]['lo'],co[k]['hi']],[i,i],color=B,lw=1.5); ax.plot(co[k]['beta'],i,'o',color=B,label='adjusted (95% CI)' if i==0 else None); ax.plot(un[k]['beta'],i+.22,'s',color=G,ms=5,label='unadjusted' if i==0 else None)
ax.axvline(0,color=K,lw=.6); ax.set_yticks(range(5)); ax.set_yticklabels([l for _,l in ks[::-1]]); ax.set_xlabel('Effect on logit pass rate'); ax.legend(frameon=False,fontsize=8); fig.savefig('out/F5_forest.png'); plt.close()
# Fig6 nulls + power
rng=np.random.default_rng(7); X=R.values.astype(float); rk=lambda s: stats.rankdata(-s,method='min')
pc=pd.read_csv('data/paichecker/labelled_data/swe-bench-analysis.csv',nrows=1).columns[13:-2]; X131=R[list(pc)].values.astype(float); mis=F.S_any.values==1
f=rk(X131.mean(0)); o1=(f!=rk(X131[~mis].mean(0))).mean(); n1=[(f!=rk(X131[rng.choice(500,432,replace=False)].mean(0))).mean() for _ in range(4000)]
d=F.defect.values==1; fs=X.mean(0); o2=stats.kendalltau(fs,X[~d].mean(0))[0]; n2=[stats.kendalltau(fs,X[rng.choice(500,369,replace=False)].mean(0))[0] for _ in range(3000)]
fig,axs=plt.subplots(1,3,figsize=(8.6,2.7))
axs[0].hist(100*np.array(n1),bins=28,color='#cfd4da'); axs[0].axvline(100*o1,color=B,lw=2); axs[0].set_xlabel('% of 131 agents changing rank\n(68 instances removed)'); axs[0].set_title('(a) Replication of [5]',fontsize=9,loc='left'); axs[0].set_ylabel('random removals')
axs[1].hist(n2,bins=28,color='#cfd4da'); axs[1].axvline(o2,color=B,lw=2); axs[1].set_xlabel("Kendall's tau, 174 systems\n(131 instances removed)"); axs[1].set_title('(b) All flagged instances',fontsize=9,loc='left')
pw=a5['power']; xs=[100*float(k) for k in pw]; axs[2].plot(xs,[100*pw[k] for k in pw],'o-',color=B); axs[2].axhline(5,color=G,lw=.8,ls='--'); axs[2].set_xlabel('Planted differential bias\n(percentage points)'); axs[2].set_ylabel('% of simulations detected'); axs[2].set_title('(c) Power of the null test',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig('out/F6_nulls.png'); plt.close()
# Fig7 ratio
fig,ax=plt.subplots(figsize=(6.2,3.1)); ax.scatter(100*S.score,S.Plab/S.clean,s=11,color=B,label='Patch-labelled (scope creep or defective gold)'); ax.scatter(100*S.score,S.I/S.clean,s=11,color=G,label='Issue-flagged')
ax.set_xlabel('Resolved on all 500 instances (%)'); ax.set_ylabel('Pass rate relative to clean instances'); ax.legend(frameon=False,fontsize=8); ax.set_ylim(0,1.3); fig.savefig('out/F7_ratio.png'); plt.close()
# Fig8 frontier + cost
fr=a3['frontier']['10']; c=a3['cost']; fig,axs=plt.subplots(1,2,figsize=(8.4,2.8))
v=[100*fr['defect_share_allpass'],100*fr['defect_share_inf'],100*fr['defect_share_allfail']]; axs[0].bar(['resolved by all\n(290)','separating\n(162)','resolved by none\n(48)'],v,color=[G,B,O]); [axs[0].text(i,x+.6,f'{x:.1f}%',ha='center') for i,x in enumerate(v)]; axs[0].axhline(26.2,color=K,lw=.7,ls='--'); axs[0].text(2.45,26.8,'base rate',fontsize=7,ha='right'); axs[0].set_ylabel('% flagged'); axs[0].set_title('(a) Top-10 systems',loc='left',fontsize=9)
v=[26.2,100*c['share_defect'],100*c['unresolved_cost_share_defect']]; axs[1].bar(['share of\ninstances','share of\nspend','share of spend on\nunresolved attempts'],v,color=[G,B,O]); [axs[1].text(i,x+.5,f'{x:.1f}%',ha='center') for i,x in enumerate(v)]; axs[1].set_ylabel('% attributable to flagged instances'); axs[1].set_title('(b) 39 bash-only runs, USD 7,131',loc='left',fontsize=9)
fig.tight_layout(); fig.savefig('out/F8_frontier_cost.png'); plt.close()
# Fig9 triage
t=a4['triage']; tg=list(t); sets=['static','response','human effort label','static+response']; fig,ax=plt.subplots(figsize=(7,2.9)); w=.2
for j,s in enumerate(sets): ax.bar(np.arange(len(tg))+j*w-.3,[t[k][s]['auc'] for k in tg],w,label=s,color=[B,'#8fb3dd',G,'#444'][j])
ax.axhline(.5,color=K,lw=.7,ls='--'); ax.set_xticks(range(len(tg))); ax.set_xticklabels([k.replace(' (','\n(').replace('UTBoost insufficient tests','UTBoost\ninsufficient tests').replace('PAIChecker misaligned','PAIChecker\nmisaligned') for k in tg],fontsize=7); ax.set_ylabel('Cross-validated AUC'); ax.set_ylim(.35,.85); ax.legend(frameon=False,fontsize=7.5,ncol=4,loc='upper center'); fig.savefig('out/F9_triage.png'); plt.close()
print('ok')
