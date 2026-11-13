import pandas as pd, numpy as np, json, re, ast
b=pd.read_pickle('out/base.pkl')
def parse_diff(p):
    files=[];cur=None
    for ln in p.split('\n'):
        if ln.startswith('diff --git'):
            m=re.match(r'diff --git a/(.*?) b/(.*)',ln); cur={'path':m.group(2),'hunks':[]}; files.append(cur)
        elif ln.startswith('@@') and cur is not None:
            m=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)',ln)
            cur['hunks'].append({'start':int(m.group(1)),'ctx':m.group(5).strip(),'add':[],'del':[]})
        elif cur and cur['hunks'] and not ln.startswith(('+++','---')):
            if ln.startswith('+'): cur['hunks'][-1]['add'].append(ln[1:])
            elif ln.startswith('-'): cur['hunks'][-1]['del'].append(ln[1:])
    return files
DOC=re.compile(r'(^|/)(docs?|doc|changelog|releasenotes|release_notes|whatsnew|news)(/|$)|\.(rst|md|txt)$|CHANGES|AUTHORS',re.I)
def noncode(l):
    s=l.strip()
    return s=='' or s.startswith('#') or s.startswith(('"""',"'''")) 
def unit(ctx):
    m=re.search(r'(class|def)\s+(\w+)',ctx); return m.group(2) if m else ctx[:40]
def gold_feats(p):
    fs=parse_diff(p); src=[f for f in fs if not DOC.search(f['path'])]; doc=[f for f in fs if DOC.search(f['path'])]
    hunks=[h for f in src for h in f['hunks']]
    add=sum(len(h['add']) for h in hunks); dele=sum(len(h['del']) for h in hunks)
    # non-functional hunks: every changed line is blank/comment/docstring marker, or pure whitespace change
    def nonfunc(h):
        ch=h['add']+h['del']
        if all(noncode(x) for x in ch): return True
        return sorted(x.strip() for x in h['add'])==sorted(x.strip() for x in h['del'])
    nf=sum(nonfunc(h) for h in hunks)
    units=set((f['path'],unit(h['ctx'])) for f in src for h in f['hunks'])
    dirs=set('/'.join(f['path'].split('/')[:-1]) for f in src)
    return dict(g_files=len(src),g_docfiles=len(doc),g_hunks=len(hunks),g_add=add,g_del=dele,g_churn=add+dele,
                g_nonfunc_hunks=nf,g_units=len(units),g_dirs=len(dirs),g_srcpaths=[f['path'] for f in src])
ASSERT=re.compile(r'\bassert\w*\b|\.raises\(|assertRaises|pytest\.warns|\braises\(')
LIT=re.compile(r'assertRaisesMessage|assertRaisesRegex|match\s*=|assertWarnsMessage|msg\s*=|==\s*[frbu]*["\']{1,3}.{12,}|assertEqual\(\s*(str|repr)\(|in str\(|assertIn\(\s*["\'].{12,}|assertHTMLEqual|assertMessages|\.args\[0\]|capsys|caplog|assertOutput|== *\(?\s*$')
def test_feats(tp,f2p,p2p):
    fs=parse_diff(tp); add=[l for f in fs for h in f['hunks'] for l in h['add']]
    asserts=[l for l in add if ASSERT.search(l)]
    lit=[l for l in asserts if LIT.search(l)]
    f2=json.loads(f2p); p2=json.loads(p2p)
    newtests=sum(1 for l in add if re.match(r'\s*(async\s+)?def test',l))
    return dict(t_files=len(fs),t_add=len(add),t_asserts=len(asserts),t_lit_asserts=len(lit),t_f2p=len(f2),t_p2p=len(p2),t_newtests=newtests,t_paths=[f['path'] for f in fs])
G=pd.DataFrame([gold_feats(p) for p in b.patch]); T=pd.DataFrame([test_feats(a,c,d) for a,c,d in zip(b.test_patch,b.FAIL_TO_PASS,b.PASS_TO_PASS)])
F=pd.concat([b[['instance_id','repo','difficulty','label','created_at']],G,T],axis=1)
F['ps_words']=b.problem_statement.str.split().str.len(); F['ps_has_code']=b.problem_statement.str.contains('```|>>>|Traceback').astype(int)
F['hints_words']=b.hints_text.fillna('').str.split().str.len()
# S axis labels
lab=F.label.fillna('No Misalignment')
for k in ['SC','DP','IS','FP','UL']: F['S_'+k]=lab.str.contains(k).astype(int)
F['S_any']=(lab!='No Misalignment').astype(int)
# UTBoost
a=json.load(open('data/utboost/assets/useful_scripts/augTest.json')); F['T_utboost']=F.instance_id.isin(a).astype(int)
pz=json.load(open('data/utboost/update_SWE_Bench/updated_parser_test_instance_dict_verified.json')); F['T_parserfix']=F.instance_id.isin(pz).astype(int)
F.to_pickle('out/feat.pkl')
pd.set_option('display.width',250)
print(F.S_any.sum(),F.T_utboost.sum(),F.T_parserfix.sum(), len(set(a)-set(F.instance_id)))
print(F[['g_files','g_docfiles','g_hunks','g_churn','g_nonfunc_hunks','g_units','g_dirs','t_files','t_add','t_asserts','t_lit_asserts','t_f2p','t_p2p','t_newtests','ps_words','hints_words']].describe(percentiles=[.5,.9]).round(1).T.to_string())
print((F.g_files>1).mean(),(F.g_docfiles>0).mean(),(F.g_nonfunc_hunks>0).mean(),(F.g_units>1).mean(),(F.t_f2p==1).mean(),(F.t_asserts==0).mean())
