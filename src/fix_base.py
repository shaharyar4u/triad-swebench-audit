# Replaces task fields from the PAIChecker CSV (spreadsheet cells capped at 32,767 chars, which
# truncates 26 PASS_TO_PASS lists) with the full fields from the original SWE-bench Verified dataset.
import pyarrow as pa, pandas as pd
P='data/patchdiff/data/dataset/swebench_verified/data-00000-of-00001.arrow'
with pa.memory_map(P) as s:
    try: t=pa.ipc.open_stream(s).read_all()
    except Exception: t=pa.ipc.open_file(s).read_all()
d=t.to_pandas(); b=pd.read_pickle('out/base.pkl')
keep=d.set_index('instance_id').loc[b.instance_id].reset_index()
keep['label']=b.label.values
if 'difficulty' not in keep: keep['difficulty']=b.difficulty.values
keep.to_pickle('out/base.pkl')
print('base table rebuilt from original dataset:',keep.shape,'| truncated PASS_TO_PASS cells fixed:',int((b.PASS_TO_PASS.str.len()>=32760).sum()))
