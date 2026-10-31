import json,time
from pathlib import Path
from collections import Counter
import pandas as pd
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'plan.json').read_text());rows=[];folds=[];errors=[]
for p in (ROOT/'results').glob('*.json'):
 d=json.loads(p.read_text())
 if p.name.endswith('.error.json'):errors.append(d)
 elif 'accuracy_mean' in d:rows.append({**d['task'],'task_id':d['task_id'],'accuracy_mean':d['accuracy_mean'],'redundancy_mean':d['redundancy_mean'],'folds':d['folds'],'result_origin':'rerun_v2'})
 elif 'fold' in d:folds.append(d)
status={'generated_unix':time.time(),'planned_tasks':len(plan),'completed_tasks':len(rows),'completed_folds':len(folds),'error_tasks':len(errors),'planned_by_experiment':dict(Counter(t['experiment'] for t in plan)),'completed_by_experiment':dict(Counter(t['experiment'] for t in rows)),'warning':'partial results; no complete primary benchmark inference until all required groups are complete'}
(ROOT/'progress.json').write_text(json.dumps(status,indent=2));pd.DataFrame(rows).to_csv(ROOT/'completed_records.csv',index=False)
legacy={}
for file in ['bench_foldsafe_bare.jsonl','bench_foldsafe_full.jsonl']:
 for line in (ROOT.parent.parent/'results'/file).read_text().splitlines():
  r=json.loads(line)
  if r['protocol']=='fold-safe':legacy[(r['dataset'],r['base'],r['mode'],r['run'])]=r
comparison=[]
for r in rows:
 if r['experiment']=='main':
  old=legacy[(r['dataset'],r['base'],r['mode'],r['run'])];comparison.append({**r,'old_accuracy':old['acc_mean'],'accuracy_difference':r['accuracy_mean']-old['acc_mean'],'old_red':old['red_mean'],'red_difference':r['redundancy_mean']-old['red_mean'] if r['redundancy_mean'] is not None else None,'note':'new vs legacy; Red definition changed where undefined pairs occur'})
pd.DataFrame(comparison).to_csv(ROOT/'main_completed_vs_legacy.csv',index=False)
print(json.dumps(status,indent=2))
