import json
from pathlib import Path
from run_revalidation import plan,run_task,ROOT,DATA,load_jsonl,taskid
p=plan();t=next(t for t in p if t['experiment']=='E10_matched' and t['dataset']=='colon' and t['alpha']==.5 and t['run']==0);result=run_task(t);assert result['status']!='error',result
legacy=next(r for r in load_jsonl(DATA.parent/'results/bench_multimetric_robustness_clean.jsonl') if r['dataset']=='colon' and r['mode']=='GAM-GA' and r['run']==0)
checks=[]
for fi in range(5):
 r=json.loads((ROOT/'results'/f'{taskid(t)}_fold{fi}.json').read_text());old=legacy['fold_records'][fi];assert all(abs(r['metrics'][k]-old[k])<1e-12 for k in r['metrics']);assert sorted(r['selected_features'])==sorted(old['selected_features']);checks.append({'fold':fi,'same_selected_features':True,'same_metrics':True})
(ROOT/'golden_validation.json').write_text(json.dumps({'case':'matched E10 alpha=.5 colon run0 vs historical E9 GAM-GA, N=T=50','checks':checks},indent=2));print('Golden full-budget five-fold validation PASS',flush=True)
