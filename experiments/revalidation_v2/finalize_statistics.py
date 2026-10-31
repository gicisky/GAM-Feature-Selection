from pathlib import Path
import json,math
import pandas as pd,numpy as np
from scipy.stats import friedmanchisquare
ROOT=Path(__file__).resolve().parent
rows=[]
for p in (ROOT/'results').glob('*.json'):
 r=json.loads(p.read_text())
 if 'accuracy_mean' not in r or r['task']['experiment']!='main':continue
 ff=[json.loads((ROOT/'results'/f).read_text()) for f in r['fold_files']];rows.append({'dataset':r['task']['dataset'],'base':r['task']['base'],'mode':r['task']['mode'],'run':r['task']['run'],'accuracy':r['accuracy_mean'],'red_finite_pairs':r['redundancy_mean'],'red_pair_zero':float(np.mean([x['redundancy']['red_pair_zero'] for x in ff])),'origin':'rerun_v2'})
for line in (ROOT/'retained_unchanged_bare.jsonl').read_text().splitlines():
 r=json.loads(line);rows.append({'dataset':r['dataset'],'base':r['base'],'mode':'Bare','run':r['run'],'accuracy':r['acc_mean'],'red_finite_pairs':r['red_mean'],'red_pair_zero':r['red_mean'],'origin':'retained_legacy'})
df=pd.DataFrame(rows);expected=15*5*3*5
if len(df)!=expected:raise SystemExit(f'Incomplete primary benchmark: {len(df)}/{expected}; no final inference generated')
assert not df.duplicated(['dataset','base','mode','run']).any()
assert set(df.groupby(['dataset','base','mode']).size())=={5}
dataset=df.groupby(['dataset','mode'])[['accuracy','red_finite_pairs','red_pair_zero']].mean();dataset.to_csv(ROOT/'primary_dataset_means.csv');df.to_csv(ROOT/'primary_complete_provenance.csv',index=False);macro=dataset.groupby('mode').mean();macro.to_csv(ROOT/'primary_macro.csv');a=dataset.accuracy.unstack();ranks=a.rank(axis=1,ascending=False);stat=friedmanchisquare(a.Bare,a['GAM-lambda0'],a.GAM)
def exact_wilcoxon(x,y):
 d=np.asarray(x)-np.asarray(y);d=d[d!=0]
 if not len(d):return 1.
 rank=pd.Series(np.abs(d)).rank(method='average').to_numpy();r2=np.rint(rank*2).astype(int);w=int(r2[d>0].sum());total=int(r2.sum());dp={0:1}
 for v in r2:
  for k,n in list(dp.items()):dp[k+int(v)]=dp.get(k+int(v),0)+n
 return min(1.,2*sum(n for k,n in dp.items() if k<=min(w,total-w))/2**len(d))
checks={}
for x,y in [('Bare','GAM'),('Bare','GAM-lambda0'),('GAM-lambda0','GAM')]:checks[x+' vs '+y]=exact_wilcoxon(a[x],a[y])
result={'version':'GAM-revalidation-v2','primary_records':len(df),'new_records':int((df.origin=='rerun_v2').sum()),'retained_records':int((df.origin=='retained_legacy').sum()),'macro':macro.reset_index().to_dict('records'),'friedman':{'statistic':float(stat.statistic),'pvalue':float(stat.pvalue)},'mean_ranks':ranks.mean().to_dict(),'paired_wilcoxon_unadjusted_exact':checks,'nemenyi_CD_approx':2.343*math.sqrt(3*4/(6*15)),'note':'Red finite-pair and pair-zero reported separately. No old full-data diagnostic is combined with these new fold-safe results.'};(ROOT/'primary_statistics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
