from pathlib import Path
import json,time,math
from collections import defaultdict
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parent
samples=[]
for p in (ROOT/'results').glob('*_fold*.json'):
 r=json.loads(p.read_text());t=r['task']
 if t['base']=='GA' and t['mode']!='Bare' and 'execution_engine' not in r:samples.append(r)
def features(n,k):return [1.,(.8*n)**2*k/1e6,k*k/1e5]
def guide_features(n,p):return [p/1000.,p*p*(.8*n)/1e10]
A=np.array([features(r['n_samples'],r['k']) for r in samples]);b=np.array([r['search_s']/(r['task']['N']*(r['task']['T']+1)) for r in samples]);coef=nnls(A,b)[0]
G=np.array([guide_features(r['n_samples'],r['D_search']) for r in samples]);gcoef=nnls(G,np.array([r['guidance_s'] for r in samples]))[0]
data=pd.read_csv(ROOT.parent/'code_data_verification/data_loading.csv');shapes={r.file:(int(r.default_n),int(r.features)) for r in data.itertuples()};plan=json.loads((ROOT/'plan.json').read_text());comp=json.loads((ROOT/'component_ablation/plan.json').read_text());summary=defaultdict(float);pending=defaultdict(int);details=[]
def fold_time(t,component=False):
 n,D=shapes[t['file']];p=min(5000,D) if t.get('prefilter',False) else D;k=max(1,int(.05*D))
 if t['experiment'] in ['E7','convergence','independent','filter_guided']:k=min(k,200)
 request=t['N']*(t['T']+1);base=t.get('base','GA');mode=t.get('mode','GAM');factor={'GA':1.,'CSA':.35,'SSA':1.,'DE':1.2,'PSO':1.2,'AO':1.2,'GWO':1.4,'BGWO':1.1,'BPSO':1.1}[base]
 if mode=='Bare':factor={'GA':.8,'CSA':.85,'SSA':.9,'DE':1.,'PSO':1.,'BGWO':1.,'BPSO':1.}[base]
 search=max(.0003,float(np.dot(features(n,k),coef)))*request*factor;guide=max(0.,float(np.dot(guide_features(n,p),gcoef))) if mode!='Bare' else 0.
 if component:
  group=t['group'];search*= {'A':1.,'B':.8,'C':1.,'D':1.}[group]
  if group=='A':guide=0.
 if t.get('guidance')=='mRMR-guided-GAM':
  # Prefix timing extrapolation; candidates shrink as ranking proceeds.
  benchmarks=json.loads((ROOT/'eta_guidance_bench.json').read_text());z=min(benchmarks,key=lambda z:abs(z['p']-p));limit=min(2000,p);guide+=z['seconds']*(limit/100)*(.8*n/z['n_train'])*(p/z['p'])*(1-limit/(2*p))
 if (ROOT/'fast_execution_manifest.json').exists():
  validation=json.loads((ROOT/'fast_validation.json').read_text());bench=validation['benchmarks'];inner=3 if t['experiment'] in ['E10_matched','convergence','independent','filter_guided'] else 5
  matches=[r for r in bench if r['file']==t['file'] and r['inner_folds']==inner]
  if not matches:matches=sorted([r for r in bench if r['inner_folds']==inner],key=lambda r:abs(shapes[r['file']][0]-n))[:1]
  kernel_speed=max(1.,matches[0]['speedup'])
  if mode=='Bare' or (component and t['group'] in ['A','B']):search/=kernel_speed
  elif t.get('lambda',.6)==0 or (component and t['group']=='C'):search*=max(.12,.5/kernel_speed+.03)
  else:search*=.348
 return guide+search+.2
for t in plan:
 ident=__import__('hashlib').sha256(json.dumps(t,sort_keys=True).encode()).hexdigest()[:20]
 if (ROOT/'results'/f'{ident}.json').exists():continue
 folds=[0] if t['experiment'] in ['E7','lambda','convergence','independent','filter_guided'] else range(5)
 for fi in folds:
  if (ROOT/'results'/f'{ident}_fold{fi}.json').exists():continue
  seconds=fold_time(t);summary[t['experiment']]+=seconds;pending[t['experiment']]+=1
for t0 in comp:
 ident=__import__('hashlib').sha256(json.dumps(t0,sort_keys=True).encode()).hexdigest()[:20]
 if (ROOT/'component_ablation/results'/f'{ident}.json').exists():continue
 t={**t0,'experiment':'component','base':'GA','mode':'GAM','prefilter':False}
 for fi in range(5):
  if (ROOT/'component_ablation/results'/f'{ident}_fold{fi}.json').exists():continue
  summary['component']+=fold_time(t,True);pending['component']+=1
# Two fixed workers; 10% scheduling/load imbalance allowance. Range is a forecast,
# not a statistical confidence interval. Unknown optimizer/cache behavior dominates.
now=datetime.now(ZoneInfo('Asia/Shanghai'));rows=[]
for exp,s in summary.items():rows.append({'experiment':exp,'pending_folds':pending[exp],'worker_hours':s/3600,'central_wall_hours':s/2/3600*1.10})
center=sum(r['central_wall_hours'] for r in rows);low=center*.6;high=center*1.8
result={'generated_at':now.isoformat(),'sample_folds':len(samples),'calibration_datasets':sorted(set(r['task']['dataset'] for r in samples)),'search_coefficients':coef.tolist(),'guidance_coefficients':gcoef.tolist(),'acceleration_enabled':(ROOT/'fast_execution_manifest.json').exists(),'workers':2,'remaining_hours_central':center,'remaining_hours_planning_range':[low,high],'central_finish':(now+timedelta(hours=center)).isoformat(),'finish_planning_range':[(now+timedelta(hours=low)).isoformat(),(now+timedelta(hours=high)).isoformat()],'by_experiment':rows,'limitations':['Range is a planning estimate, not confidence interval','No sleep, interruptions, thermal throttling or failures assumed','Fast mode extrapolates from exact full-budget arcene fold plus scoring/crossover benchmarks; remaining optimizer factors are assumptions','Observed calibration temporarily overlapped two normal workers, potentially conservative','Partial folds and completed tasks excluded from remaining work','Component ablation now runs first; then resume saved original batch; same two-worker ceiling']}
(ROOT/'time_estimate.json').write_text(json.dumps(result,indent=2));pd.DataFrame(rows).to_csv(ROOT/'time_estimate_by_experiment.csv',index=False);print(json.dumps(result,indent=2))
