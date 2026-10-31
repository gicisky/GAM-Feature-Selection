import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import time,json
from pathlib import Path
import numpy as np
import run_revalidation as core
from fast_evaluator import FastEvaluator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
rng=np.random.default_rng(20261031);rows=[]
files=['colon(62,2000).csv','9_Tumors(60,5726).csv','DLBCL(77,5469).csv','arcene(200,10000).csv','Lung(202,12601).csv','Eleven_Tumor(174,12534).csv']
for file in files:
 X,y,_=core.getdata(file);tr,te=next(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y));X=MinMaxScaler().fit_transform(X[tr]);y=y[tr];k=int(X.shape[1]*.05)
 for inner in [3,5]:
  ev=FastEvaluator(X,y,inner,42);rt=ft=0.;count=30
  for i in range(count):
   idx=np.sort(rng.choice(X.shape[1],k,replace=False));start=time.perf_counter();a=ev.reference(idx);rt+=time.perf_counter()-start;start=time.perf_counter();b=ev.fast_reference(idx);ft+=time.perf_counter()-start;assert a==b,(file,inner,i,a,b)
  rows.append({'file':file,'inner_folds':inner,'cases':count,'reference_s':rt,'fast_s':ft,'speedup':rt/ft,'fallback_folds':ev.boundary_fallback_folds});print(rows[-1],flush=True)
# Adversarial duplicate samples/constant dimensions force ambiguous-boundary fallback.
for X in [np.zeros((30,20)),np.repeat(np.arange(10)[:,None],3,axis=0)*np.ones((1,20))]:
 y=np.arange(30)%3;ev=FastEvaluator(X,y,5,42);idx=np.arange(20);assert ev.reference(idx)==ev.fast_reference(idx);assert ev.boundary_fallback_folds>0
# Complete toy GA trajectories must agree, including evaluated best and RNG state.
X=rng.normal(size=(45,240));y=np.arange(45)%3;scores=rng.random(240);solutions=[];states=[]
for cls in [core.Evaluator,FastEvaluator]:
 ev=cls(X,y,5,42);g=core.bg.GAM(X,scores);old=core.bg.evaluate;core.bg.evaluate=ev;np.random.seed(42)
 try:sol=core.bg.ga_gam_safe(X,y,12,g,10,8)
 finally:core.bg.evaluate=old
 solutions.append(sol);states.append(np.random.get_state())
assert np.array_equal(*solutions) and np.array_equal(states[0][1],states[1][1]) and states[0][2:]==states[1][2:]
result={'all_exact':True,'random_subset_cases':sum(r['cases'] for r in rows),'adversarial_cases':2,'GA_trajectory_and_RNG_equal':True,'benchmarks':rows};(core.ROOT/'fast_validation.json').write_text(json.dumps(result,indent=2));print('PASS',flush=True)
