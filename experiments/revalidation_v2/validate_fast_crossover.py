import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,time
import numpy as np
import run_revalidation as c
from fast_crossover import fast_crossover
rng=np.random.default_rng(20261031);X=rng.normal(size=(45,1500));scores=rng.random(1500);g=c.bg.GAM(X,scores);old=c.bg.GAM.safe_crossover;oldtime=newtime=0;cases=0
for k in [1,100,286,500,630]:
 for penalty in [0.,.6,.8]:
  for repeat in range(12):
   p1=np.zeros(1500,dtype=bool);p2=p1.copy();p1[rng.choice(1500,k,False)]=True;p2[rng.choice(1500,k,False)]=True;seed=1000+cases
   np.random.seed(seed);start=time.perf_counter();a=old(g,p1,p2,k,gen=99,max_gen=100,penalty=penalty);oldtime+=time.perf_counter()-start;ra=np.random.get_state()
   np.random.seed(seed);start=time.perf_counter();b=fast_crossover(g,p1,p2,k,gen=99,max_gen=100,penalty=penalty);newtime+=time.perf_counter()-start;rb=np.random.get_state();assert np.array_equal(a,b),(k,penalty,repeat);assert np.array_equal(ra[1],rb[1]) and ra[2:]==rb[2:];cases+=1
# Exact ties / nearly tied relevance and correlation must choose original index.
for scale in [0.,1e-15,1e-10]:
 g.scores=np.ones(1500)+scale*rng.random(1500);g.C[:]=.5
 for seed in range(10):
  p1=np.arange(1500)%3==0;p2=np.arange(1500)%3==1
  np.random.seed(seed);a=old(g,p1,p2,500,gen=99,max_gen=100,penalty=.6)
  np.random.seed(seed);b=fast_crossover(g,p1,p2,500,gen=99,max_gen=100,penalty=.6);assert np.array_equal(a,b);cases+=1
result={'all_exact':True,'cases':cases,'random_state_equal':True,'ordinary_cases_old_s':oldtime,'ordinary_cases_fast_s':newtime,'speedup':oldtime/newtime};(c.ROOT/'fast_crossover_validation.json').write_text(json.dumps(result,indent=2));print(result)
