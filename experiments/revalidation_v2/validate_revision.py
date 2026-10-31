import sys,json,ast
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));from run_revalidation import bg,Evaluator,BARE,GAM,taskid
from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
rng=np.random.default_rng(9);X=rng.normal(size=(35,240));y=np.arange(35)%3;scores=rng.random(240)
# Cache must preserve literal reference score and consume no global RNG.
ev=Evaluator(X,y);idx=np.arange(12);np.random.seed(123);before=np.random.get_state();v=ev(X,y,idx);w=ev(X,y,idx);after=np.random.get_state();ref=float(np.mean(cross_val_score(KNeighborsClassifier(5,metric='manhattan',n_jobs=1),X[:,idx],y,cv=KFold(5,shuffle=True,random_state=42))))
assert v==w==ref and len(ev.cache)==1 and ev.calls==2
assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
# Explicit SSA must invoke the lambda-controlled refinement and preserve every evaluated best.
ssa=[]
for lam in [0.,.6]:
 g=bg.GAM(X,scores);orig=g.safe_crossover;calls=[]
 def cross(p1,p2,k,gen=0,max_gen=100,**kwargs):calls.append((gen,max_gen));return orig(p1,p2,k,gen=gen,max_gen=max_gen,penalty=lam)
 g.safe_crossover=cross;values=[]
 def score(X,y,idx):val=float(scores[idx].sum());values.append(val);return val
 old=bg.evaluate;bg.evaluate=score;np.random.seed(1)
 try:idx=bg.ssa_gam_safe(X,y,12,g,6,5)
 finally:bg.evaluate=old
 assert len(idx)==12 and len(calls)==30 and set(t for t,T in calls)==set(range(5))
 assert abs(float(scores[idx].sum())-max(values))<1e-12
 ssa.append({'lambda':lam,'crossover_calls':len(calls),'best':max(values),'selected':idx.tolist()})
# All copied experiment crossover wrappers must forward the exact gen/max_gen arguments.
wrapper_tests=[]
for p in (ROOT/'source/experiments').glob('*.py'):
 tree=ast.parse(p.read_text())
 for n in ast.walk(tree):
  if isinstance(n,ast.Lambda) and [a.arg for a in n.args.args]==['p1','p2','k','gen','max_gen']:
   received=[]
   def target(*a,**kw):received.append(kw)
   ns={'o':target,'o_gam':target,'lam':.6,'penalty':.6,'LAMBDA':.6};f=eval(compile(ast.Expression(n),str(p),'eval'),ns);f(None,None,12,gen=49,max_gen=50)
   assert received[0]['gen']==49 and received[0]['max_gen']==50
   wrapper_tests.append(p.name)
report={'evaluator_matches_sklearn':True,'cache_does_not_consume_rng':True,'ssa_cases':ssa,'wrapper_forwarding_tests':wrapper_tests};(ROOT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
