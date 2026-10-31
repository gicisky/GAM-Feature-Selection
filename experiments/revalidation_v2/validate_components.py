import numpy as np,json
from component_ablation import ComponentGAM,ROOT,core
rng=np.random.default_rng(19);X=rng.normal(size=(30,240));scores=rng.random(240);y=np.arange(30)%2;out=[]
for group in ['A','B','C','D']:
 g=ComponentGAM(X,scores,group);vals=[]
 def fit(X,y,idx):v=float(scores[idx].sum());vals.append(v);return v
 old=core.bg.evaluate;core.bg.evaluate=fit;np.random.seed(42)
 try:idx=core.bg.ga_gam_safe(X,y,12,g,6,5)
 finally:core.bg.evaluate=old
 assert len(vals)==36 and len(idx)==12 and abs(scores[idx].sum()-max(vals))<1e-12
 out.append({'group':group,'fitness_calls':len(vals),'cardinality':len(idx),'historical_best_preserved':True})
 # D must be literally equivalent to current full GAM given same inputs and RNG.
 if group=='D':
  g2=core.bg.GAM(X,scores);vals=[];core.bg.evaluate=fit;np.random.seed(42)
  try:idx2=core.bg.ga_gam_safe(X,y,12,g2,6,5)
  finally:core.bg.evaluate=old
  assert np.array_equal(idx,idx2)
(core.ROOT/'component_ablation/validation.json').write_text(json.dumps({'groups':out,'D_matches_full_GAM':True},indent=2));print('4 groups matched GA/count/cardinality/best checks PASS; D equals full GAM')
