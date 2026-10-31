"""Equivalent GAM crossover with rolling correlation sums and exact tie fallback."""
import numpy as np
def fast_crossover(self,p1,p2,k,gen=0,max_gen=100,penalty=.6):
 ratio_original=.8-.5*gen/max_gen
 if np.random.rand()<ratio_original:
  child=np.zeros(self.p,dtype=bool);mask=np.random.rand(self.p)<.5;child[mask]=p1[mask];child[~mask]=p2[~mask];cnt=int(child.sum())
  if cnt>k:
   pos=np.flatnonzero(child);np.random.shuffle(pos);child[pos[k:]]=False;cnt=k
  while cnt<k:child[np.random.choice(np.flatnonzero(~child))]=True;cnt+=1
  return child
 u=np.unique(np.concatenate([np.flatnonzero(p1),np.flatnonzero(p2)]));child=np.zeros(self.p,dtype=bool)
 if len(u)<=k:
  child[u]=True
  while child.sum()<k:child[np.random.choice(np.flatnonzero(~child))]=True
  return child
 rel=self.scores[u]
 if penalty==0:
  # Original greedy order is descending relevance, stable by original index.
  chosen=np.argsort(-rel,kind='stable')[:k];child[u[chosen]]=True;return child
 selected=[];used=np.zeros(len(u),dtype=bool);rolling=np.zeros(len(u),dtype=float)
 for step in range(k):
  if step==0:best=int(np.argmax(rel))
  else:
   values=rel-penalty*rolling/min(step,10);values[used]=-np.inf;best=int(np.argmax(values))
   if len(u)-step>1:
    top=np.partition(values,-2)[-2:];tol=128*np.finfo(float).eps*(step+1)*max(1.,abs(penalty))
    if top[-1]-top[-2]<=tol:
     cand=np.flatnonzero(~used);recent=np.asarray(selected[-10:],dtype=int);red=self.C[np.ix_(u[cand],u[recent])].mean(axis=1);best=int(cand[np.argmax(rel[cand]-penalty*red)])
  selected.append(best);used[best]=True
  if step<k-1:
   rolling+=self.C[u,u[best]]
   if len(selected)>10:rolling-=self.C[u,u[selected[-11]]]
 child[u[selected]]=True;return child
