"""Equivalent uniform 5-NN inner CV; ambiguous neighbor boundaries use sklearn.
Original reference scoring remains available for independent final-best checks.
"""
import numpy as np
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import KNeighborsClassifier
from run_revalidation import Evaluator
class FastEvaluator(Evaluator):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.fast_scores=0;self.boundary_fallback_folds=0
 def fast_reference(self,idx):
  if not len(idx):return 0.
  z=np.ascontiguousarray(self.X[:,idx],dtype=np.float64)
  if not np.isfinite(z).all():return super().reference(idx)
  distances=squareform(pdist(z,metric='cityblock'));scores=[];classes=np.unique(self.Y)
  for tr,te in self.cv:
   d=distances[np.ix_(te,tr)];ntrain=len(tr);knn=5
   if ntrain<=knn:
    self.boundary_fallback_folds+=1;scores.append(KNeighborsClassifier(5,metric='manhattan',n_jobs=1).fit(z[tr],self.Y[tr]).score(z[te],self.Y[te]));continue
   part=np.argpartition(d,knn,axis=1);neighbors=part[:,:knn];near=np.take_along_axis(d,neighbors,axis=1);kth=near.max(axis=1);nextdist=d[np.arange(len(te)),part[:,knn]]
   # Conservative floating-point summation bound for k-dimensional distances.
   tolerance=16*len(idx)*np.finfo(np.float64).eps*np.maximum(1.,np.maximum(kth,nextdist))
   if np.any(nextdist-kth<=tolerance):
    self.boundary_fallback_folds+=1;scores.append(KNeighborsClassifier(5,metric='manhattan',n_jobs=1).fit(z[tr],self.Y[tr]).score(z[te],self.Y[te]));continue
   labels=self.Y[tr][neighbors];votes=(labels[:,:,None]==classes[None,None,:]).sum(axis=1);pred=classes[votes.argmax(axis=1)];scores.append(float(np.mean(pred==self.Y[te])))
  self.fast_scores+=1;return float(np.mean(scores))
 def __call__(self,X,Y,idx):
  self.calls+=1;idx=np.sort(np.asarray(idx,dtype=int));key=idx.tobytes()
  if key not in self.cache:self.cache[key]=self.fast_reference(idx)
  value=self.cache[key]
  if value>self.best:self.best=value;self.best_idx=idx.copy()
  if self.calls%self.trace_interval==0:self.trace.append({'fitness_calls':self.calls,'unique_fitness_calls':len(self.cache),'best_so_far':float(self.best)})
  return value
