"""Strict component ablation: same GA implementation; only init/crossover/lambda differ."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,time,argparse,multiprocessing as mp,traceback
from pathlib import Path
import numpy as np
import run_revalidation as core
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
ROOT=core.ROOT/'component_ablation';ROOT.mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
class ComponentGAM(core.bg.GAM):
 def __init__(self,X,scores,group):
  self.group=group
  if group in ['C','D']:super().__init__(X,scores)
  else:self.X=X;self.p=X.shape[1];self.scores=scores;self.C=None
 def safe_init(self,N,k):
  if self.group!='A':return super().safe_init(N,k)
  pop=np.zeros((N,self.p),dtype=bool)
  for i in range(N):pop[i,np.random.choice(self.p,min(k,self.p),replace=False)]=True
  return pop
 def safe_crossover(self,p1,p2,k,gen=0,max_gen=100,**kwargs):
  if self.group in ['C','D']:return super().safe_crossover(p1,p2,k,gen=gen,max_gen=max_gen,penalty=.6 if self.group=='D' else 0.)
  # Same feasibility-preserving random branch as GAM. Consume the branch draw
  # for consistency, but never read HMRF scores or correlations in groups A/B.
  np.random.rand();mask=np.random.rand(self.p)<.5;child=np.where(mask,p1,p2).astype(bool);count=int(child.sum())
  if count>k:
   pos=np.flatnonzero(child);np.random.shuffle(pos);child[pos[k:]]=False;count=k
  while count<k:child[np.random.choice(np.flatnonzero(~child))]=True;count+=1
  return child

def plan():
 tasks=[{'dataset':ds,'file':file,'run':r,'group':g,'N':100,'T':100,'outer_folds':5,'inner_folds':5,'seed':42+7*r} for ds,file in [('colon','colon(62,2000).csv'),('DLBCL','DLBCL(77,5469).csv'),('9_Tumors','9_Tumors(60,5726).csv')] for r in range(5) for g in ['A','B','C','D']]
 core.atomic(ROOT/'plan.json',tasks);return tasks

def run(t):
 ident=core.taskid(t);target=ROOT/'results'/f'{ident}.json'
 if target.exists():return {'status':'resumed','task':t}
 try:
  X,y,datahash=core.getdata(t['file']);D=X.shape[1];k=max(1,int(D*.05));folds=[]
  for fi,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=t['seed']).split(X,y)):
   p=ROOT/'results'/f'{ident}_fold{fi}.json'
   if p.exists():folds.append(json.loads(p.read_text()));continue
   start=time.perf_counter();sc=MinMaxScaler();xt=sc.fit_transform(X[tr]);xe=sc.transform(X[te]);yt,ye=y[tr],y[te];scores=np.zeros(D)
   if t['group']!='A':
    mi=mutual_info_classif(xt,yt,random_state=42);rf=core.reliefFScore(xt,yt);scores=.5*(mi-mi.min())/(mi.max()-mi.min()+1e-12)+.5*(rf-rf.min())/(rf.max()-rf.min()+1e-12)
   gam=ComponentGAM(xt,scores,t['group']);ev=core.Evaluator(xt,yt,5,42,trace_interval=100);old=core.bg.evaluate;core.bg.evaluate=ev;np.random.seed(t['seed'])
   try:idx=core.bg.ga_gam_safe(xt,yt,k,gam,100,100)
   finally:core.bg.evaluate=old
   assert len(idx)==k and ev.calls==10100
   assert ev.reference(idx)>=ev.best-1e-12
   pred=KNeighborsClassifier(5,metric='manhattan',n_jobs=1).fit(xt[:,idx],yt).predict(xe[:,idx]);record={'task':t,'fold':fi,'accuracy':float(np.mean(pred==ye)),'redundancy':core.redundancy(xt,idx),'n_samples':len(X),'D':D,'k':k,'data_sha256':datahash,'selected_features':idx.tolist(),'train_indices':tr.tolist(),'test_indices':te.tolist(),'test_labels':ye.tolist(),'predictions':pred.tolist(),'fitness_calls':ev.calls,'unique_fitness_calls':len(ev.cache),'fitness_trace':ev.trace,'elapsed_s':time.perf_counter()-start,'component_runner_sha256':core.sha(__file__),'note':'All groups share bg.ga_gam_safe, tournament selection, acceptance and swap mutation. Only listed components differ.'};core.atomic(p,record);folds.append(record)
   print(json.dumps({'event':'component_fold','task':t,'fold':fi,'seconds':record['elapsed_s'],'accuracy':record['accuracy']}),flush=True)
  result={'task':t,'accuracy_mean':float(np.mean([r['accuracy'] for r in folds])),'folds':5,'elapsed_s':sum(r['elapsed_s'] for r in folds)};core.atomic(target,result);return {'status':'complete',**result}
 except Exception:
  error={'status':'error','task':t,'traceback':traceback.format_exc()};core.atomic(ROOT/'results'/f'{ident}.error.json',error);return error

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--wait-for-main',action='store_true');ap.add_argument('--plan-only',action='store_true');args=ap.parse_args();tasks=plan()
 core.atomic(ROOT/'manifest.json',{'runner_sha256':core.sha(__file__),'core_runner_sha256':core.sha(core.__file__),'groups':{'A':'uniform full-space init, random crossover','B':'HMRF top-pool init only, random crossover','C':'B plus mixed relevance-guided crossover, lambda=0','D':'same as C, lambda=.6'},'budget':'same GA; N=T=100; requested fitness evaluations 10100 per fold; 3 datasets x5 runs x4 groups x5 outer folds','inner_protocol':'guidance fit on whole outer-train, inner KFold5 seed42; outer-test isolated','pool_scope':'initialization only; no permanent feature-space restriction'})
 if args.plan_only:return
 if args.wait_for_main:
  core.atomic(ROOT/'status.json',{'state':'queued_after_revalidation','planned':60,'workers':args.workers})
  while True:
   status=json.loads((core.ROOT/'status.json').read_text())
   if status['state']=='complete':break
   if status['state']=='failed':raise SystemExit('Main queue failed; component queue held for review')
   time.sleep(30)
 pending=[t for t in tasks if not (ROOT/'results'/f'{core.taskid(t)}.json').exists()];core.atomic(ROOT/'status.json',{'state':'running','planned':60,'pending':len(pending),'pid':os.getpid(),'workers':args.workers})
 with mp.get_context('spawn').Pool(args.workers) as pool:
  for i,r in enumerate(pool.imap_unordered(run,pending)):
   print(json.dumps(r),flush=True);core.atomic(ROOT/'status.json',{'state':'running' if r['status']!='error' else 'failed','planned':60,'completed_this_run':i+1,'pending_at_start':len(pending),'last_result':r})
   if r['status']=='error':raise RuntimeError(r['traceback'])
 core.atomic(ROOT/'status.json',{'state':'complete','planned':60,'finished_unix':time.time()})
if __name__=='__main__':main()
