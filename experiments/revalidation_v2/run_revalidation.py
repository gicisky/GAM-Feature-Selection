#!/usr/bin/env python3
"""Versioned, resumable revalidation. Original inputs/results are read-only.
CPU threads fixed to one per worker; no outcome-based rerun or parameter tuning.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,hashlib,argparse,platform,traceback,multiprocessing as mp,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source/code'),str(ROOT/'source/utils')]
import numpy as np,pandas as pd
import bench_gam as bg
import bench_bare as bb
import bench_gam_sota as sota
from Filter import reliefFScore
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold,KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
import sklearn, scipy
DATA=ROOT.parent.parent/'data'
OLD=ROOT.parent.parent/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_jsonl(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def taskid(t):return hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest()[:20]
def atomic(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def redundancy(X,idx):
 z=X[:,idx]; constant=int((np.std(z,axis=0)==0).sum())
 if len(idx)<2:return {'red_finite_pairs':0.,'red_pair_zero':0.,'red_legacy':0.,'undefined_pairs':0,'total_pairs':0,'constant_features':constant}
 with np.errstate(invalid='ignore',divide='ignore'):C=np.abs(np.corrcoef(z.T))
 v=C[np.triu_indices(len(idx),1)];finite=np.isfinite(v)
 return {'red_finite_pairs':float(v[finite].mean()) if finite.any() else None,'red_pair_zero':float(np.where(finite,v,0).mean()),'red_legacy':float(v.mean()) if finite.all() else 0.,'undefined_pairs':int((~finite).sum()),'total_pairs':len(v),'constant_features':constant}
def de_gam(X,Y,k,gam,N=100,T=100,CR=.9):
 p=gam.p;k=min(k,p);pop=gam.safe_init(N,k);fit=np.array([bg.evaluate(X,Y,np.flatnonzero(s)) for s in pop]);bi=fit.argmax();best_fit,best=fit[bi],pop[bi].copy()
 for t in range(T):
  for i in range(N):
   a,b,c=np.random.choice([j for j in range(N) if j!=i],3,replace=False);diff=pop[b]!=pop[c];donor=pop[a].copy();donor[diff]=pop[b][diff];jr=np.random.randint(p);trial=pop[i].copy()
   for j in range(p):
    if np.random.rand()<CR or j==jr:trial[j]=donor[j]
   child=gam.safe_crossover(pop[i],trial,k,gen=t,max_gen=T);child=gam.safe_mutate(child,k);nf=bg.evaluate(X,Y,np.flatnonzero(child))
   if nf>=fit[i]:pop[i]=child;fit[i]=nf
   if nf>best_fit:best_fit,best=nf,child.copy()
 return np.flatnonzero(best)
BARE={'GA':bb.ga_bare,'CSA':bb.csa_bare,'SSA':bb.ssa_bare,'PSO':bb.pso_bare,'DE':bb.de_bare}
GAM={'GA':bg.ga_gam_safe,'CSA':bg.csa_gam_safe,'SSA':bg.ssa_gam_safe,'PSO':bg.pso_gam_safe,'DE':de_gam,'AO':sota.ao_gam_safe,'GWO':bg.gwo_gam_safe}
def helper_functions(file,names,namespace):
 tree=ast.parse((ROOT/'source/experiments'/file).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),file,'exec'),namespace)
helper_ns={'np':np,'mi_c':mutual_info_classif}
helper_functions('run_filter_wrapper.py',{'mrmr_ranking'},helper_ns)
mrmr_ranking=helper_ns['mrmr_ranking']
ind_ns={'np':np,'evaluate_fn':lambda X,Y,idx:bg.evaluate(X,Y,idx)}
helper_functions('run_independent_wrappers.py',{'bgwo','bpso','topk_binary','k_repair'},ind_ns)
INDEPENDENT={'BGWO':ind_ns['bgwo'],'BPSO':ind_ns['bpso']}
class Evaluator:
 def __init__(self,X,Y,folds=5,seed=42,stratified=False,trace_interval=100):
  self.X=X;self.Y=Y;splitter=StratifiedKFold(folds,shuffle=True,random_state=seed) if stratified and np.unique(Y,return_counts=True)[1].min()>=folds else KFold(folds,shuffle=True,random_state=seed);self.cv=list(splitter.split(X,Y));self.cache={};self.calls=0;self.best=-np.inf;self.best_idx=None;self.trace=[];self.trace_interval=trace_interval
 def reference(self,idx):
  return float(np.mean(cross_val_score(KNeighborsClassifier(5,metric='manhattan',n_jobs=1),self.X[:,idx],self.Y,cv=self.cv,n_jobs=1))) if len(idx) else 0.
 def __call__(self,X,Y,idx):
  self.calls+=1;idx=np.sort(np.asarray(idx,dtype=int));key=idx.tobytes()
  if key not in self.cache:self.cache[key]=self.reference(idx)
  v=self.cache[key]
  if v>self.best:self.best=v;self.best_idx=idx.copy()
  if self.calls%self.trace_interval==0:self.trace.append({'fitness_calls':self.calls,'unique_fitness_calls':len(self.cache),'best_so_far':float(self.best)})
  return v
_DATA_CACHE={}
def getdata(file):
 if file not in _DATA_CACHE:
  p=DATA/file;d=pd.read_csv(p);X=d.iloc[:,:-1].to_numpy(dtype=float);Y=LabelEncoder().fit_transform(d.iloc[:,-1]);_DATA_CACHE.clear();_DATA_CACHE[file]=(X,Y,sha(p))
 return _DATA_CACHE[file]
def run_task(t):
 ident=taskid(t);target=ROOT/'results'/f'{ident}.json'
 if target.exists():return {'id':ident,'status':'resumed'}
 try:
  X,Y,datahash=getdata(t['file']);D=X.shape[1];seed=(20260713+17*t['run']) if t['experiment']=='E10_matched' else 42+7*t['run'];outer=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,Y));k=max(1,int(D*.05));folds=[0] if t['experiment'] in ['E7','lambda','convergence','independent','filter_guided'] else range(5)
  if t['experiment'] in ['E7','convergence','independent','filter_guided']:k=min(k,200)
  fold_results=[];task_start=time.perf_counter()
  for fi in folds:
   foldpath=ROOT/'results'/f'{ident}_fold{fi}.json'
   if foldpath.exists():fold_results.append(json.loads(foldpath.read_text()));continue
   tr,te=outer[fi];st=time.perf_counter();sc=MinMaxScaler();xt=sc.fit_transform(X[tr]);xe=sc.transform(X[te]);yt,ye=Y[tr],Y[te];pre=np.arange(D)
   if t.get('prefilter',False):pre=np.argsort(-mutual_info_classif(xt,yt,random_state=42))[:5000];xt=xt[:,pre];xe=xe[:,pre]
   preprocessing_s=time.perf_counter()-st;inner_seed=seed+fi if t['experiment']=='E10_matched' else 42;inner_folds=3 if t['experiment'] in ['E10_matched','convergence','independent','filter_guided'] else 5;ev=Evaluator(xt,yt,inner_folds,inner_seed,stratified=t['experiment'] in ['convergence','independent','filter_guided'],trace_interval=t['N']);old=bg.evaluate;bg.evaluate=ev;gam=None;cross_calls=[];guide_start=time.perf_counter()
   if t['mode']!='Bare':
    mi_seed=seed+fi if t['experiment']=='E10_matched' else 42;mi=mutual_info_classif(xt,yt,random_state=mi_seed);rf=reliefFScore(xt,yt);alpha=t.get('alpha',.5);scores=alpha*(mi-mi.min())/(mi.max()-mi.min()+1e-12)+(1-alpha)*(rf-rf.min())/(rf.max()-rf.min()+1e-12)
    if t.get('guidance')=='ReliefF-guided-GAM':scores=(rf-rf.min())/(rf.max()-rf.min()+1e-12)
    elif t.get('guidance')=='mRMR-guided-GAM':scores=mrmr_ranking(xt,yt,max_features=min(2000,xt.shape[1]))
    gam=bg.GAM(xt,scores);original=gam.safe_crossover
    def crossover(p1,p2,kk,gen=0,max_gen=100,**kw):
     cross_calls.append((int(gen),int(max_gen)));return original(p1,p2,kk,gen=gen,max_gen=max_gen,penalty=t['lambda'])
    gam.safe_crossover=crossover
   guidance_s=time.perf_counter()-guide_start;np.random.seed(seed+fi if t['experiment']=='E10_matched' else seed);search_start=time.perf_counter()
   try:
    if t['experiment']=='independent' and t['base'] in ['BGWO','BPSO']:idx=INDEPENDENT[t['base']](xt,yt,k,t['N'],t['T'])[0]
    elif gam is None:idx=BARE[t['base']](xt,yt,k,t['N'],t['T'])
    else:idx=GAM[t['base']](xt,yt,k,gam,t['N'],t['T'])
   finally:bg.evaluate=old
   search_s=time.perf_counter()-search_start;idx=np.sort(np.asarray(idx,dtype=int));assert len(idx)==k and len(np.unique(idx))==k
   returned=ev.reference(idx)
   # Exact historical best is required only for GAM. Bare implementations are frozen.
   if gam is not None and returned<ev.best-1e-12:raise AssertionError(f'returned fitness {returned} below best {ev.best}')
   knn=KNeighborsClassifier(5,metric='manhattan',n_jobs=1);pred=knn.fit(xt[:,idx],yt).predict(xe[:,idx]);metrics={'accuracy':float(accuracy_score(ye,pred)),'balanced_accuracy':float(balanced_accuracy_score(ye,pred)),'macro_f1':float(f1_score(ye,pred,average='macro',zero_division=0)),'weighted_f1':float(f1_score(ye,pred,average='weighted',zero_division=0))}
   rec={'task_id':ident,'task':t,'fold':fi,'data_sha256':datahash,'n_samples':len(X),'D_original':D,'D_search':xt.shape[1],'k':k,'seed':seed,'inner_seed':inner_seed,'inner_folds':inner_folds,'inner_strategy':'stratified_if_class_counts_allow' if t['experiment'] in ['convergence','independent','filter_guided'] else 'KFold','train_indices':tr.tolist(),'test_indices':te.tolist(),'selected_features':pre[idx].tolist(),'predictions':pred.tolist(),'test_labels':ye.tolist(),'metrics':metrics,'redundancy':redundancy(xt,idx),'fitness_trace':ev.trace,'runner_sha256':sha(__file__),'fitness_calls':ev.calls,'unique_fitness_calls':len(ev.cache),'returned_fitness':returned,'best_evaluated_fitness':float(ev.best),'crossover_calls':len(cross_calls),'generation_range':[min(g[0] for g in cross_calls),max(g[0] for g in cross_calls)] if cross_calls else None,'preprocessing_s':preprocessing_s,'guidance_s':guidance_s,'search_s':search_s,'elapsed_fold_s':time.perf_counter()-st,'timing_note':'single-thread workers; deterministic fitness cache enabled, runtime not directly comparable with legacy uncached results','version':'GAM-revalidation-v2'}
   atomic(foldpath,rec);fold_results.append(rec);print(json.dumps({'event':'fold_complete','task':t,'fold':fi,'accuracy':metrics['accuracy'],'seconds':rec['elapsed_fold_s']},ensure_ascii=False),flush=True)
   del gam
  aggregate={'task_id':ident,'task':t,'version':'GAM-revalidation-v2','folds':len(fold_results),'accuracy_mean':float(np.mean([f['metrics']['accuracy'] for f in fold_results])),'redundancy_mean':float(np.mean([f['redundancy']['red_finite_pairs'] for f in fold_results])) if all(f['redundancy']['red_finite_pairs'] is not None for f in fold_results) else None,'elapsed_this_invocation_s':time.perf_counter()-task_start,'fold_files':[f'{ident}_fold{fi}.json' for fi in folds]};atomic(target,aggregate);return {'id':ident,'status':'complete','task':t,'accuracy':aggregate['accuracy_mean']}
 except Exception:
  err={'id':ident,'task':t,'traceback':traceback.format_exc()};atomic(ROOT/'results'/f'{ident}.error.json',err);return {'id':ident,'status':'error','task':t,'error':err['traceback']}
def plan():
 tasks=[];files={p.name.split('(')[0]:p.name for p in DATA.glob('*.csv')}
 def add(exp,r,mode=None):
  t={'experiment':exp,'dataset':r['dataset'],'file':files[r['dataset']],'base':r['base'],'mode':mode or r['mode'],'run':int(r['run']),'N':int(r.get('N',100)),'T':int(r.get('T',100)),'lambda':float(r.get('lambda',.6))};t['prefilter']=exp=='E7' or (exp in ['lambda','E4'] and r['dataset'] in ['Brain_Tumor_2','arcene']);tasks.append(t)
 # E1 rerun provides actual subsets for consistent redundancy and exact provenance.
 retained=[]
 for r in load_jsonl(OLD/'bench_foldsafe_bare.jsonl'):
  if any(v==0 for v in r['per_fold_red']) or (r['dataset']=='colon' and r['base']=='SSA' and r['run']==0):add('main',r,'Bare')
  else:retained.append({**r,'reuse_reason':'Bare algorithm unchanged; all saved per-fold Red finite/nonzero, finite-pair convention leaves these values unchanged','source_file':'results/bench_foldsafe_bare.jsonl','source_sha256':sha(OLD/'bench_foldsafe_bare.jsonl'),'result_origin':'retained_legacy_not_rerun'})
 (ROOT/'retained_unchanged_bare.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in retained))
 for r in load_jsonl(OLD/'bench_foldsafe_full.jsonl'):
  if r['protocol']=='fold-safe':add('main',r)
 for r in load_jsonl(OLD/'bench_foldsafe_large_optimized.jsonl'):add('E7',r)
 for r in load_jsonl(OLD/'bench_lambda_sensitivity.jsonl'):add('lambda',{**r,'N':100,'T':100,'mode':'GAM'})
 for r in load_jsonl(OLD/'bench_foldsafe_sota.jsonl'):
  if r['protocol']=='fold-safe':add('E4',r)
 # E10 matched splits/seeds: alpha must not alter split or search random seed.
 for r in load_jsonl(OLD/'bench_alpha_sensitivity_merged_5proc.jsonl'):
  t={'experiment':'E10_matched','dataset':r['dataset'],'file':files[r['dataset']],'base':'GA','mode':'GAM','run':int(r['run']),'N':int(r['N']),'T':int(r['T']),'lambda':float(r['lambda']),'prefilter':False,'alpha':float(r['alpha'])};tasks.append(t)
 # Convergence is recorded against actual evaluator calls, without truncating initialization.
 for r in load_jsonl(DATA.parent/'results/bench_convergence.jsonl'):
  add('convergence',r);tasks[-1]['prefilter']=r['D_orig']>10000
 # Rerun both comparator arms to obtain saved subsets and one consistent Red definition.
 for r in load_jsonl(DATA.parent/'results/bench_independent_wrappers.jsonl'):
  base='GA' if r['method']=='GAM-GA' else r['method'];mode='GAM' if base=='GA' else 'Bare';add('independent',{**r,'base':base,'mode':mode});tasks[-1]['prefilter']=r['D_orig']>10000
 for r in load_jsonl(DATA.parent/'results/bench_filter_guided_gam.jsonl'):
  add('filter_guided',{**r,'mode':'GAM','lambda':r['penalty']});tasks[-1]['prefilter']=r['D_orig']>10000;tasks[-1]['guidance']=r['method']
 unique={taskid(t):t for t in tasks};tasks=list(unique.values())
 # Start with the mechanism most in doubt; order is fixed before observing outcomes.
 tasks.sort(key=lambda t:(0 if t['experiment']=='main' and t['dataset']=='colon' and t['base']=='SSA' else {'main':3,'E7':2,'lambda':4,'E4':5,'E10_matched':1,'convergence':6,'independent':7,'filter_guided':8}[t['experiment']],t['run'],t['dataset'],t['base'],t['mode']))
 atomic(ROOT/'plan.json',tasks);print('planned',len(tasks),'tasks',flush=True);return tasks
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int);ap.add_argument('--experiment');args=ap.parse_args();tasks=plan()
 manifest={'version':'GAM-revalidation-v2','python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'workers':args.workers,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'source').rglob('*.py'))},'runner_sha256':sha(__file__),'seed_policy':'42+7*run; reset same seed each outer fold as original main/E7 protocols','primary_protocol':'outer-fold-safe, guidance built on whole outer training fold; inner-CV guidance not fully nested','data_policy':'header=0 retained; report actual loaded n','revision_notes':json.loads((ROOT/'revision_notes.json').read_text())};atomic(ROOT/'execution_manifest.json',manifest)
 if args.pilot:tasks=[t for t in tasks if t['experiment']=='main' and t['dataset']=='colon' and t['base']=='SSA' and t['run']==0]
 if args.experiment:tasks=[t for t in tasks if t['experiment']==args.experiment]
 if args.limit:tasks=tasks[:args.limit]
 atomic(ROOT/'status.json',{'state':'running','pid':os.getpid(),'started_unix':time.time(),'planned':len(tasks),'completed_before':sum((ROOT/'results'/f'{taskid(t)}.json').exists() for t in tasks),'workers':args.workers})
 pending=[t for t in tasks if not (ROOT/'results'/f'{taskid(t)}.json').exists()];print('pending',len(pending),flush=True)
 with mp.get_context('spawn').Pool(args.workers) as pool:
  for i,result in enumerate(pool.imap_unordered(run_task,pending,chunksize=1)):
   atomic(ROOT/'status.json',{'state':'running' if result['status']!='error' else 'failed','pid':os.getpid(),'planned':len(tasks),'completed_this_run':i+1,'pending_at_start':len(pending),'last_result':result,'updated_unix':time.time(),'workers':args.workers})
   print(json.dumps({'event':'task_complete','completed_this_run':i+1,'total_this_run':len(pending),**result},ensure_ascii=False),flush=True)
   if result['status']=='error':raise RuntimeError('Stopping scheduling after failed task; see saved error. No failed row is silently discarded.')
 atomic(ROOT/'status.json',{'state':'complete','pid':os.getpid(),'planned':len(tasks),'finished_unix':time.time()})
if __name__=='__main__':
 import fcntl
 lock=(ROOT/'run.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Another revalidation coordinator is already running')
 main()
