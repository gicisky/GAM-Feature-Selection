"""Resume saved work with validated equivalent kernels; run new ablation first."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,time,fcntl,argparse
import run_revalidation as core
from fast_evaluator import FastEvaluator
from fast_crossover import fast_crossover
import component_ablation as component
core.Evaluator=FastEvaluator;core.bg.GAM.safe_crossover=fast_crossover
ORIGINAL_MAIN_TASK=core.run_task;ORIGINAL_COMPONENT_TASK=component.run
ENGINE={'name':'validated_fast_v1','evaluator_sha256':core.sha(core.ROOT/'fast_evaluator.py'),'crossover_sha256':core.sha(core.ROOT/'fast_crossover.py'),'launcher_sha256':core.sha(__file__)}
def annotate(folder,t,prior):
 ident=core.taskid(t)
 for p in folder.glob(f'{ident}*.json'):
  if p.name in prior:continue
  r=json.loads(p.read_text());r['execution_engine']=ENGINE
  if 'fold' in r:r['timing_note']='Fast equivalent distance/rolling-correlation kernels; tie fallback to reference. Not directly comparable with older kernel timings.'
  core.atomic(p,r)
def fast_main_task(t):
 folder=core.ROOT/'results';prior={p.name for p in folder.glob(f'{core.taskid(t)}*.json')};r=ORIGINAL_MAIN_TASK(t);annotate(folder,t,prior);return r
def fast_component_task(t):
 folder=component.ROOT/'results';prior={p.name for p in folder.glob(f'{core.taskid(t)}*.json')};r=ORIGINAL_COMPONENT_TASK(t);annotate(folder,t,prior);return r
core.run_task=fast_main_task;component.run=fast_component_task
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
 lock=(core.ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for file in ['fast_validation.json','fast_crossover_validation.json','fast_full_fold_validation.json']:
  r=json.loads((core.ROOT/file).read_text());assert r.get('all_exact',r.get('same_subset',False)),file
 core.atomic(core.ROOT/'fast_execution_manifest.json',{'engine':ENGINE,'validation_files':['fast_validation.json','fast_crossover_validation.json','fast_full_fold_validation.json'],'schedule':'strict component ablation first, then resume original queue; two workers','started_unix':time.time()})
 core.atomic(core.ROOT/'status.json',{'state':'running_component_priority','pid':os.getpid(),'planned':1524,'workers':args.workers})
 print('Running the 60-task matched component ablation first',flush=True);component.main()
 print('Resuming original tasks from saved folds',flush=True);core.main()
