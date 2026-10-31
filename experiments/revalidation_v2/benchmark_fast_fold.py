import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import ast,json,time
from pathlib import Path
import run_revalidation as c
from fast_evaluator import FastEvaluator
from fast_crossover import fast_crossover
c.Evaluator=FastEvaluator;c.bg.GAM.safe_crossover=fast_crossover
root=c.ROOT/'fast_benchmark';(root/'results').mkdir(parents=True,exist_ok=True);plan=json.loads((c.ROOT/'plan.json').read_text());t=next(t for t in plan if t['experiment']=='main' and t['dataset']=='arcene' and t['base']=='GA' and t['mode']=='GAM' and t['run']==0)
fn=next(n for n in ast.parse(Path(c.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_task')
for node in ast.walk(fn):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='folds' for t in node.targets):node.value=ast.List(elts=[ast.Constant(0)],ctx=ast.Load())
ns=dict(vars(c));ns['ROOT']=root;exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),c.__file__,'exec'),ns);r=ns['run_task'](t);assert r['status']!='error',r
ident=c.taskid(t);a=json.loads((c.ROOT/'results'/f'{ident}_fold0.json').read_text());b=json.loads((root/'results'/f'{ident}_fold0.json').read_text());assert a['selected_features']==b['selected_features'];assert a['predictions']==b['predictions'];assert a['returned_fitness']==b['returned_fitness'];assert a['fitness_trace']==b['fitness_trace'];result={'same_subset':True,'same_predictions':True,'same_fitness_trace':True,'same_best_fitness':True,'slow_fold_s':a['elapsed_fold_s'],'fast_fold_s':b['elapsed_fold_s'],'speedup':a['elapsed_fold_s']/b['elapsed_fold_s']};(c.ROOT/'fast_full_fold_validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
