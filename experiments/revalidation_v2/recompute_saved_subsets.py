import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
sys.path.insert(0,str(Path(__file__).resolve().parent));from run_revalidation import redundancy,DATA,ROOT,load_jsonl,sha
sources={'E9':ROOT.parent.parent/'results/bench_multimetric_robustness_clean.jsonl','E10':ROOT.parent.parent/'results/bench_alpha_sensitivity_merged_5proc.jsonl'}
allsummary={}
for exp,path in sources.items():
 rows=load_jsonl(path);output=[];foldout=[]
 for ds in sorted(set(r['dataset'] for r in rows)):
  dsrows=[r for r in rows if r['dataset']==ds];file=dsrows[0]['dataset_file'];p=DATA/file;d=pd.read_csv(p,header=None);X=d.iloc[:,:-1].to_numpy(dtype=float);y=LabelEncoder().fit_transform(d.iloc[:,-1]);maxerr=0
  for r in dsrows:
   reds=[];metrics=[];outer=StratifiedKFold(r['outer_folds'],shuffle=True,random_state=20260713+17*r['run']+(int(r['alpha']*1000) if exp=='E10' else 0))
   for fi,(tr,te) in enumerate(outer.split(X,y)):
    saved=r['fold_records'][fi];idx=np.array(r.get('selected_features_by_fold',[f['selected_features'] for f in r['fold_records']])[fi],dtype=int);sf=MinMaxScaler();xt=sf.fit_transform(X[tr]);xe=sf.transform(X[te]);pred=KNeighborsClassifier(r['knn_k'],metric='manhattan',n_jobs=1).fit(xt[:,idx],y[tr]).predict(xe[:,idx]);mm={'accuracy':accuracy_score(y[te],pred),'balanced_accuracy':balanced_accuracy_score(y[te],pred),'macro_f1':f1_score(y[te],pred,average='macro',zero_division=0),'weighted_f1':f1_score(y[te],pred,average='weighted',zero_division=0)};err=max(abs(mm[k]-saved[k]) for k in mm);maxerr=max(maxerr,err)
    if err>1e-12:raise AssertionError(f'{exp} {ds} run{r["run"]} fold{fi}: prediction metrics do not reproduce, error {err}')
    red=redundancy(xt,idx);reds.append(red);metrics.append(mm);foldout.append({'experiment':exp,'dataset':ds,'mode':r['mode'],'alpha':r.get('alpha'),'run':r['run'],'fold':fi,'legacy_saved_red':saved['redundancy'],'prediction_metric_max_error':err,**red})
   rec={'experiment':exp,'dataset':ds,'mode':r['mode'],'alpha':r.get('alpha'),'run':r['run'],'data_sha256':sha(p),'n_samples':len(X)}
   for key in ['red_finite_pairs','red_pair_zero','red_legacy']:rec[key]=float(np.mean([a[key] for a in reds])) if all(a[key] is not None for a in reds) else None
   for key in metrics[0]:rec[key]=float(np.mean([a[key] for a in metrics]))
   output.append(rec)
  print(exp,ds,'records',len(dsrows),'max prediction metric error',maxerr,flush=True)
 pd.DataFrame(output).to_csv(ROOT/f'{exp}_saved_subsets_recomputed.csv',index=False);pd.DataFrame(foldout).to_csv(ROOT/f'{exp}_saved_subsets_folds.csv',index=False)
 summary=pd.DataFrame(output).groupby(['mode'] if exp=='E9' else ['alpha'])[['accuracy','balanced_accuracy','macro_f1','weighted_f1','red_finite_pairs','red_pair_zero','red_legacy']].mean();summary.to_csv(ROOT/f'{exp}_saved_subsets_summary.csv');allsummary[exp]={'source':str(path),'source_sha256':sha(path),'records':len(output),'folds':len(foldout),'max_prediction_metric_error':max(a['prediction_metric_max_error'] for a in foldout),'summary':summary.reset_index().to_dict('records')}
(ROOT/'saved_subset_recomputation.json').write_text(json.dumps(allsummary,indent=2))
