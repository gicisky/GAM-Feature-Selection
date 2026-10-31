from pathlib import Path
import shutil,ast,json,hashlib
root=Path(__file__).resolve().parent;src=root.parent.parent;manifest=[]
for folder in ['code','utils','experiments']:
 for p in (src/folder).glob('*.py'):
  dst=root/'source'/folder/p.name;shutil.copy2(p,dst);manifest.append({'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'original_source_manifest.json').write_text(json.dumps(manifest,indent=2))
class Fix(ast.NodeTransformer):
 def visit_Lambda(self,node):
  node=self.generic_visit(node)
  if any(a.arg=='g' for a in node.args.args) and any(a.arg=='mg' for a in node.args.args) and any(isinstance(n,ast.Attribute) and n.attr=='safe_crossover' for n in []):pass
  # Only the known crossover wrapper signature, not arbitrary lambdas.
  if [a.arg for a in node.args.args]==['p1','p2','k','g','mg']:
   for a in node.args.args:
    a.arg={'g':'gen','mg':'max_gen'}.get(a.arg,a.arg)
   for n in ast.walk(node.body):
    if isinstance(n,ast.Name):n.id={'g':'gen','mg':'max_gen'}.get(n.id,n.id)
  return node
for p in (root/'source'/'experiments').glob('*.py'):
 tree=Fix().visit(ast.parse(p.read_text()));ast.fix_missing_locations(tree);p.write_text(ast.unparse(tree)+'\n')
p=root/'source/code/bench_gam_sota.py';tree=ast.parse(p.read_text());changes=0
for fn in tree.body:
 if isinstance(fn,ast.FunctionDef):
  for n in ast.walk(fn):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='safe_crossover':
    for kw in n.keywords:
     if kw.arg=='gen' and isinstance(kw.value,ast.Name) and kw.value.id=='T':kw.value.id='t';changes+=1
p.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
# New SSA integration v2: keep continuous proposal equations, explicitly refine
# each top-k proposal with GAM, retain exact discrete best and its fitness.
ssa='''def ssa_gam_safe(X, Y, k, gam, N=100, T=100):
    k_eff = min(k, gam.p)
    salps = gam.safe_init(N, k_eff).astype(float)
    def discretize(s):
        threshold = np.sort(s)[-k_eff]
        idx = np.where(s >= threshold)[0]
        if len(idx) > k_eff:
            idx = np.random.choice(idx, k_eff, False)
        binary = np.zeros(gam.p, dtype=bool)
        binary[idx] = True
        return binary
    subsets = salps.astype(bool)
    fits = np.array([evaluate(X, Y, np.flatnonzero(s)) for s in subsets])
    bi = int(fits.argmax())
    best_fit, best = float(fits[bi]), subsets[bi].copy()
    best_salp = salps[bi].copy()
    for gen in range(T):
        c1 = 2.0 - 2.0 * gen / T
        for i in range(N):
            if i == 0:
                salps[i] = np.clip(salps[i] + c1 * (best_salp - salps[i]) * np.random.rand(gam.p), 0, 1)
            else:
                salps[i] = np.clip((salps[i] + salps[i-1]) / 2, 0, 1)
        new_subsets = []
        new_fits = []
        for i in range(N):
            proposal = discretize(salps[i])
            child = gam.safe_crossover(proposal, best, k_eff, gen=gen, max_gen=T)
            child = gam.safe_mutate(child, k_eff)
            value = evaluate(X, Y, np.flatnonzero(child))
            new_subsets.append(child)
            new_fits.append(value)
            if value > best_fit:
                best_fit, best = float(value), child.copy()
                best_salp = child.astype(float)
        subsets = np.array(new_subsets)
        fits = np.array(new_fits)
    return np.flatnonzero(best)
'''
p=root/'source/code/bench_gam.py';tree=ast.parse(p.read_text());replacement=ast.parse(ssa).body[0]
for i,n in enumerate(tree.body):
 if isinstance(n,ast.FunctionDef) and n.name=='ssa_gam_safe':tree.body[i]=replacement
# Keep source comments in other sections, replace only function block.
original=p.read_text();old=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='ssa_gam_safe');lines=original.splitlines(keepends=True);lines[old.lineno-1:old.end_lineno]=[ssa];p.write_text(''.join(lines).replace("warnings.filterwarnings('ignore')",'# warnings remain visible in revalidation'))
print('Prepared immutable copied sources; AO/POA/TSO gen=T fixes:',changes)
(root/'revision_notes.json').write_text(json.dumps({'version':'GAM-revalidation-v2','SSA':'new explicit GAM refinement integration and exact historic discrete best; not old implementation','wrapper':'forward gen and max_gen without swallowing','sota_generation_calls_corrected':changes,'data':'preserve existing header=0 loading; report actual n','redundancy':'report finite-pair mean plus legacy zero and pair-zero sensitivity','GA_CSA':'preserve existing pipeline implementations; no claim of isolated operator-only ablation'},indent=2))
