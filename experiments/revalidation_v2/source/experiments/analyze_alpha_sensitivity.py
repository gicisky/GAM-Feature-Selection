"""Analyze HMRF alpha sensitivity results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
EXP_DIR = Path(__file__).resolve().parents[1]
IN_DEFAULT = EXP_DIR / 'results' / 'bench_alpha_sensitivity_shard_*_of_5.jsonl'
OUT_DIR_DEFAULT = EXP_DIR / 'results' / 'alpha_sensitivity_summary'
METRICS = ['accuracy_mean', 'balanced_accuracy_mean', 'macro_f1_mean', 'weighted_f1_mean', 'redundancy_mean', 'runtime_sec']

def load_jsonl(path):
    import glob
    files = sorted(glob.glob(str(path))) or [str(path)]
    rows = []
    for fp in files:
        with Path(fp).open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if 'error' not in row:
                        rows.append(row)
    if not rows:
        raise SystemExit(f'No successful rows in {path}')
    return pd.DataFrame(rows)

def fmt(x):
    return 'NA' if pd.isna(x) else f'{float(x):.4f}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(IN_DEFAULT))
    parser.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_jsonl(args.input)
    df.to_csv(out_dir / 'all_successful_runs.csv', index=False)
    by_dataset_alpha = df.groupby(['dataset', 'alpha'], as_index=False)[METRICS].mean().sort_values(['dataset', 'alpha'])
    by_dataset_alpha.to_csv(out_dir / 'dataset_alpha_means.csv', index=False)
    macro = df.groupby('alpha', as_index=False)[METRICS].mean().sort_values('alpha')
    macro.to_csv(out_dir / 'macro_alpha_means.csv', index=False)
    best_rows = []
    for dataset, sub in by_dataset_alpha.groupby('dataset'):
        best = sub.loc[sub['accuracy_mean'].idxmax()]
        default = sub[np.isclose(sub['alpha'], 0.5)]
        default = default.iloc[0] if len(default) else None
        best_rows.append({'dataset': dataset, 'best_alpha': float(best['alpha']), 'best_accuracy': float(best['accuracy_mean']), 'alpha_0_5_accuracy': float(default['accuracy_mean']) if default is not None else np.nan, 'gap_0_5_to_best': float(best['accuracy_mean'] - default['accuracy_mean']) if default is not None else np.nan, 'best_macro_f1': float(best['macro_f1_mean']), 'alpha_0_5_macro_f1': float(default['macro_f1_mean']) if default is not None else np.nan, 'best_redundancy': float(best['redundancy_mean']), 'alpha_0_5_redundancy': float(default['redundancy_mean']) if default is not None else np.nan})
    best = pd.DataFrame(best_rows).sort_values('dataset')
    best.to_csv(out_dir / 'alpha_0_5_gap_to_best.csv', index=False)
    md = []
    md.append('# HMRF Alpha Sensitivity Summary\n')
    md.append('HMRF_alpha = alpha * normalized MI + (1 - alpha) * normalized ReliefF. The experiment is a supplementary robustness check for the fixed alpha=0.5 design choice.\n')
    md.append('## Macro Alpha Means\n')
    md.append('| Alpha | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | Redundancy | Runtime(s) |')
    md.append('|---:|---:|---:|---:|---:|---:|---:|')
    for _, r in macro.iterrows():
        md.append(f"| {r['alpha']:.1f} | {fmt(r['accuracy_mean'])} | {fmt(r['balanced_accuracy_mean'])} | {fmt(r['macro_f1_mean'])} | {fmt(r['weighted_f1_mean'])} | {fmt(r['redundancy_mean'])} | {fmt(r['runtime_sec'])} |")
    md.append('\n## Alpha=0.5 Gap to Dataset Best\n')
    md.append('| Dataset | Best alpha | Best Acc | Acc at alpha=0.5 | Gap |')
    md.append('|---|---:|---:|---:|---:|')
    for _, r in best.iterrows():
        md.append(f"| {r['dataset']} | {r['best_alpha']:.1f} | {fmt(r['best_accuracy'])} | {fmt(r['alpha_0_5_accuracy'])} | {fmt(r['gap_0_5_to_best'])} |")
    md.append('\nRecommended positioning: report this as a supplementary sensitivity check. If alpha=0.5 remains close to the best observed alpha, describe it as a stable default rather than a tuned optimum.')
    (out_dir / 'SUMMARY.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'Wrote {out_dir}')
    print(macro.to_string(index=False))
if __name__ == '__main__':
    main()
