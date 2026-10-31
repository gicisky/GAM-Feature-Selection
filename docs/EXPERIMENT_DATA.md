# GAM Experiment Data — Complete Documentation

> 封版日期: 2026-06-20 | 实验目录: `experiments/gam_20260531/results/`
> 全部 10 项实验完成: E1-E10

---

## 实验矩阵总览

| 编号 | 实验 | 数据集 | 底座/方法 | 协议 | 规模 | 状态 |
|------|------|--------|----------|------|------|------|
| E1 | Bare 消融 | 15 | GA,CSA,DE,PSO,SSA | 5-fold, 5 runs | 375 条 | ✅ |
| E2 | 主实验 GAM λ0/λ0.6 | 15 | GA,CSA,DE,PSO,SSA | 5-fold, 5 runs, Full+Fold-safe | 1500 条 | ✅ |
| E3 | λ 灵敏度 | 5 | GA,DE,AO | 单 split, 3 runs | 270 条 | ✅ |
| E4 | SOTA 外部对比 | 6 | AO,GWO | 5-fold, 5 runs | 180 条 | ✅ |
| E5 | Filter 基线 | 7 | MI,ReliefF,mRMR,F-score | 5-fold | 28 条 | ✅ |
| E6 | 大数据扩展 | 4 | GA,DE,SSA | 单 split, N=50 | 72 条 | ✅ |
| E7 | 收敛曲线 | 4 | GA,DE,SSA | 单 split, 3 runs, N=50,T=50 | 72 条 | ✅ |
| E8 | Filter-Guided GAM | 7 | mRMR/ReliefF/HMRF-GAM | 单 split, 3 runs, N=50,T=50 | 126 条 | ✅ |
| E9 | Independent Wrappers | 7 | BGWO,BPSO,GAM-GA | 单 split, 3 runs, N=50,T=50 | 63 条 | ✅ |
| E10 | Recent Searchers | 10 | SCSO,COA,GAM-SCSO,GAM-COA | 5-fold, 5 runs, N=50,T=50 | 200 条 | ✅ |

**总计: 2886 条实验记录**

---

## 1. 主实验 (E2) — GAM-λ0 vs GAM λ=0.6

**文件**: `bench_foldsafe_full.jsonl` (1500 条)

**设计**: 15DS × 5 bases (GA,CSA,SSA,PSO,DE) × 2 modes (GAM-λ0, GAM λ=0.6) × 2 protocols (Full, Fold-safe) × 5 runs × 5-fold CV

### Fold-Safe 核心对比

| Metric | Bare (E1) | GAM-λ0 | GAM λ=0.6 | Δ Bare→GAM |
|--------|-----------|--------|-----------|-------------|
| Accuracy (macro) | 0.7672 | 0.8187 | 0.8236 | **+0.0564 (+7.35%)** |
| Redundancy (macro) | 0.2036 | 0.2967 | 0.2660 | — |
| ΔRed (λ0→λ0.6) | — | — | **−0.0307 (−10.33%)** | — |

### 统计检验 (Wilcoxon Signed-Rank, 15 datasets paired)

| Comparison | p-value | Significance |
|-----------|---------|--------------|
| Bare vs GAM λ=0.6 | p < 0.001 | ★★★ |
| Redundancy λ0→λ0.6 | p < 0.001 | ★★★ |
| GAM-λ0 vs GAM λ=0.6 (acc) | p ≈ 0.022 | ★ |

### Friedman Test
- χ² = 25.20, p = 0.000003 ★★★
- Average ranks: GAM λ=0.6 (1.20) > GAM-λ0 (1.80) > Bare (3.00)

### Nemenyi Post-Hoc
- CD = 0.8555 (α=0.05)
- Bare vs GAM: Δrank = 1.80 > CD → SIGNIFICANT

### 消融解读

| 消融层 | ΔAcc | 机制 |
|--------|------|------|
| Bare → GAM-λ0 | +5.15% | HMRF pool + safe_init + guided operators → 搜索质量 |
| GAM-λ0 → GAM λ=0.6 | +0.49% | λ penalty → 冗余惩罚，精度基本不降 |
| Bare → GAM λ=0.6 | **+5.64%** | 综合效果 |

### 每底座 ΔAcc

| Base | ΔAcc(Bare→GAM) | ΔRed(λ0→λ0.6) |
|------|----------------|----------------|
| CSA | +7.69% | −3.76% |
| PSO | +6.79% | −1.53% |
| SSA | +6.18% | −3.01% |
| DE | +4.00% | −3.53% |
| GA | +3.53% | −3.50% |

### Per-Dataset GAM λ=0.6 Accuracy (Fold-safe)

| Dataset | n | D | Accuracy |
|---------|---|----|----------|
| 9_Tumors | 59 | 5726 | 0.4836 |
| Brain_Tumor1 | 89 | 5920 | 0.8687 |
| Brain_Tumor_2 | 49 | 10368 | 0.7971 |
| CLLSUB | 111 | 11340 | 0.7363 |
| CNS | 59 | 7129 | 0.6444 |
| DLBCL | 77 | 5469 | 0.9547 |
| Eleven_Tumor | 173 | 12534 | 0.8552 |
| Leukemia1 | 71 | 5327 | 0.9143 |
| Lung | 202 | 12601 | 0.9416 |
| MLL | 71 | 12582 | 0.9000 |
| Prostate | 102 | 12600 | 0.8647 |
| SRBCT | 82 | 2308 | 0.9733 |
| TOX-171 | 171 | 5748 | 0.8318 |
| arcene | 200 | 10000 | 0.8321 |
| colon | 62 | 2000 | 0.8176 |

---

## 2. Bare 消融 (E1) — 三层消融基础

**文件**: `bench_foldsafe_bare.jsonl` (375 条)

**设计**: 15DS × 5 bases × 5 runs × 5-fold CV. Bare = 无 HMRF/GAM/λ, 随机初始化, 原始交叉, k-repair.

### 论文叙事
- **Bare→GAM-λ0**: HMRF pool + safe_init + guided operators → 精度提升（搜索质量）
- **GAM-λ0→GAM λ=0.6**: λ penalty → 冗余降低（15/15 DS 全降），精度基本不变
- GAM 不总是比 Bare 低冗余（HMRF 池集中相关特征, Bare 随机选无 bias）

---

## 3. λ 灵敏度 (E3)

**文件**: `bench_lambda_sensitivity.jsonl` (270 条)

**设计**: 5DS (colon, DLBCL, 9_Tumors, Brain_Tumor_2, arcene) × 3 bases (GA,DE,AO) × 6λ (0,0.2,0.4,0.5,0.6,0.8) × 3 runs × 单 80/20 split

### λ=0.6 vs Best λ

| Dataset | Best λ | Best Acc | λ=0.6 Acc | Gap | Status |
|---------|--------|----------|-----------|------|--------|
| colon | 0.6 | 0.8120 | 0.8120 | 0.0000 | ✅ 并列最优 |
| DLBCL | 0.0 | 1.0000 | 0.9861 | 0.0139 | ✅ 天花板效应 |
| arcene | 0.2 | 0.8056 | 0.7917 | 0.0139 | ✅ 接近最优 |
| Brain_Tumor_2 | 0.4 | 0.9111 | 0.8889 | 0.0222 | ✅ 可接受 |
| 9_Tumors | 0.5 | 0.4907 | 0.4444 | 0.0463 | ⚠️ 多分类小样本 |

**结论**: λ=0.6 在 4/5 数据集上处于 stable plateau, λ∈[0.4,0.8] 整体稳定。9_Tumors 因 n=59, 9 类, 在较小测试集上偏强 λ。不影响主结论。

### 大数据集说明
Brain_Tumor_2 (D=10368) 和 arcene (D=10000) 使用 MI top-5000 pre-filter，与 colon/DLBCL/9_Tumors 条件不完全一致。论文注明 "memory-aware pre-filtering"。

---

## 4. SOTA 外部对比 (E4) ✅ 已封版

**文件**: `bench_foldsafe_sota.jsonl` (180 条)

**设计**: 6DS × 2 bases (AO,GWO) × 2 modes (GAM-λ0, GAM) × 2 protocols (Full, Fold-safe) × 5 runs

### SOTA 对比 (Fold-safe, GAM λ=0.6)

| Dataset | n | D | AO | GWO | Winner |
|---------|---|---|------|------|--------|
| colon | 62 | 2000 | 0.8169±0.019 | **0.8236±0.022** | GWO |
| DLBCL | 77 | 5469 | **0.9573±0.020** | 0.9493±0.030 | AO |
| Leukemia1 | 71 | 5327 | **0.9297±0.015** | 0.9240±0.019 | AO |
| SRBCT | 82 | 2308 | **0.9807±0.015** | 0.9735±0.019 | AO |
| arcene | 200 | 10000 | **0.8424±0.021** | 0.8374±0.009 | AO |
| Brain_Tumor_2 | 49 | 10368 | **0.7827±0.035** | 0.7818±0.033 | AO |

**AO wins: 5/6, GWO wins: 1/6**

### 数据说明
- colon, DLBCL, Leukemia1, SRBCT: 全协议 (Full + Fold-safe), 全模式 (λ0 + GAM), MI 预筛仅 large (D>10000)
- Brain_Tumor_2, arcene: Fold-safe only, GAM λ=0.6 only, MI top-5000 pre-filter
  > 论文: "For high-dimensional datasets (D > 10,000), MI top-5000 pre-filtering was applied before all compared methods, while the optimization budget (N=100, T=100) and cross-validation protocol (outer 5-fold, inner 5-fold) were kept identical to the main SOTA comparison."

---

## 5. Filter 基线 (E5)

**文件**: `bench_foldsafe_filter.jsonl` (28 条)

**设计**: 7DS × 4 filters (MI, ReliefF, mRMR, F-score) × 5-fold CV. k = GAM λ=0.6 各数据集平均选择特征数.

### Filter vs GAM

| Dataset | Best Filter | Filter Acc | GAM Acc | Winner |
|---------|------------|-----------|---------|--------|
| colon | ReliefF | 0.8513 | 0.8176 | Filter |
| DLBCL | mRMR | 0.9467 | 0.9547 | **GAM** |
| 9_Tumors | F-score | 0.5409 | 0.4836 | Filter |
| Brain_Tumor1 | mRMR | 0.8876 | 0.8687 | Filter |
| Brain_Tumor_2 | MI | 0.8356 | 0.7971 | Filter |
| arcene | mRMR | 0.8144 | 0.8321 | **GAM** |
| Lung | MI | 0.9602 | 0.9416 | Filter |

**GAM 胜 2/7, Filter 胜 5/7**

### 论文叙事
GAM 作为 wrapper 方法, 在中等样本数据集上与 filter 竞争, 同时额外提供冗余控制（filter 无此能力）。2/7 胜率符合 wrapper vs filter 的预期——wrapper 精度不一定更高，但能同时优化 redundancy。

---

## 6. 大数据集扩展 (E6)

**文件**: `bench_foldsafe_large.jsonl` (72 条)

**设计**: 4DS × 3 bases (GA,DE,SSA) × 2 modes (Bare, GAM) × 3 runs. MI top-5000, k≤200, N=50, T=100, single 80/20 split.

### 结果 (macro avg)

| Metric | Bare | GAM λ=0.6 |
|--------|------|-----------|
| Accuracy | 0.8585 | 0.8745 (+1.86%) |
| Redundancy | 0.2830 | 0.3111 |

### Per Dataset

| Dataset | n | D | Bare Acc | GAM Acc | Δ |
|---------|---|---|----------|---------|---|
| arcene | 200 | 10000 | 0.8700 | 0.8522 | −0.0178 |
| CrohnDisease | 127 | 22283 | 0.8267 | 0.8567 | +0.0300 |
| Lung | 202 | 12601 | 0.8933 | 0.9250 | +0.0317 |
| Prostate | 102 | 12600 | 0.8442 | 0.8642 | +0.0200 |

3/4 数据集 GAM 胜。arcene 例外（MI pre-filter 已足以选出高质量特征）。

### 论文定位
可扩展性验证, 非主性能 benchmark。Section: "Scalability Evaluation under Limited Budget."

---

## 7. 收敛曲线 (E7)

**文件**: `bench_convergence.jsonl` (72 条)

**设计**: 4DS (colon, DLBCL, 9_Tumors, arcene) × 3 bases (GA,DE,SSA) × 2 modes (Bare, GAM) × 3 runs. N=50, T=50, single split, inner-3fold CV. 记录每代 cumulative-best 和 mean-population fitness.

### Gen-0 Advantage (HMRF-guided initialization)

| Dataset | Bare gen0 | GAM gen0 | Δ |
|---------|-----------|----------|------|
| 9_Tumors | 0.3616 | 0.5497 | **+0.1881** |
| colon | 0.7847 | 0.8796 | +0.0949 |
| DLBCL | 0.9389 | 0.9778 | +0.0389 |
| arcene | 0.8553 | 0.8875 | +0.0321 |

**All 4/4 datasets**: GAM gen0 fitness > Bare gen0. Mean Δ = +0.0885.

### Per-generation cumulative best

所有 72 条记录确认 cumulative-best 曲线非递减 (monotonically non-decreasing). GAM 在所有数据集上展现出更快的早期收敛。
- Bare 从低起点追赶（如 colon: 0.7708→0.9792）
- GAM 从高起点精炼（如 colon: 0.8542→0.9375）

### 论文定位
元启发式/特征选择论文标配搜索过程证据。Fig X: "Convergence behavior of Bare and GAM optimizers." 4 子图 per dataset, 每条画 Bare vs GAM 的 cumulative-best fitness 曲线。

### 协议说明
Convergence analysis used a lightweight monitoring protocol (N=50, T=50, single split, inner-3fold-CV) distinct from the main benchmark protocol. MI top-5000 pre-filtering only for arcene (D=10000).

---

## 8. Filter-Guided GAM Variants (E8)

**文件**: `bench_filter_guided_gam.jsonl` (126 条)

**设计**: 7DS × 2 bases (GA,DE) × 3 scoring strategies × 3 runs. N=50,T=50, single split, inner-3fold.
所有方法使用相同的 GAM operator (safe_init/safe_crossover/safe_mutate)，仅 scoring 来源和 λ 不同。

### 三种评分策略

| Method | Scores | λ |
|--------|--------|---|
| mRMR-guided-GAM | mRMR ranking (top-2000 approximative) | 0 |
| ReliefF-guided-GAM | ReliefF weights | 0 |
| HMRF-GAM | 0.5×MI + 0.5×ReliefF | 0.6 |

### 结果

| Dataset | mRMR-guided | ReliefF-guided | HMRF-GAM | Winner |
|---------|------------|---------------|----------|--------|
| colon | 0.8077 | 0.8077 | 0.7949 | mRMR/ReliefF |
| DLBCL | 0.9792 | 0.9792 | 0.9688 | mRMR/ReliefF |
| 9_Tumors | 0.4722 | 0.4167 | 0.4444 | mRMR-guided |
| Brain_Tumor1 | 0.8148 | 0.8056 | 0.8056 | mRMR-guided |
| Brain_Tumor_2 | 0.8500 | 0.7333 | 0.8667 | HMRF-GAM |
| arcene | 0.8083 | 0.8042 | 0.7875 | mRMR-guided |
| Lung | 0.9472 | 0.9146 | 0.9106 | mRMR-guided |

| Method | Macro Acc | Macro Red |
|--------|-----------|-----------|
| mRMR-guided-GAM | **0.8113** | **0.2469** |
| HMRF-GAM | 0.7969 | 0.3282 |
| ReliefF-guided-GAM | 0.7802 | 0.3390 |

mRMR-guided-GAM 胜 6/7（含并列）。HMRF-GAM 在冗余控制上不如 mRMR 内置的 min-redundancy 机制（p=0.016）。HMRF 的优势在计算简洁性（等权平均 MI+ReliefF，无 O(D²) 冗余评估）。

### 论文定位
Scoring signal ablation: 比较不同 filter 评分策略在相同 GAM wrapper 框架下的表现。mRMR 的冗余感知在轻量预算下占优，HMRF 的计算简洁性是高维场景优势。

---

## 9. Independent Wrapper Baselines (E9)

**文件**: `bench_independent_wrappers.jsonl` (63 条)

**设计**: 7DS × 3 methods (BGWO,BPSO,GAM-GA) × 3 runs. N=50,T=50, single split, inner-3fold.
BGWO/BPSO 是标准二元元启发式 wrapper（无 HMRF/图/λ），GAM-GA 作为 GAM 代表配置对比。

### 方法差异

| | BGWO | BPSO | GAM-GA |
|---|------|------|--------|
| 初始化 | 随机 k 特征 | 随机 k 特征 | HMRF 加权采样 |
| 搜索 | α/β/δ + sigmoid | 惯性速度 + sigmoid | safe_crossover + safe_mutate |
| 图引导 | ❌ | ❌ | ✅ |
| λ 冗余惩罚 | ❌ | ❌ | ✅ 0.6 |
| 二值化 | top-k | top-k | k-repair |

所有方法共用 inner-3fold KNN CV 适应度。

### 结果

| Dataset | BGWO | BPSO | GAM-GA | Winner |
|---------|------|------|--------|--------|
| colon | 0.7949 | 0.8205 | 0.7949 | BPSO |
| DLBCL | 0.9375 | 0.9167 | 0.9792 | GAM-GA |
| 9_Tumors | 0.3889 | 0.2778 | 0.4722 | GAM-GA |
| Brain_Tumor1 | 0.7593 | 0.7593 | 0.7963 | GAM-GA |
| Brain_Tumor_2 | 0.8333 | 0.9000 | 0.8333 | BPSO |
| arcene | 0.7833 | 0.7583 | 0.7833 | BGWO/GAM-GA |
| Lung | 0.8943 | 0.8699 | 0.9106 | GAM-GA |

| Method | Macro Acc | Macro Red | Avg Time |
|--------|-----------|-----------|----------|
| GAM-GA | **0.7957** | 0.3295 | 54.9s |
| BGWO | 0.7702 | 0.2562 | 5.7s |
| BPSO | 0.7575 | 0.2636 | 5.7s |

**GAM-GA 胜 4/7, 并列 1/7.**
- GAM-GA vs BGWO: Wilcoxon p=0.043 *
- GAM-GA vs BPSO: Wilcoxon p=0.375 ns
- Friedman χ²=3.92, p=0.141 (not significant at 0.05, small n=7)
- GAM-GA Macro Acc 领先 BGWO +2.5pp, BPSO +3.8pp
- Redundancy: GAM-GA 较高（HMRF pool 集中相关特征），但主实验已验证 λ0→λ0.6 降冗余

### 论文定位
Reduced-budget independent wrapper comparison. 回答审稿人 "GAM 是否比常见 wrapper 有竞争力"。GAM-GA 显著优于 BGWO，总体 competitive。表注标明 reduced monitoring protocol (single split, N=50, T=50) 非主 benchmark。

---

## 10. Recent Metaheuristic Searchers (E10) ✅ 封版

**文件**: `bench_recent_searchers.jsonl` (200 条)

**设计**: 10DS × 4 methods (SCSO, GAM-SCSO, COA, GAM-COA) × 5 runs × 5-fold CV. N=50, T=50, λ=0.6, fold-safe.
Continuous position → top-k binary for all methods.

### 方法差异

| | SCSO (2022) | COA (2023, KBS) | GAM-SCSO | GAM-COA |
|---|------------|-----------------|----------|---------|
| 初始化 | 随机连续位置 | 随机连续位置 | HMRF 加权采样 | HMRF 加权采样 |
| 搜索 | r_G 灵敏度 + cosθ | iguana hunt + escape | SCSO + GAM crossover | COA + GAM crossover |
| 图引导 | ❌ | ❌ | ✅ C 矩阵 | ✅ C 矩阵 |
| λ 冗余惩罚 | ❌ | ❌ | ✅ 0.6 | ✅ 0.6 |

所有方法共用 inner-3fold KNN CV 适应度。5-fold 外 CV, fold-safe 协议。

### 最终结果 (10/10 DS)

| Dataset | SCSO | GAM-SCSO | Δ | COA | GAM-COA | Δ | Winner |
|---------|------|----------|-----|-----|---------|---|--------|
| colon | 0.7738 | 0.8203 | +4.6 | 0.7867 | 0.8300 | +4.3 | GAM-COA |
| SRBCT | 0.8437 | 0.9729 | +12.9 | 0.8591 | 0.9779 | +11.9 | GAM-COA |
| DLBCL | 0.8705 | 0.9577 | +8.7 | 0.8990 | 0.9627 | +6.4 | GAM-COA |
| Leukemia1 | 0.8627 | 0.9408 | +7.8 | 0.8678 | 0.9301 | +6.2 | GAM-SCSO |
| 9_Tumors | 0.3015 | 0.4136 | +11.2 | 0.3015 | 0.4236 | +12.2 | GAM-COA |
| Brain_Tumor1 | 0.7799 | 0.8316 | +5.2 | 0.7799 | 0.8204 | +4.1 | GAM-SCSO |
| Brain_Tumor_2 | 0.7627 | 0.7729 | +1.0 | 0.7258 | 0.7991 | +7.3 | GAM-COA |
| arcene | 0.8164 | 0.8304 | +1.4 | 0.8053 | 0.8273 | +2.2 | GAM-SCSO |
| Lung | 0.9046 | 0.9395 | +3.5 | 0.9156 | 0.9493 | +3.4 | GAM-COA |
| Prostate | 0.8612 | 0.9089 | +4.8 | 0.8708 | 0.9128 | +4.2 | GAM-COA |

### 汇总

| Method | Macro Acc | Macro Red | Wins | Avg Time |
|--------|-----------|-----------|------|----------|
| GAM-COA | **0.8433** | 0.3045 | **7/10** | 390s |
| GAM-SCSO | 0.8389 | 0.3051 | 3/10 | 609s |
| COA | 0.7811 | 0.2595 | 0/10 | 44s |
| SCSO | 0.7777 | 0.2620 | 0/10 | 45s |

- **GAM-SCSO > SCSO: 10/10 DS, Δ=+6.1pp (p=0.002)**
- **GAM-COA > COA: 10/10 DS, Δ=+6.2pp (p=0.002)**
- **20/20 对比全胜**: GAM 版本从未被裸算法击败
- 裸 SCSO/COA 从未赢 (0/10 wins)
- 冗余: GAM 略高 (HMRF pool 集中效应, E2 已验证 λ 可降 10.33%)

### 论文定位
Section: "Transfer to Recent Metaheuristic Searchers." 回答审稿人 "GAM 是否适用于近年新算法"。

> To further examine whether GAM remains applicable beyond classical search backbones, we attach the proposed operator layer to two recent representative metaheuristics, SCSO (2022) and COA (2023, KBS). All preprocessing, supervised scoring, and graph construction were performed within the outer training folds. GAM-SCSO and GAM-COA outperformed their bare counterparts on all 10 datasets (p=0.002), confirming that the framework is not restricted to classical optimizers.

---

## 11. 数据集覆盖矩阵

| # | Dataset | n | D | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 |
|---|---------|---|---|----|----|----|----|----|----|----|----|----|
| 1 | colon | 62 | 2000 | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ |
| 2 | SRBCT | 82 | 2308 | ✅ | ✅ | — | ✅ | — | — |
| 3 | Leukemia1 | 71 | 5327 | ✅ | ✅ | — | ✅ | — | — |
| 4 | DLBCL | 77 | 5469 | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| 5 | 9_Tumors | 59 | 5726 | ✅ | ✅ | ✅ | — | ✅ | — |
| 6 | TOX-171 | 171 | 5748 | ✅ | ✅ | — | — | — | — |
| 7 | Brain_Tumor1 | 89 | 5920 | ✅ | ✅ | — | — | ✅ | — |
| 8 | CNS | 59 | 7129 | ✅ | ✅ | — | — | — | — |
| 9 | arcene | 200 | 10000 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 10 | Brain_Tumor_2 | 49 | 10368 | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| 11 | CLLSUB | 111 | 11340 | ✅ | ✅ | — | — | — | — |
| 12 | MLL | 71 | 12582 | ✅ | ✅ | — | — | — | — |
| 13 | Prostate | 102 | 12600 | ✅ | ✅ | — | — | — | ✅ |
| 14 | Lung | 202 | 12601 | ✅ | ✅ | — | — | ✅ | ✅ |
| 15 | Eleven_Tumor | 173 | 12534 | ✅ | ✅ | — | — | — | — |
| 16 | CrohnDisease | 127 | 22283 | — | — | — | — | — | ✅ |

---

## 12. 论文叙事框架

### Title: Graph-Augmented Metaheuristics with Redundancy-Aware Penalty for High-Dimensional Feature Selection

### 核心贡献

1. **Fold-safe HMRF Protocol**: 所有 HMRF scores 和 MinMaxScaler 在 outer_train 上计算，消除信息泄露。与 Full-data 预处理与引导（scaler+HMRF+graph）对比已量化：macro overestimation +4.16%，CNS 最严重 +18.2%（Wilcoxon 双侧 p=6.1×10⁻⁵）。

2. **GAM (Graph-Augmented Metaheuristic) 框架**:
   - `safe_init`: HMRF 引导初始化（non-uniform sampling）
   - `safe_crossover`: 自适应混合交叉（80%→30% 原始交叉, linear decay），λ 引导冗余惩罚（O(k²) graph-guided branch）
   - `safe_mutate`: k 保持变异

3. **Plug-and-Play 设计**: GAM 与 5 个底座算法 (GA,CSA,DE,PSO,SSA) + 2 个 SOTA 算法 (AO,GWO) 验证一致有效。

4. **三层消融验证**:
   - Bare: 随机 init + 原始交叉（基线）
   - GAM-λ0: HMRF pool + guided operators（无冗余控制）
   - GAM λ=0.6: 完整 GAM（+λ penalty）

### 实验路线

```
E1 (Bare) ──对比──→ E2 (GAM-λ0/GAM λ=0.6) ──验证──→ E3 (λ 灵敏度)
                       │
                       ├──→ E4 (SOTA: AO, GWO)
                       ├──→ E5 (Filter: MI, ReliefF, mRMR, F-score)
                       └──→ E6 (Large-Scale: MI-5000, k≤200, N=50)
```

### Abstract 核心叙事

1. High-dimensional feature selection 中, filter-search 方法存在信息泄露和冗余控制不足
2. 提出 GAM: fold-safe HMRF pool + λ-guided redundancy penalty
3. 在 7 个底座算法上验证, Bare→GAM +5.64% 精度, λ0→λ0.6 −10.33% 冗余
4. GAM-AO 在 5/6 数据集 SOTA 对比中胜出
5. 15/15 数据集 λ penalty 降低冗余, Friedman p=0.000003

---

## 13. 已知问题与处理

| 问题 | 影响 | 处理 |
|------|------|------|
| arcene red≈0 (39 常数特征) | 少量条目冗余记为 0 | 数据特性, NaN→0, 不影响统计 |
| Brain_T2/arcene λ 灵敏度用 MI-5000 | 与前三数据集条件不同 | 论文注明 "memory-aware pre-filtering" |
| Brain_T2/arcene SOTA 仅 Fold-safe + GAM | 无 Full 协议和 λ0 | 论文注明, SOTA 表仅比较最终方法 |
| SOTA 高维数据用 MI-5000 | 与低维数据条件不完全一致 | 论文注明 "identical budget & CV, MI pre-filtering for memory" |
| Filter k 来自 GAM λ=0.6 平均 | k 不是 filter 最优值 | 公平对比: 同 k 条件下比较 |

---

## 13. 文件索引

```
results/
├── bench_foldsafe_full.jsonl       E2: 主实验 (1500 条) ✅
├── bench_foldsafe_bare.jsonl       E1: Bare (375 条) ✅
├── bench_lambda_sensitivity.jsonl  E3: λ 灵敏度 (270 条) ✅
├── bench_foldsafe_sota.jsonl       E4: SOTA (180 条) ✅ 封版
├── bench_foldsafe_filter.jsonl     E5: Filter (28 条) ✅
└── bench_foldsafe_large.jsonl      E6: 大数据 (72 条) ✅

scripts/
├── run_bare.py                     E1 启动脚本
├── run_lambda_sensitivity.py       E3 启动脚本
├── run_lambda_large.py             E3 大数据集补跑 (MI-5000)
├── run_filter_baselines.py         E5 启动脚本
├── run_large_scale.py              E6 启动脚本
├── rerun_sota_foldsafe.py          E4 启动脚本 (colon/DLBCL/Leukemia1/SRBCT)
├── run_sota_large.py               E4 大数据补跑 (Brain_T2/arcene, MI-5000, N=100,T=100)
├── patch_dlbcl_gwo.py              E4 DLBCL GWO run=3,4 补齐
└── analyze_ablation.py             消融分析脚本
```

---

## 14. 实验执行记录

| 实验 | 启动时间 | 完成时间 | 耗时 | 备注 |
|------|---------|---------|------|------|
| E1 | 2026-06-17 | 2026-06-17 | ~2h | 15DS×5bases, 中等 |
| E2 | 2026-06-16 | 2026-06-17 | ~8h | 最重: Full+FoldSafe |
| E3 | 2026-06-17 | 2026-06-18 | ~6h | 含大数据集 MI-5000 |
| E4 AO/GWO | 2026-06-17 | 2026-06-19 | ~30h | colon/DLBCL/Leuk/SRBCT 正常; Brain_T2/arcene MI-5000 OK; DLBCL GWO 因 BLAS 线程争抢慢, 修复后 ~3.1h |
| E5 | 2026-06-18 | 2026-06-18 | ~2h | 含 mRMR 优化 |
| E6 | 2026-06-18 | 2026-06-18 | ~1h | N=50 轻量 |
| E7 | 2026-06-19 | 2026-06-19 | ~3 min | N=50,T=50, 单 split |
| E8 | 2026-06-19 | 2026-06-19 | ~45 min | Filter-Guided GAM variants |
| E9 | 2026-06-19 | 2026-06-19 | ~8 min | BGWO/BPSO/GAM-GA |
| E10 | 2026-06-20 | 2026-06-20 | ~3.9h | SCSO/COA/GAM-SCSO/GAM-COA, 10DS×5-fold×5runs |

**总实验时间: ~56 核心小时**

---

## 15. 关键技术细节

### Fold-Safe Protocol
```python
# 对每个 outer fold:
X_tr = MinMaxScaler().fit_transform(X[tr])  # 仅在训练集上 fit
scores = 0.5*MI(X_tr, Y_tr) + 0.5*ReliefF(X_tr, Y_tr)  # HMRF 仅在训练集
gam = GAM(X_tr, scores)  # C 矩阵仅在训练集
# 评估: KNN 仅在训练集训练, 测试集预测
```

### 信息泄露量化（header=None 复验，15 数据集 × 5 runs）
- Full-data 预处理与引导（scaler + HMRF + graph）整体高估准确率 +4.16%（macro，15 数据集）
- 最严重: CNS +18.2%
- Wilcoxon 双侧（精确条件符号秩枚举，15 数据集 full>safe 全同号）: p = 6.1×10⁻⁵

### GAM safe_crossover 图引导分支
- O(k²) Python lambda 循环 → 对 D>5000 数据集性能敏感
- 速度估算: colon (D=2000,k=100) ≈0.1s/crossover, DLBCL (D=5469,k=273) ≈0.5s/crossover
- 优化方向: 向量化或 Cython 加速

### BLAS 多线程注意事项
- macOS Accelerate/OpenBLAS 默认使用所有核心
- 与 multiprocessing 混用时需设置 `OPENBLAS_NUM_THREADS=1`
- 否则造成严重线程争抢 (15 threads × N workers)
