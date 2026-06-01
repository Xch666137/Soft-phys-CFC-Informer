---
title: "P2 Replacement-Mechanism Search for C09 (detach aggregate dominance)"
date: 2026-05-25
status: approved
owner: 刘行宇
provenance: brainstorm-skill (Claude + user)
ara_refs:
  claims:    [C07, C08, C09]
  decisions: [N86]
  observations_open: [O35, O39, O40]
  dead_ends_recent: [N84]
related_runs:
  - runs/physformer_c23_{baseline,e3,detach}_vgpu_s{2025,2026,2027}/  # T1, 9 runs
  - scripts/c08_t1_3seed_results.json
---

# P2: Replacement-Mechanism Search for C09 (detach aggregate dominance)

## 1. Motivation

T1 Phase B (2026-05-24/25) hardened **C09** (selective detach dominates aggregate
test MAE / MSE / RMSE / net_ramp_violation across 3 seeds with smallest std), but
also **refuted N84 / O37** — the May 23 single-seed bias-clearance mechanism story
collapsed when checked against s2026/s2027 (3 of 4 mechanism signals flip
direction; see `scripts/c08_t1_3seed_results.json`).

C09 is therefore an *outcome* claim with no surviving mechanism explanation.
The lesson from N84 is sharp: single-seed mechanism stories are unsafe; future
candidates must be designed for cross-seed corroboration from day one.

In parallel, **O39** (e3 wins all 5 component MAEs but loses aggregate) and
**O40** (theory_mae reversed vs aggregate) point to a second cancellation regime
on the encoder-depth axis. These are mechanistic clues, not explanations.

## 2. Candidate Mechanisms

Four candidate mechanisms are drawn from ARA signals; each is paired with the
cheapest diagnostic that can support or refute it without re-running training:

| ID  | Mechanism                              | ARA evidence trail                                  | Cheapest diagnostic |
|-----|----------------------------------------|-----------------------------------------------------|---------------------|
| M1  | **Capacity-cancellation 解耦**         | O39 (component-aggregate paradox on depth axis) + N83 s2027 (cov-cross weakened, agg improved) + C08 capacity regime | D1 — extend C08 variance decomposition to 12 runs (incl. detach×e3), test `cov(theory, residual)` collapse |
| M2  | **Residual fraction 重构**             | `diagnostic_summary` residual_std nearly equal across arms (0.00226 vs 0.00226) but mean differs | D2 — `\|residual\|/(\|theory\|+\|residual\|)` distribution + Pearson(theory_net, residual_net); already-saved npy only |
| M3  | **Flat minimum (sharpness)**           | N82 (detach has smallest cross-seed std on ALL 4 aggregate metrics — reproducibility signature of a flat minimum) | D3 — finite-difference parameter perturbation on encoder weights, report mean Δloss/\|ε\|² |
| M4  | **Encoder representation disentanglement** | User candidate 1 (representation-level); no direct prior evidence | E1 — detach×e3 result naturally adjudicates M1 vs M4 (see §3); explicit hidden-state PCA deferred |

### M1 vs M4 adjudication via E1

E1 = detach×e3 joint, 3-seed (the configuration ARA flagged as **O23 open thread**
since 2026-05-17 and has been unrun across two sessions).

- **If M1**: e3's deeper encoder gives a larger "cancellation budget" through
  shared representation, but detach cuts the residual→encoder→theory backward path
  that *uses* this budget. Prediction: `detach×e3 ≈ detach` on aggregate MAE.
- **If M4**: deeper encoder gives more expressive disentangled hidden states, which
  detach lets the model use cleanly. Prediction: `detach×e3 < detach` on aggregate.

A single joint experiment thus discriminates the two; M4 hidden-state PCA is only
needed if E1 says M4 and we want to confirm the *mechanism* in representation space.

## 3. Selected Approach: P2 (Balanced)

P2 was chosen over P1 (tight loop, single seed) and P3 (full sweep incl. M4 forward
replay) because:

1. **N84 lesson**: any mechanism candidate that wants to enter ARA as a claim must
   be cross-seed-validated. E1 must be 3-seed regardless of P1/P2/P3.
2. **M1 + M2 + M3 are all cheap and orthogonal** — they probe error structure,
   output structure, and loss-landscape geometry respectively. Running them in
   parallel maximizes information per local CPU hour.
3. **M4 is best deferred**: detach×e3 itself adjudicates M1 vs M4. Forward replay
   for hidden-state PCA only pays off if M1 is also refuted by D1.

### Component summary

| ID  | Name                                        | Type        | Where        | Wall-clock    | Data source |
|-----|---------------------------------------------|-------------|--------------|---------------|-------------|
| E1  | detach×e3 cross-3-seed                      | training    | remote GPU   | ~25h (3-par)  | new runs    |
| D1  | M1 cov-cross + per-component bias 时序      | diagnostic  | local CPU    | ~3-4h         | 9 runs `physics_states.npz` + `theory_net.npy` + `residual_net.npy` |
| D2  | M2 residual fraction & corr                 | diagnostic  | local CPU    | ~1-2h         | 9 runs `theory_net.npy` + `residual_net.npy` + `pred.npy` + `true.npy` |
| D3  | M3 parameter-perturbation sharpness         | diagnostic  | local CPU    | ~10-15h       | 9 runs `checkpoint.pth` + test loader |

E1 results, when available, fold back into D1/D2/D3 as a 4th arm (`detach_e3`),
yielding 12 runs total.

## 4. Falsification Protocol

Each sub-experiment defines its falsification *before* running. This protocol
binds the future research-manager run to a clear claim/falsified/supported
verdict.

### E1: detach×e3 cross-3-seed

```
Claim: detach×e3 联合的 aggregate test MAE 跨 3 seed mean 显著 < detach 单独 OR
       ≈ detach 单独 (区分 M4 vs M1).
证伪条件:
  - detach×e3 mean MAE ≥ detach mean MAE + 1 std (即 ≥ 2.010e-3) 且 std 相当
    → "更深 encoder 无叠加效益" → 强证据指向 M1 (capacity 通道被 detach 关闭)
  - detach×e3 mean MAE < detach mean MAE - 1 std (即 < 1.936e-3)
    → "更深 encoder 有叠加效益" → 强证据指向 M4 (disentanglement)
  - detach×e3 跨 seed std > detach 跨 seed std × 2
    → C09 reproducibility 假设受挑战, sharpness-flat-minimum 解释 (M3) 被加强
通过条件: 3 seed 都自然 early stop (E15-E22), 无 NaN/梯度死亡, MAE 范围在 detach
   mean ± 5 std 内, train.log 见 Phase 1→2 cw transition (epoch 8) 正常发生.
基线: T1 c23_detach mean MAE = 1.973e-3 ± 3.7e-5, c23_e3 mean MAE = 2.101e-3 ± 5.3e-5
   (同种子集 {2025, 2026, 2027}, 同 A+B 配置 6dca986).
混杂因素:
  - PCIe 带宽波动 (N69) → 同一 vGPU-32GB 实例, 3-parallel 一致
  - autodl-push ckpt 污染 (N78) → exclude 已修
  - early-stop 提前 → 配置同 N75 (Phase 2b 切除, patience=8/start=12)
  - e3 + detach 交互不稳定 → 失败立即停手, 进 N66-style dead_end
```

### D1: M1 cov-cross + per-component bias 时序

```
Claim: detach 跨 seed 平均 |Cov(theory_net, residual_net)| 显著 < baseline; 同时
   detach 在 per-component bias 上至少 3/5 component 接近 0 (|bias|减小).
证伪条件:
  - detach 三 seed mean |cov(theory, residual)| 与 baseline 三 seed mean 差异 < 1 std → M1 不成立
  - per-component |bias|^detach < |bias|^baseline 在 3 seed × 5 component = 15 个
    对比中成立比例 < 60% (即不到 9/15)
通过条件:
  - detach mean |cov(theory, residual)| < 30% × baseline mean |cov| AND
  - per-component |bias| 至少 3/5 component 在 3 seed mean 上 detach < baseline AND
  - 当 E1 落盘后, detach_e3 的 cov-cross 与 detach 而非 e3 同号 (验证 M1 prediction)
基线: N83 (s2025) baseline cov_cross = -3.57e-5, detach cov_cross = -1.28e-5;
   本次扩到 3 seed mean ± std + 12-run after E1.
混杂因素:
  - s2026 反向 (diag/cov 翻转) → 必须报告 per-seed + mean ± std, 不能只报 mean
  - 数值精度 (cov 量级 1e-5, FP32 浮点误差 ~1e-7 量级, 安全)
  - residual = pred - theory_net (aggregate 1D), 不是 per-component residual
```

### D2: M2 residual fraction & corr

```
Claim: detach 与 baseline 在 |residual|/|theory+residual| 分布上有显著差异 (说明
   residual head 的 "absorbing role" 改变), 即使 residual_std 几乎相等.
证伪条件:
  - K-S test on |residual|/(|theory|+|residual|) 分布: detach vs baseline p > 0.05
    在 3 seed 全部成立
  - 或: |ΔPearson r(theory_net, residual_net)| < 0.05 跨 3 seed 一致
通过条件:
  - K-S p < 0.05 在至少 2/3 seed 成立 AND
  - mean |ΔPearson(theory, residual)| 跨 seed > 0.1 AND
  - residual fraction 分位数 (Q25, Q50, Q75) 至少 1 个分位在 detach vs baseline
    上有 > 10% 相对差异
基线: 当前 diagnostic_summary 仅有 residual_std/mean, 无分布 / corr;
   首次系统测量.
混杂因素:
  - 时间窗口效应 (96 步预测窗口起点不同) → 用全 test set flatten
  - O39 已显示 detach residual 跨 seed std 3× 大于 baseline → 必须 per-seed 报告
  - K-S 在 N≈3M samples 上极敏感, p value 几乎必然小; 重点看 D statistic 量级
```

### D3: M3 parameter-perturbation sharpness

```
Claim: detach 收敛到的 minimum 比 baseline/e3 更平坦, 由此解释 N82 "cross-seed
   std 最小" 现象.
证伪条件:
  - detach 三 seed mean sharpness ≥ baseline mean sharpness 跨任一 ε
  - 或: detach 三 seed sharpness 自身 std > baseline 三 seed std × 2
    (即 detach 的 sharpness 自己就不稳定, 无法支撑"鲁棒解"叙事)
通过条件:
  - detach mean sharpness < baseline mean sharpness × 0.7 AND
  - detach 三 seed sharpness std < baseline 三 seed std AND
  - 单调性: ε ∈ {1e-3, 3e-3, 1e-2} 上 sharpness 单调增 (sanity check on 计算正确性)
基线: 无 (首次测量); 用 baseline_s2025 作 dry-run 校准.
扰动方法:
  - 30 个随机方向 ξ ~ N(0, I) (per-parameter shape), 每个方向 normalize 到 ||ξ||=1
  - 仅扰动 model.encoder.parameters() (不扰 phys_layer / residual_head / film), 与
    detach 因果通路对齐
  - ε ∈ {1e-3, 3e-3, 1e-2}, 每次扰动后 forward 5% test set (~1700 samples), 算
    net_mse_real (test loss term, 与训练监控一致)
  - sharpness := mean over directions of [loss(θ + εξ) - loss(θ)] / ε²
混杂因素:
  - 扰动方向数量 30 可能不够 → 先 30, 若 95% CI 跨 0 则扩到 50-100
  - encoder 参数数量级 ~600k, 高斯方向覆盖性需 mean-of-many; 用 fixed seed 确保
    可复现
  - CPU FP32 vs 训练 FP16/AMP 数值差异 → 接受, 因为 evaluation 模式已经是 FP32
  - test loader 跨 9 runs 共享 (same data); 不要在多个 ckpt 上重新 build loader
```

## 5. Changes (file-by-file)

### New files

```
configs/physformer_c23_detach_e3.yaml         — E1 config (detach + e_layers=3)
scripts/m1_cov_cross_t1.py                    — D1 diagnostic
scripts/m2_residual_fraction.py               — D2 diagnostic
scripts/m3_sharpness_perturbation.py          — D3 diagnostic
docs/plans/2026-05-25-detach-mechanism-search.md — this design doc
```

### No modifications to existing code

E1 inherits `configs/base/v5_base.yaml` and only overrides `e_layers=3` and the
c23-specific schedule fields. No changes to `physformer/`, `run.py`, or loss code.
This keeps E1 a clean ablation point for ARA.

D1/D2/D3 are standalone scripts that read pre-existing artifacts; they do not
touch `physformer/` either.

## 6. Verification Steps

### E1 (remote training)

```
1. Create configs/physformer_c23_detach_e3.yaml
   → verify: yaml loads, e_layers=3 present, detach_mode_phase2=selective present
2. autodl-push to remote vGPU-32GB (instance: connect.westc.seetacloud.com:52613 or current)
   → verify: file lands on remote, no exclude breakage (N78 fix verified)
3. Launch 3-parallel s2025/s2026/s2027
   → verify: Monitor sees 3 PIDs + first epoch wraps within ~5 min
   → verify: train.log shows fresh start, NO "Resumed from checkpoint" line (N78 sanity)
4. Wait for natural early-stop (counter 8/8, expected E15-E22)
   → verify: train.log final lines contain EarlyStopping counter 8/8 and a best epoch
5. Run `python run.py test --config ... --run-name ... --resume` per arm
   → verify: extras/{physics_states.npz, theory_net.npy, residual_net.npy} present;
            metrics.json contains MAE/MSE/RMSE/theory_mae/component_*_mae
6. Run `python run.py export-forecast` per arm (optional but parallels T1)
7. autodl-pull 3 runs back to local runs/
   → verify: ls runs/physformer_c23_detach_e3_vgpu_s* shows 3 dirs each w/ extras/
```

### D1 (local CPU)

```
1. Write scripts/m1_cov_cross_t1.py (reuse analyze_run + extract_y_aux from
   c08_vgpu_3way.py)
2. Run with PYTHONPATH set
   → verify: prints per-run cov_cross_term, var_sum_diag, bias_per_comp;
            writes scripts/m1_cov_cross_t1_results.json
3. Threshold check:
   - mean(|cov_cross|_detach) / mean(|cov_cross|_baseline) < 0.30  ?
   - 3 seed × 5 comp |bias| reduction count ≥ 9/15  ?
4. After E1 lands, re-run with detach_e3 included
   → verify: detach_e3 cov_cross sign and magnitude classified vs detach/e3/baseline
```

### D2 (local CPU)

```
1. Write scripts/m2_residual_fraction.py
2. Run
   → verify: outputs K-S D + p, Pearson r per (arm × seed) and per (arm × arm × seed)
            pair; writes scripts/m2_residual_fraction_results.json
3. Threshold check:
   - K-S detach vs baseline p < 0.05 in ≥ 2/3 seed?
   - mean |ΔPearson(theory, residual)| > 0.1?
4. After E1: re-run incl. detach_e3
```

### D3 (local CPU)

```
1. Write scripts/m3_sharpness_perturbation.py
2. Dry-run on baseline_s2025 with 5 directions × 1 epsilon to estimate timing
   → verify: single forward on 5% test ≤ 30s; 5 dirs × 1 ε ≤ 3 min
   → decision: if single forward > 60s, drop to 2% test set; if > 120s, reduce directions
3. Full run on 9 ckpts × 30 dirs × 3 epsilons = 810 forward passes
   → verify: writes scripts/m3_sharpness_results.json with per-run mean ± std;
            sanity check: sharpness monotonically increases with ε
4. Threshold check:
   - mean(sharpness_detach) / mean(sharpness_baseline) < 0.70?
   - std(sharpness_detach) < std(sharpness_baseline)?
5. After E1: re-run incl. detach_e3 (3 more ckpts)
```

## 7. Schedule

```
Day 1 (today, 2026-05-25):
  [done] Brainstorm approved
  [next] Write design doc + E1 config + D1/D2/D3 scripts (all local, reversible)
  [pause] Confirm with user before pushing E1 to remote
  [later] On approval: autodl-push + launch E1 3-parallel; start D1/D2 locally

Day 2 (2026-05-26):
  - E1 finishes ~mid-day (25h from launch)
  - D1 results on existing 9 runs
  - D2 results on existing 9 runs
  - D3 dry-run baseline_s2025 + decide on full sweep
  - autodl-pull E1; rerun test + export if needed

Day 3 (2026-05-27):
  - D3 full sweep on 9 (or 12) ckpts
  - Rerun D1/D2 incl. detach_e3 (12-run analysis)
  - Compose mechanism-search report
  - /research-manager: promote/falsify M1/M2/M3/M4

Day 4 (optional):
  - If M1/M2/M3 all fail: launch M4 encoder hidden-state forward replay
  - If E1 surprises (e.g., detach×e3 wins big): targeted follow-up experiments
```

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| E1 training NaN (e3 + detach interaction) | low (NVIDIA has not produced NaN; only AMD ROCm did per N66/N67) | blocks E1 | stop immediately on first NaN; record as ARA dead_end; D1/D2/D3 still proceed on existing 9 runs |
| E1 PCIe bandwidth slow (N69 redux) | medium | doubles wall-clock | confirm Gen4 link on instance before launch; accept if 25h slips to 30h |
| autodl-push reintroduces ckpt pollution (N78) | low (config patched) | wasted launch | verify NO "Resumed from checkpoint" in train.log of E1 first epoch |
| D3 single-ckpt forward > 60s on CPU | medium (CPU FP32 inference, model has 600k+ params) | doubles D3 wall-clock | dry-run first; if slow, reduce test subset from 5% → 2% |
| D1 cov-cross dominated by float-precision noise | low | unclear conclusion | use float64 accumulators; N83 already saw stable 1e-5 magnitude |
| D2 K-S over N≈3M samples: p always near 0 | high | distorts interpretation | report D statistic (effect size), not p, as headline; supplement with quantile diffs |
| M1/M2/M3 all fail to differentiate detach from baseline | medium | another N84-style mechanism-open status | escalate to M4 (hidden-state PCA) and an OOD/cross-portfolio robustness test (open thread per ARA PAPER.md) |
| M3 perturbation directions too few → noisy mean | medium | wrong conclusion | start at 30, expand to 100 if 95% CI of mean(sharpness) crosses 0 |
| E1 1/3 seed gradient death (N22-style) | low | partial result | retain 2/3 seeds as preliminary; flag in ARA; do not promote claim on 2 seeds |
| Local D3 sweep blocks remote E1 monitoring | medium | delayed pull / diagnosis | run D3 in background (Bash run_in_background); reserve foreground for Monitor + remote events |

## 9. ARA Integration Plan

After execution, /research-manager will be invoked to write:

```
N87 (decision):    P2 three-track parallel mechanism search launched
N88 (experiment):  E1 detach×e3 cross-3-seed (results)
N89 (experiment):  D1 M1 cov-cross + per-component bias on 12 runs (results)
N90 (experiment):  D2 M2 residual fraction & corr on 12 runs (results)
N91 (experiment):  D3 M3 parameter-perturbation sharpness on 12 runs (results)
N92 (insight or dead_end): synthesis of M1/M2/M3/M4 verdicts
N93 (decision):    next-step strategy (claim promotion, or M4 escalation, or OOD)

Candidate claim promotions (depending on outcomes):
  C10  if M1 supported:  detach 通过断开 capacity-cancellation 反向通路实现 aggregate dominance
  C10' if M3 supported:  detach 通过 flat-minimum 实现 cross-seed std 最小
  Both can coexist if data supports — they are not mutually exclusive.

Updates to existing claims:
  C07: evidence base extended with E1 (detach×e3) data point
  C08: capacity regime extended to encoder-depth axis (already O39); E1 widens to (depth, detach) factorial
  C09: outcome unchanged regardless of M1/M2/M3 outcome; only the mechanism interpretation changes
```

## 10. Appendix

### A. Numeric baselines from T1 (for falsification thresholds)

```
T1 9-run aggregate (real MW), mean ± std across seeds {2025, 2026, 2027}:
  Aggregate MAE:
    baseline 2.069e-3 ± 1.34e-4
    e3       2.101e-3 ± 5.3e-5
    detach   1.973e-3 ± 3.7e-5   ← C09 winner

  Aggregate MSE (MW²):
    baseline 8.111e-6 ± 6.8e-7
    e3       8.146e-6 ± 2.6e-7
    detach   7.377e-6 ± 1.4e-7   ← smallest std

  net_ramp_violation:
    baseline 3.728e-3 ± 1.06e-3
    e3       3.971e-3 ± 4.1e-4
    detach   2.989e-3 ± 5.0e-4   ← smallest mean

  Component MAE means (real MW): e3 wins all 5 (O39)
  Theory MAE means: baseline 2.475e-3 < detach 2.491e-3 < e3 2.577e-3 (O40)

Single-seed (s2025) C08 from N74:
  baseline cov_cross_term = -3.57e-5
  e3       cov_cross_term = (largest |cov|, exact in c08_vgpu_3way_results.json)
  detach   cov_cross_term = -1.28e-5  (|cov| 64% smaller than baseline)
```

### B. Threshold derivations

```
D1 cov-cross threshold 30%:
  - Based on N74 s2025 evidence: detach has 64% smaller |cov|
  - For cross-seed mean to register as "collapsed", require at least 70% smaller
    (i.e. detach mean < 30% × baseline mean)
  - Less stringent than s2025 single-seed (64%) because of multi-seed averaging

D1 bias 9/15 threshold:
  - 5 components × 3 seeds = 15 comparisons
  - 9/15 = 60% one-sided binomial p ≈ 0.30 with null (0.5); 10/15 = 0.15; 11/15 = 0.06
  - 9/15 chosen as conservative — single-seed s2025 had 4/5 bias clearances; need
    ≥ 3/5 average sustained across seeds
  - HIGHER bar than N83's 1/4 signal replication for O37; designed so M1 must beat
    the failed N84 mechanism's replication rate

D2 ΔPearson 0.1 threshold:
  - residual is by construction a function of theory; small Pearson differences are
    expected; 0.1 chosen as smallest difference visible across 9-run sample without
    being attributable to per-seed initialization noise
  - To be re-estimated after dry-run on s2025 if differences are << 0.1 (rendering
    threshold inert) or >> 0.1 (rendering threshold trivially passed)

D3 sharpness ratio 0.70 threshold:
  - C09 detach std on agg MAE (3.7e-5) is 26% of baseline std (1.34e-4); if
    sharpness explains reproducibility, expect sharpness ratio in the same ballpark
  - 0.70 chosen as conservative middle ground: tighter than 0.85 (which any
    regularization would pass), looser than 0.5 (which N82's 26% might suggest
    but could be aliased by sample size)
```

### C. Symmetry with prior protocols

This plan mirrors the structure of T1 (N77-N82): config-fixed, 3-seed, A+B
configuration, natural early-stop, autodl-push/pull with N78-fixed exclude.

Each diagnostic script is patterned on `scripts/c08_t1_3seed.py` (reuse
`analyze_run` from `c08_vgpu_3way.py`), keeping the new artifacts compatible with
ARA evidence binding.

### D. Open questions deferred to next round

- M4 encoder hidden-state PCA / disentanglement metrics — deferred unless M1/M2/M3
  all fail
- OOD / cross-portfolio robustness of detach — separate session, open per
  PAPER.md "next steps"
- e3 cancellation-on-depth (O39) standalone confirmation via e_layers ∈ {2, 3, 4}
  sweep — separate task, deferred until M1 verdict
- Why e3 has REVERSED theory_mae ranking (O40) — relates to M2 (residual takes on
  more bias) but not pursued separately in this plan

---

## 11. Approval

- [x] User approved P2 (2026-05-25, in /brainstorm session)
- [x] Falsification protocol reviewed
- [ ] Ready to push E1 to remote (separate explicit user confirmation required)
