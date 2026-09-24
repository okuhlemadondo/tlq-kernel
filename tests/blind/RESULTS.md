# Blind case — results

The pre-registered runs (R1–R4) used the frozen v0.2 code. Their raw output is in `out_preregistered/`. The post-hoc runs (P1–P4) used v0.3 with the corrected encoding and are reported separately. The independent evaluator is `independent.py`, which does not import `tlq`.

## Pre-registered (v0.2, frozen)

| Run | Status | What happened | Independent check |
|---|---|---|---|
| R1 team, default budget | `NOT_ESTABLISHED` | The only applicable solved class (TEAM_ENUM, 3.3×10⁵ profiles) exceeds the 2×10⁵ budget. Common information is blocked (generated loci; label collision). Fan-outs are uncertified. | optimum exists: **0.825** |
| R2 team, budget ×10 | `FOUND` | TEAM_ENUM gives **0.825** with 8 inequivalent optimal solutions (376 s). One false **REFUTED** verdict (F13). | **0.825 = 0.825** |
| R3 game, `pbe` | `NOT_ESTABLISHED` | No solved class for Σ = PBE. PBE_TO_NASH is blocked because noiseless signals leave off-path records unreached. | — |
| R4 game, `nash`, ×10 | `NOT_ESTABLISHED` | Normal form built and reduced by IESDS, but support enumeration would cost ~3×10¹⁸. | **100 pure equilibria exist**: solvable, but not derived |

## Post-hoc (v0.3, corrected labels)

| Run | Status | What happened | Independent check |
|---|---|---|---|
| (pre-registered encoding) | `ILL_FORMED` | The label collision is now caught before search (F14). | — |
| P1 team, default budget | `NOT_ESTABLISHED` | budget-limited, exactly as R1 (TEAM_ENUM 3.3×10⁵ > 2×10⁵); 262 s | — |
| P2 team, ×10 | `FOUND` | 0.825, 8 classes. BEHAVIOURAL_OPT is now **UNESTABLISHED** ("bypass solver output not certified"), no longer falsely refuted. | 0.825 |
| P3 game, `pbe` | `NOT_ESTABLISHED` | as R3 | — |
| P4 game, `nash`, ×10 (no diagnosis) | `FOUND` | two derivations: NORMAL_FORM → PURE_NASH and NORMAL_FORM → IESDS → PURE_NASH. **100 equilibria, 18 classes** under realization equivalence; certified `sound`. | **100 equilibria, same 5 payoff pairs** |

## Failure classification

**No #3 (missing structure).**

- **#1** (implementation):
  - IESDS lift recomputed the elimination per solution;
  - an empty solution set crashed a V check and was counted as found.
- **#2** (semantic definition):
  - **F13**: a refutation rested on an uncertified bypass solver;
  - **F14**: the kernel did not check that a world label denotes one item.
- **#4** (interpretation/encoding): the label collision was my encoding error. It did not change any status: common information was also blocked by generated loci.
- **Registry gaps and scaling:**
  - no cheap Nash solved class (pure equilibria) in v0.2;
  - brute-force solved classes are slow (the kernel takes 376 s where the independent evaluator takes 0.3 s);
  - diagnosis re-derives expensive intermediate problems (the diagnosed P4 run was stopped after 26 min).
