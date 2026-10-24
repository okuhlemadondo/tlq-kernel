# TLQ Kernel (`v2.2.2-RC`)

**Certified Representation Routing for Stochastic Control**

This repository contains the reference Python kernel, formal specification ([`SPECIFICATION.md`](./SPECIFICATION.md)), and adversarial falsification & obligation regression suite for **TLQ (`v2.2.2-RC`)**.

> **Historical Note:** The earlier `v0.1–v0.3` finite-game / team-signaling prototype is preserved unchanged on the `archive/v0.3-legacy` branch (`git checkout archive/v0.3-legacy`).

## Architecture Overview

| Module | Specification Scope | Core Responsibilities |
| :--- | :--- | :--- |
| [`SPECIFICATION.md`](./SPECIFICATION.md) | §§0–12 (`v2.2.2-RC`) | Complete mathematical formalisation, theorem proofs, claim status ledger (`PROVED`, `PROVED UNDER ASSUMPTIONS`, `REFUTED`, `OPEN`, `CONJECTURE`, `EMPIRICAL`), and finite-progress release contract (Packages A–D). |
| [`tlq/kernel.py`](./tlq/kernel.py) | Obligation & Rewrite Spine | Generic obligation verification (`S` syntactic, `L` lemma, `D` declared assumption in ledger, `V` verified a-posteriori), problem rewrite morphisms with quantitative slack tracking, and named rejection diagnostics (`Rejected`). |
| [`tlq/hypergraph.py`](./tlq/hypergraph.py) | §3 (`Prop. 3.3`) | Typed stream signature $\Sigma = (\mathcal{S}, \mathcal{O})$, Colored Copy-Discard PROP $\mathbf{CD}(\Sigma)$, port-grafting composition $\circ_i$, Cartesian DAG evaluation, and universal shift-register overwrite impossibility validator on $\mathcal{M}_{\mathrm{reg}}(\mathrm{add}, \mathrm{mul})$. |
| [`tlq/certificates.py`](./tlq/certificates.py) | §7 (`Def. 7.1`, `Lemma 7.1a`, `Prop. 7.2`) | Quantitative 3-stratum epistemic certificate algebra $\mathcal{K} = \{\mathtt{EXACT}, \mathtt{MOMENT}(\varepsilon, \mathbf{V}), \mathtt{EVENT}(r, \delta, M_{\max}), \mathtt{UNESTABLISHED}(\text{code})\}$, repaired $(1/2)$-net sub-Gaussian tail radius $r_\delta^{\mathrm{subG}}$, fourth-moment Cauchy–Schwarz pushforwards, and guarded `KellyTarget` nonlinear pushforward. |
| [`tlq/vertical.py`](./tlq/vertical.py) | §§2, 5, 6 (`Lemma 2.2`, `Def. 2.3`, `Thm. 2.4`, `Thm. 5.2`, `Thm. 6.1`, `Cor. 6.2`) | Bounded-support frictionless inventory resizing model $\mathfrak{P}_{\mathrm{resize}}^{\mathrm{bdd}}$, the 3-stage vertical derivation chain (`MTM_SHIFT` $\to$ `ADAPT` $\to$ `CUMULANT_QUAD` $\to$ `HAMILTONIAN_COMPLETE`), constrained variational action-distance and objective-gap bounds, and ruin-boundary log-growth diagnostics. |
| [`tlq/router.py`](./tlq/router.py) | §§8, 9 (`Thms. 8.4a–d`, `Prop. 8.4e`, `Thm. 9.1`) | Context-dependent horizontal hypergraph router $G_t \Rightarrow_{\mathcal{C}_t} G_{t+1}$ with Guard 1 immediate epistemic collapse to $G_{\mathrm{null}}$, Guard 2 hysteresis + symmetric dwell reset, all-pairs Azuma concentration $\kappa_{\mathrm{search}}$, self-normalized predictable variation $\kappa_{\mathrm{pred}}$ via Gaussian-mixture supermartingales, post-switch regime-shift drift $D_{\mathrm{regime}}(t_j)$, and end-to-end vertical-to-horizontal composition. |

## Running the Falsification & Regression Suite

Run the complete test suite with:

```bash
python3 run_suite.py
```

`run_suite.py` uses `pytest` when installed or falls back automatically to the built-in zero-dependency runner in `shim/pytest.py`.
