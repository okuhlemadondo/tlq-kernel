# TLQ Kernel (`v2.2.2-RC`)

**Certified Representation Routing for Stochastic Control**

This repository contains the reference Python kernel, formal specification ([`SPECIFICATION.md`](./SPECIFICATION.md)), and adversarial falsification & obligation regression suite for **TLQ (`v2.2.2-RC`)**.

> **Historical Note:** The earlier `v0.1–v0.3` finite-game / team-signaling prototype is preserved unchanged on the `archive/v0.3-legacy` branch (`git checkout archive/v0.3-legacy`).

## Architecture Overview

| Module | Specification Scope | Core Responsibilities |
| :--- | :--- | :--- |
| [`SPECIFICATION.md`](./SPECIFICATION.md) | §§0–12 (`v2.2.2-RC`) | Complete mathematical formalisation, theorem proofs, claim status ledger (`PROVED`, `PROVED UNDER ASSUMPTIONS`, `REFUTED`, `OPEN`, `CONJECTURE`, `EMPIRICAL`), and finite-progress release contract (Packages A–D). |
| [`tlq/kernel.py`](./tlq/kernel.py) | Obligation & Rewrite Spine | Generic obligation verification (`S` syntactic, `L` lemma, `D` declared assumption in ledger, `V` verified a-posteriori), problem rewrite morphisms with validated non-negative finite slack ($\varepsilon_\rho \ge 0$), structural `apply`/`fan` state guards, and named rejection diagnostics (`Rejected`). |
| [`tlq/hypergraph.py`](./tlq/hypergraph.py) | §3 (`Prop. 3.3`) | Typed stream signature $\Sigma = (\mathcal{S}, \mathcal{O})$ with full `Sort` contract equality (`name`, `domain_spec`, and `domain_check`), Colored Copy-Discard PROP $\mathbf{CD}(\Sigma)$, port-grafting composition $\circ_i$, Cartesian DAG evaluation, and universal shift-register overwrite impossibility validator on $\mathcal{M}_{\mathrm{reg}}(\mathrm{add}, \mathrm{mul})$. |
| [`tlq/certificates.py`](./tlq/certificates.py) | §7 (`Def. 7.1`, `Lemma 7.1a`, `Prop. 7.2`) | Multi-stratum epistemic certificate algebra $\mathcal{K} = \{\mathtt{EXACT}, \mathtt{MOMENT}(\varepsilon, \mathbf{V}), \mathtt{CONF\_RADIUS}(r, \delta), \mathtt{EVENT}(r, \delta, M_{\max}), \mathtt{UNESTABLISHED}(\text{code})\}$, compact input moment box $D_{\mathrm{bdd}} = [-\xi_{\max}, \xi_{\max}] \times [\sigma_{\min}^2, \xi_{\max}^2]$ projection and diameter $M_{\mathrm{in},\max}$, repaired $(1/2)$-net sub-Gaussian tail radius $r_\delta^{\mathrm{subG}}$, and two-stage `KellyTarget` pushforward (unclamped ratio `CONF_RADIUS` $\to$ projected action `EVENT`). |
| [`tlq/vertical.py`](./tlq/vertical.py) | §§2, 5, 6 (`Lemma 2.2`, `Def. 2.3`, `Thm. 2.4`, `Thm. 5.2`, `Thm. 6.1`, `Cor. 6.2`) | Bounded-support frictionless inventory resizing model $\mathfrak{P}_{\mathrm{resize}}^{\mathrm{bdd}}$, the 3-stage vertical derivation chain (`MTM_SHIFT` $\to$ `ADAPT_INCREMENTS` $\to$ `CUMULANT_QUAD` $\to$ `HAMILTONIAN_COMPLETE` $\to$ `SOLVE_KELLY_CLAMPED`), constrained variational action-distance and objective-gap bounds, and ruin-boundary log-growth diagnostics. |
| [`tlq/router.py`](./tlq/router.py) | §§8, 9 (`Thms. 8.4a–d`, `Prop. 8.4e`, `Thm. 9.1`) | Context-dependent horizontal hypergraph router $G_t \Rightarrow_{\mathcal{C}_t} G_{t+1}$ with Guard 1 immediate epistemic collapse to $G_{\mathrm{null}}$, Guard 2 hysteresis + symmetric dwell reset, all-pairs Azuma concentration $\kappa_{\mathrm{search}}$, self-normalized predictable variation $\kappa_{\mathrm{pred}}$ via Gaussian-mixture supermartingales, post-switch regime-shift drift $D_{\mathrm{regime}}(t_j)$, and 4-stage end-to-end vertical-to-horizontal composition (`compose_end_to_end_certificate`). |

## Running the Falsification & Regression Suite

Run the complete test suite (22 tests across `tests/test_v222_*.py`) and verify file integrity with:

```bash
python3 run_suite.py
shasum -a 256 -c MANIFEST.sha256
```

`run_suite.py` uses `pytest` when installed or falls back automatically to the built-in zero-dependency runner in `shim/pytest.py`.
