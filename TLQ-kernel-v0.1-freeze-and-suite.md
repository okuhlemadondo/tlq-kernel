# TLQ kernel v0.1: the freeze, the regression suite, and what the prototype found

*Follows `TLQ-round6-composite-and-kernel.md`. Claude, 24 Sep 2026. Code: `tlq-kernel-v0.1.zip` (kernel prototype + 47-test regression suite, all passing).*

---

## 1. The freeze, with your eight refinements applied

These are **interfaces for organizing the survivors, not ontological primitives** (refinement 1).

**World layer (RC).** Probes, contexts, response structure; realization and observation. TLQ uses it; it does not re-derive it.

**Execution structure** $E=(\text{sites},\ G,\ \mathscr A,\ X,\ \Gamma)$:

- **$\mathscr A(n,m)$**: admissible record transformations between sites, with identities and composition. Unchanged (refinement 3). $\mathscr A(n,n)$ is the site's memory.
- **$\Gamma$**: the *generation rule*. It says which evaluations, sites, arrows and owners can come into being as functions of realized history (refinement 2).
- **$X$**: the *execution semantics*: how a run resolves (a well-founded order, or a fixed point with a certificate).
- **The unfolding** is not a primitive. It is what applying $\Gamma$ under $X$ produces.
- **New from the prototype: sites vs evaluations** (§3, F1).

**Task** $T$: objectives on histories, ownership, solution concept.

**Problem** $P=(W,E,T)$: a structural interface, **with dependencies allowed**, especially $T\leftrightarrow E$ (refinement 5).

**Record route ≠ world-mediated route.** No universal factorization into "information + effect" (refinement 4). The prototype split what used to be one reduction into two, each with its own conditions (F2).

**Transformation** $\rho=(f,\ell,\Phi,\text{obligations})$:

- **Semantic legitimacy and executability are separate.** The code carries an explicit `executable` flag (refinement 6).
- **The fundamental object is a preservation obligation under stated assumptions** (refinement 7). S/L/D/V is the *justification mode*:
  - S: syntactic, checked on the specification;
  - L: lemma;
  - D: declared assumption, entered in the ledger unless the world model contradicts it;
  - V: checked a posteriori.
- **$\Phi$ uses only sound / complete / full**, which compose by intersection. Quantitative slack is left out of the general form.
- **Confluence is a property of rule systems.** It is tested by exploring every order of application, not declared per transformation.

**Derivations are DAGs** (refinement 8): linear steps, fan-out with recombination, terminal solvers. A solver is a transformation into an explicitly presented solution set.

---

## 2. The regression suite

Nine cases, 47 tests. For each forcing case the suite encodes:

- the problem (builders contain **no solutions**);
- the derivation;
- the expected transformation sequence;
- certificate kinds and statuses;
- the ledger;
- the expected solution, against an **independent oracle** where one exists (BFGS, exact DP, brute force, Monte Carlo);
- **at least one invalid derivation**, with the obligation the kernel must name when it rejects it.

| Case | Reproduced through the kernel's own derivations | Rejections the kernel must make |
|---|---|---|
| C1 sinh schedule | path, $u_0$, 820.86, Monte Carlo | 7 (free terminal, wrong order, mean–variance, anticipative, drift under SCALE, integrability contradicted, binding constraints → QP route) |
| C2 unknown drift | state $(t,Q,X_t-X_0)$; actions match exact DP to $10^{-10}$ | 2 |
| C3 13/20 | brute force **and** common information → coordinator → lifted distributed controller; beliefs 3/4, 1/2, 1/4 | 2 |
| C4 driver | 4/3 at $p=2/3$; 1; 4 | 2, plus a semantic regression |
| C5 silence | 0.75 / 0.70 / 0.50 | 1 |
| C6 endogenous absence | 3/4; certified 0.7 vs **uncertified 0.6** | 1, plus an accepted variant |
| C7 non-confluence | ≥2 normal forms, (2,1) vs (1,1); strict elimination confluent | 2 |
| C8 round 6 | baseline and expensive-decoy equilibria exact; world-leak multiplicity (the kernel refuses to select) | 3 |
| C9 routes and branching | exact route swap; pure-message equivalence; branch bookkeeping | 5 |

Where the code lives:

- **Kernel core** (`kernel.py`): no case knowledge.
- **Rule files:** general over their problem classes. A grep for the expected values across `tlq/` finds none.

---

## 3. What the prototype found (the adversary phase has begun)

Every failure was classified with your scheme. **No finding is #3 (genuinely missing structure).** Four are #2 (definitional), one is #1 (implementation). Two more observations sharpen earlier claims.

### F1 — Sites are not evaluations (#2, load-bearing)

"Locus" was doing two jobs:

- a **site**: a program with a record store;
- an **evaluation**: one execution of that site.

Rounds 3–4 used time-indexed loci, which silently treats every evaluation as its own site. The absent-minded driver cannot be run without choosing, and the choice changes the answer:

- one site evaluated twice, forgetful, fresh coin: **4/3**;
- the same driver encoded as two sites: **4**, i.e. perfect recall granted by the encoding.

**Fix:**

- policies belong to *sites* and read the site's record;
- $\Gamma$ generates *evaluations*;
- $\mathscr A(n,n)$ is the site's memory;
- perfect recall = $\mathscr A(n,n)$ carries the site's history *including a visit counter*.

Your round-4 phrasing — "a site at which a record transformation is evaluated" — already contained the distinction. It is a split of an existing notion, not a new primitive. The suite now guards it with a regression test.

### F2 — "WORLD ⇒ RECORD" was two transformations (#2)

The prototype showed that the separability condition did no work for the rule as implemented. The rule has to split in two:

- **OBS_AS_RECORD**: re-express a world observation as a record transfer. It is exact whenever the observation is noise-free, has a single source and has no other observer, **even with interacting side effects**, because the side effect stays in the payoff either way.
- **WORLD_CHANNEL_AS_MESSAGE**: the stronger claim that the channel *is* a pure message whose only world effect is a cost. This one needs separability. When separability fails, the values really do differ.

Rounds 4–6 attached separability to the wrong claim.

### F3 — Round 6's "one arrow" sentence was wrong (#2, my modelling error)

Round 6 said the leaky dark pool meant "B gains an arrow from D", but it *modelled* the leak as a noisy dip: a world route. Encoded literally as a record arrow, the kernel **refuses** PBE ⇒ Nash:

- a noiseless arrow leaves some of B's records unreached after some histories;
- so off-path beliefs are unconstrained;
- so the full-support certificate fails.

**Consequences:**

- The round-6 table row is correct for a *world-route* leak.
- The claim that one admissible arrow changes the equilibrium structure still stands. The arrow does something stronger: it removes the certificate that made Nash sufficient.
- Solving the arrow version needs a genuine PBE solver, which the prototype does not have. That is a solved-class gap, not a kernel gap.

### F4 — Solutions need an identity (#2)

The baseline equilibrium comes back as **four** distinct mixed normal-form profiles that induce identical behaviour. The world-leak case has *genuine* multiplicity: different payoffs to A. So "how many solutions" is meaningless without an equivalence on solutions. Here the right one is realization-equivalence.

**Candidate:** the solution concept Σ carries its solution equivalence. Currently the tests quotient behaviourally; the kernel should.

### F5 — Lifts are typed by solution form (#1)

The SHIFT lift assumed a path-valued solution, but CE_MARTINGALE returns a controller with a residual solver. So "the output is not fixed to controllers" (your point 7) has an implementation consequence: a lift must be defined for every solution form it can receive.

### F6 — Certificates are sufficient, not necessary (observation)

The kernel is conservative:

- FAN_OUT drops completeness when branches don't provably cover the policy class, even in an instance where no value is lost (0.775 = 0.775);
- rejected transformations can happen to be value-preserving.

This is the intended direction of error, and it should be stated as a property of the kernel.

### F7 — The endogenous-absence criterion made precise (observation)

Round 5 said absence is informative when it is *endogenous*. The executable check is sharper: **the null reading must leave the task-relevant state at its prior under every admissible program.** In the accepted C6 variant, $\Gamma$ still depends on an action (the spawn is triggered by P's coin-driven choice), yet absence is uninformative about $x$, and the kernel correctly accepts it. Structural endogeneity is not the criterion; informativeness about the task is.

**Also corrected:** round 5's 3/4 vs 5/8 relied on random tie-breaking. The suite uses an asymmetric prior so the gap is tie-free: 0.7 vs 0.6.

---

## 4. Honest limits of v0.1

- **Derivations are scripted, not discovered.** The kernel checks, composes, lifts and rejects. It does not search.
- **Rule classes are narrow:**
  - liquidation is a symbolic class;
  - common information handles fixed events with one evaluation per site;
  - PBE is reached only through full-support noise;
  - behavioural optimization is certified only by multistart agreement.
- **D obligations are contradicted only by explicit world-model fields.**

## 5. Next adversary

1. **Derivation search.** Can the kernel *find* each case's derivation from the rule library, and does it ever find a *different* valid one? A second valid derivation is a strong test of confluence and ledger bookkeeping.
2. **Add solution equivalence to Σ** (F4), then re-run C8.
3. **A PBE solver** for the record-arrow leak (F3).
4. **Grow one rule class at a time,** adding each new forcing case to the suite *before* writing its rule.
