# Blind forcing case — pre-registration

Written **before** encoding or running anything. The rules and registry are frozen at v0.2 (zip sha256 `247d93c7…31f5`). No rule, registry or search code will change between the runs below. Any fix goes into a later version, after the results are reported.

## Source text (the second audit's proposal, verbatim)

> A two-player coordination game where each player privately observes a signal before both simultaneously bid on a shared resource. The game is in two stages: first each can spawn one of two possible signals (at a cost), then each chooses an action based on any signal seen. The payoff depends on hidden types and joint actions.

## Interpretation choices (fixed now; each is a risk to the test's validity)

| # | Ambiguity | Choice | Alternative not taken |
|---|---|---|---|
| I1 | What each player privately observes | its own hidden type $t_i\in\{L,H\}$, iid, probability ½ each | a noisy signal of a common state |
| I2 | "spawn one of two possible signals (at a cost)" | stage-1 action $\in$ {none, sigA, sigB}. A spawned signal is a **new locus** created by Γ (it exists only if spawned), cost $c=0.05$ | a signal must always be sent |
| I3 | Who sees the signal | the **other** player, by **record route** (a message from the spawned signal locus). If there was no spawn, the reader gets ABSENT → "none" | world-mediated (noisy) signal |
| I4 | Stage-2 reads | own type + the other's signal | also own past signal. Under pure policies this is a function of own type, so it is omitted. This is **a risk for mixed strategies** |
| I5 | "Simultaneous" bids | stage-2 events read only stage-1 information, never each other's bid | — |
| I6 | Payoff ("coordination", "shared resource") | exactly one bidder → the bidder's value $v(H)=1$, $v(L)=0.4$; both bid → clash $-0.2$; nobody bids → 0; minus signal costs | — |
| I7 | Team or game | **both variants are run**. **Team:** one common payoff (the total above). **Game:** each player receives its own share (sole bidder gets $v$; a clash costs each bidder $-0.1$) minus its own signal cost | — |
| I8 | Solution concept for the game | **both are run**: `pbe` (the natural concept for sequential moves with private information) and `nash` (declared directly) | — |

## Runs (fixed now)

| Run | Variant | Search parameters |
|---|---|---|
| R1 | team | default (`max_nodes=400`, `cost_budget=2e5`) |
| R2 | team | cost budget ×10 (`2e6`) — a declared resource change, not a registry change |
| R3 | game, `pbe` | default |
| R4 | game, `nash` | cost budget ×10 |

Each run reports: status, reasons, derivations, blocked-step verdicts, and any value obtained.

**Independent check.** Any value found is re-computed by a from-scratch brute-force evaluator (`independent.py`, which does not import `tlq`), written from the case text and the table above only.

## Pre-declared failure classes

| Class | Meaning |
|---|---|
| #1 | implementation bug |
| #2 | semantic-definition bug |
| #3 | genuinely missing structure |
| #4 | interpretation error in this pre-registration |
| **Scaling** | a solved class exists but exceeds budget — recorded separately |
