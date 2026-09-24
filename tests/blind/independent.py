"""Independent evaluator for the blind case. Written from PREREGISTRATION.md alone; does NOT import tlq.

Pure policies: stage 1  type -> {none, sigA, sigB};  stage 2  (own type, other's signal seen) -> {0, 1}.
"""
import itertools
import numpy as np

T = ("L", "H")
SIG = ("none", "sigA", "sigB")
V = {"L": 0.4, "H": 1.0}
C = 0.05

S1 = list(itertools.product(range(3), repeat=2))              # stage-1 policy: index by type
S2 = list(itertools.product((0, 1), repeat=6))                 # stage-2 policy: index by (type, seen) -> 2*3 = 6 records


def rec2(t, seen):
    return T.index(t) * 3 + seen


def team_optimum():
    best, arg = -np.inf, None
    s2 = np.array(S2)                                          # (64, 6)
    for pa in S1:
        for pb in S1:
            total = np.zeros((64, 64))
            for ta in T:
                for tb in T:
                    sa, sb = pa[T.index(ta)], pb[T.index(tb)]
                    x = s2[:, rec2(ta, sb)][:, None]           # A bids, depends on B's signal
                    y = s2[:, rec2(tb, sa)][None, :]           # B bids, depends on A's signal
                    pay = np.where((x == 1) & (y == 0), V[ta], 0.0) + np.where((x == 0) & (y == 1), V[tb], 0.0) \
                        + np.where((x == 1) & (y == 1), -0.2, 0.0)
                    total += 0.25 * (pay - C * (sa != 0) - C * (sb != 0))
            m = total.max()
            if m > best + 1e-12:
                best, arg = m, (pa, pb)
    return best, arg


def game_matrices():
    """Payoff matrices for the game variant over the 9*64 pure policies of each player."""
    pols = [(p1, p2) for p1 in S1 for p2 in S2]
    n = len(pols)
    A = np.zeros((n, n)); B = np.zeros((n, n))
    s1 = np.array([p[0] for p in pols]); s2 = np.array([p[1] for p in pols])
    for ta in T:
        for tb in T:
            sa = s1[:, T.index(ta)]; sb = s1[:, T.index(tb)]
            x = s2[np.arange(n)[:, None], (T.index(ta) * 3 + sb[None, :])]      # A's bid given B's signal
            y = s2[np.arange(n)[None, :], (T.index(tb) * 3 + sa[:, None])]      # B's bid given A's signal
            A += 0.25 * (np.where((x == 1) & (y == 0), V[ta], 0.0) + np.where((x == 1) & (y == 1), -0.1, 0.0) - C * (sa[:, None] != 0))
            B += 0.25 * (np.where((y == 1) & (x == 0), V[tb], 0.0) + np.where((x == 1) & (y == 1), -0.1, 0.0) - C * (sb[None, :] != 0))
    return pols, A, B


def is_nash(A, B, p, q, tol=1e-7):
    return (A @ q).max() <= p @ A @ q + tol and (p @ B).max() <= p @ B @ q + tol


if __name__ == "__main__":
    v, arg = team_optimum()
    print("team optimum", round(v, 6), "stage-1 policies", [(T[i], SIG[a]) for i, a in enumerate(arg[0])], [(T[i], SIG[b]) for i, b in enumerate(arg[1])])
