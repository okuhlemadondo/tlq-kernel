"""The liquidation problem class and its certified transformations (round-2 chain).

Problem: sell Q0 over N steps; a_t in [0, Q_t]; revenue a_t (X_t - eta a_t); holding charge phi Q_{t+1}^2;
X_{t+1} = X_t + xi_{t+1}. The `form` field records how far the payoff has been rewritten:
  levels -> increments -> forecast -> known_forecast -> deterministic -> normalized
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np
from .kernel import Transformation, Obligation, Explicit, S, L, D, V


@dataclass
class Liquidation:
    N: int
    Q0: float
    eta: float
    phi: float
    X0: float = 50.0
    terminal: str = "forced"          # 'forced' (a_{N-1} = Q_{N-1}) | 'free'
    target: str = "expectation"       # 'expectation' | 'mean_variance'
    policies: str = "adapted"         # 'adapted' | 'anticipative'
    context: dict = field(default_factory=lambda: {"forecast": "constant", "mu": 0.0, "integrable": True})
    form: str = "levels"
    kind: str = "liquidation"


def ok(c, y, n):
    return (True, y) if c else (False, n)


def at(form):
    return lambda P: ok(getattr(P, "kind", None) == "liquidation" and P.form == form, f"payoff in {form} form",
                        f"payoff is in '{getattr(P, 'form', '?')}' form, rule needs '{form}'")


def closed_form_path(n, Q, eta, phi, mu):
    k = np.arccosh(1 + phi / (2 * eta))
    s = np.arange(n + 1)
    Qb = mu / (2 * phi) if phi > 0 else 0.0
    return Qb + (Q - Qb) * np.sinh(k * (n - s)) / np.sinh(k * n) - Qb * np.sinh(k * s) / np.sinh(k * n)


def objective(P, Q, mu):
    a = -np.diff(Q)
    return float(np.sum(mu * Q[1:P.N] - P.phi * Q[1:P.N] ** 2) - P.eta * np.sum(a ** 2))


SHIFT = Transformation(
    "SHIFT", frozenset({"sound", "complete", "full"}), at("levels"),
    lambda P: [Obligation("forced liquidation (Q_N = 0)", S, lambda P: ok(P.terminal == "forced", "a_{N-1} = Q_{N-1} forces Q_N = 0",
                                                                          "Q_N is free: the residual -Q_N X_{N-1} keeps the price level in the payoff")),
               Obligation("Abel summation by parts", L, lambda P: (True, "pathwise identity"), lemma="summation by parts")],
    lambda P: replace(P, form="increments"),
    lambda P, s: ({**s, "F": s["F"] + P.Q0 * P.X0} if "F" in s else {**s, "value_offset": P.Q0 * P.X0}))

ADAPT = Transformation(
    "ADAPT", frozenset({"sound", "complete", "full"}), at("increments"),
    lambda P: [Obligation("admissible policies are adapted", S, lambda P: ok(P.policies == "adapted", "Q_t is F_{t-1}-measurable",
                                                                             "anticipative policies: E[Q_t xi_t] != E[Q_t f_{t-1}]")),
               Obligation("target is an expectation of the payoff", S, lambda P: ok(P.target == "expectation", "risk-neutral in J",
                                                                                    f"target '{P.target}' is not linear in the law")),
               Obligation("increments integrable", D, lambda P: (P.context.get("integrable") is False, "world model says E|xi| is infinite")),
               Obligation("tower property", L, lambda P: (True, ""), lemma="E[Q_t xi_t] = E[Q_t E[xi_t | F_{t-1}]]")],
    lambda P: replace(P, form="forecast"))

CONST_FORECAST = Transformation(
    "CONST_FORECAST", frozenset({"sound", "complete", "full"}), at("forecast"),
    lambda P: [Obligation("forecast E[xi_t | F_{t-1}] is deterministic", D,
                          lambda P: (P.context.get("forecast") != "constant",
                                     f"the world model specifies a '{P.context.get('forecast')}' forecast process"))],
    lambda P: replace(P, form="known_forecast"))

OPEN_LOOP = Transformation(
    "OPEN_LOOP", frozenset({"sound", "complete"}), at("known_forecast"),
    lambda P: [Obligation("payoff contains no random symbol", S, lambda P: (True, "known forecast: payoff is a function of the Q-path")),
               Obligation("pointwise bound", L, lambda P: (True, ""), lemma="F(Q(w)) <= sup_K F for every w, deterministic paths are adapted")],
    lambda P: replace(P, form="deterministic"))

SCALE = Transformation(
    "SCALE", frozenset({"sound", "complete", "full"}), at("deterministic"),
    lambda P: [Obligation("homogeneous of degree 2 (zero drift)", S, lambda P: ok(P.context.get("mu", 0) == 0, "every term has degree 2",
                                                                                  f"drift mu={P.context.get('mu')} adds a degree-1 term")),
               Obligation("feasible set is a cone", S, lambda P: (True, "K(lambda Q0) = lambda K(Q0)"))],
    lambda P: replace(P, form="normalized", Q0=1.0),
    lambda P, s: {**s, "Q": s["Q"] * P.Q0, "F": s["F"] * P.Q0 ** 2})


def _solve_closed(P):
    mu = P.context.get("mu", 0.0) if P.form == "deterministic" else 0.0
    Q = closed_form_path(P.N, P.Q0, P.eta, P.phi, mu)
    u = 1 - Q[1:] / np.where(Q[:-1] == 0, 1, Q[:-1])
    return Explicit([{"Q": Q, "u": u, "F": objective(P, Q, mu)}])


def _constraints_inactive(P, s):
    a = -np.diff(s["Q"])
    return ok(np.all(a >= -1e-12) and np.all(s["Q"] >= -1e-12), "0 <= a_t <= Q_t along the path",
              f"inequality constraints bind (min a_t = {a.min():.3g}): the unconstrained first-order conditions are invalid")


SOLVE_RECURRENCE = Transformation(
    "SOLVE_RECURRENCE", frozenset({"sound", "complete"}),
    lambda P: ok(getattr(P, "kind", None) == "liquidation" and P.form in ("deterministic", "normalized"), "deterministic problem", "not deterministic"),
    lambda P: [Obligation("strict concavity", S, lambda P: ok(P.eta > 0 and P.phi >= 0, "eta > 0, phi >= 0", "not strictly concave")),
               Obligation("no inequality constraint active", V, _constraints_inactive)],
    _solve_closed, terminal=True)


def _solve_qp(P):
    from scipy.optimize import minimize
    mu = P.context.get("mu", 0.0)
    n = P.N - 1
    f = lambda q: -objective(P, np.r_[P.Q0, q, 0.0], mu)
    cons = [{"type": "ineq", "fun": (lambda q, i=i: (np.r_[P.Q0, q][i] - np.r_[P.Q0, q][i + 1]))} for i in range(n)] + \
           [{"type": "ineq", "fun": (lambda q: q)}]
    r = minimize(f, np.linspace(P.Q0, 0, P.N + 1)[1:-1], method="SLSQP", constraints=cons,
                 options={"ftol": 1e-12, "maxiter": 2000})
    Q = np.r_[P.Q0, r.x, 0.0]
    return Explicit([{"Q": Q, "u": 1 - Q[1:] / np.where(Q[:-1] == 0, 1, Q[:-1]), "F": -r.fun}])


SOLVE_QP = Transformation(
    "SOLVE_QP", frozenset({"sound", "complete"}), at("deterministic"),
    lambda P: [Obligation("concave objective, polyhedral feasible set", S, lambda P: ok(P.eta > 0 and P.phi >= 0, "convex QP", "not a convex QP")),
               Obligation("feasible solution", V, lambda P, s: ok(np.all(np.diff(s["Q"]) <= 1e-7) and np.all(s["Q"] >= -1e-7), "feasible", "infeasible"))],
    _solve_qp, terminal=True)


def _ce_controller(P):
    c = P.context
    m0, s0, sig = c["m0"], c["s0"], c["sigma"]

    def mean(t, disp):
        return (m0 / s0 ** 2 + disp / sig ** 2) / (1 / s0 ** 2 + t / sig ** 2)

    def ctrl(t, Q, disp):
        n = P.N - t
        if n == 1:
            return Q
        path = closed_form_path(n, Q, P.eta, P.phi, mean(t, disp))
        return Q - path[1]
    return Explicit([{"controller": ctrl, "state": ("t", "Q", "X_t - X_0"), "posterior_mean": mean}])


CE_MARTINGALE = Transformation(
    "CE_MARTINGALE", frozenset({"sound", "complete"}), at("forecast"),
    lambda P: [Obligation("Bayesian model with a specified prior", S,
                          lambda P: ok(P.context.get("forecast") == "bayes_gaussian", "conjugate Gaussian prior on the drift",
                                       f"forecast model '{P.context.get('forecast')}' has no specified prior")),
               Obligation("predictive mean is a martingale", L, lambda P: (True, ""), lemma="tower property of conditional expectation"),
               Obligation("quadratic in Q with the forecast entering linearly", S, lambda P: ok(P.eta > 0 and P.phi >= 0, "LQ structure", "not LQ")),
               Obligation("no inequality constraint binds on realized paths", D, lambda P: (False, "cannot be checked before the path is realized"))],
    _ce_controller, terminal=True)
