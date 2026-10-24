"""Sections 2, 5, and 6: Vertical Rewrite Chain, Bounded-Shock Resizing Model, and Theorem 2.4.

Implements:
- ResizingProblem (Definition 2.3: P_resize^{bdd})
- Theorem 2.4(2): constrained_action_gap_bound, exact_log_objective_gap_bounds, solve_constrained_log_optimum
- Ruin-boundary diagnostics: truncated_log_compare
- Universal Epistemic Control Loss (Theorem 6.1 & Corollary 6.2)
- Kernel Rewrite Transformations: MTM_SHIFT, ADAPT_INCREMENTS, CUMULANT_QUAD, HAMILTONIAN_COMPLETE, SOLVE_KELLY_CLAMPED
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import math
from typing import Any
from .kernel import Transformation, Obligation, Explicit, S, L, D, V


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def kelly_target(wealth: float, drift: float, variance: float) -> float:
    return (wealth * drift) / variance


def kelly_fraction(drift: float, variance: float) -> float:
    return drift / variance


def log_curvature(variance: float, wealth: float) -> float:
    return variance / (2.0 * wealth * wealth)


def control_loss(variance: float, bias: float, estimator_variance: float) -> float:
    return 0.5 * variance * (bias * bias + estimator_variance)


def null_loss(drift: float, variance: float) -> float:
    theta = drift / variance
    return 0.5 * variance * theta * theta


def ema_theta_variance(alpha: float, variance: float) -> float:
    return alpha / ((2.0 - alpha) * variance)


def ema_sharpe_hurdle(alpha: float) -> float:
    return alpha / (2.0 - alpha)


def squared_sharpe(drift: float, variance: float) -> float:
    return (drift * drift) / variance


def constrained_action_gap_bound(
    lambda_max: float,
    third_abs_moment: float,
    sigma2: float,
    eta_margin: float,
) -> float:
    """Theorem 2.4(2): |theta_quad - theta_log| <= (Lambda_max^2 * E[|xi|^3 | F_t]) / (sigma_t^2 * eta_margin)."""
    return (lambda_max * lambda_max * third_abs_moment) / (sigma2 * eta_margin)


def exact_log_objective_gap_bounds(
    lambda_max: float,
    third_abs_moment: float,
    sigma2: float,
    eta_margin: float,
) -> dict[str, float]:
    """Theorem 2.4(2): Self-contained exact-log objective suboptimality bounds."""
    grad_diff = (lambda_max * lambda_max * third_abs_moment) / eta_margin
    variational_bound = (grad_diff * grad_diff) / sigma2
    taylor_bound = (2.0 * (lambda_max ** 3) * third_abs_moment) / (3.0 * (eta_margin ** 3))
    return {
        "variational_bound": variational_bound,
        "taylor_bound": taylor_bound,
        "combined_bound": min(variational_bound, taylor_bound),
    }


def log_obj(theta: float, xs: list[float], ps: list[float]) -> float:
    return sum(p * math.log(1.0 + theta * x) for x, p in zip(xs, ps))


def log_grad(theta: float, xs: list[float], ps: list[float]) -> float:
    return sum((p * x) / (1.0 + theta * x) for x, p in zip(xs, ps))


def solve_constrained_log_optimum(
    xs: list[float],
    ps: list[float],
    lambda_max: float,
    n_iter: int = 120,
) -> float:
    """Finds the exact constrained maximizer of E[ln(1 + theta * xi)] on [-lambda_max, +lambda_max]."""
    g_lo = log_grad(-lambda_max, xs, ps)
    g_hi = log_grad(+lambda_max, xs, ps)
    if g_lo <= 0.0:
        return -lambda_max
    if g_hi >= 0.0:
        return +lambda_max
    lo, hi = -lambda_max, +lambda_max
    flo = g_lo
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fm = log_grad(mid, xs, ps)
        if abs(fm) < 1e-14:
            return mid
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def truncated_log_compare(
    drift: float,
    sigma: float,
    near_ruin_threshold: float = 1e-8,
) -> dict[str, Any]:
    """Compare quadratic log-growth proxy with E[ln(1 + theta * xi)] on [f - 3*sigma, f + 3*sigma].
    Enforces exact -inf whenever 1 + theta * x <= 0 on any positive-weight node.
    """
    variance = sigma * sigma
    theta_star = drift / variance
    lo = drift - 3.0 * sigma
    hi = drift + 3.0 * sigma
    n = 241
    dx = (hi - lo) / (n - 1)
    zs: list[tuple[float, float]] = []
    mass = 0.0
    for i in range(n):
        x = lo + i * dx
        z = (x - drift) / sigma
        dens = math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
        w = dens * dx
        zs.append((x, w))
        mass += w

    def eval_theta(theta: float) -> tuple[float, float, float]:
        if mass <= 0.0:
            return (float("nan"), 1.0, 0.0)
        acc = 0.0
        ruin_weight = 0.0
        near_ruin_weight = 0.0
        for x, w in zs:
            arg = 1.0 + theta * x
            if arg <= 0.0:
                ruin_weight += w
            else:
                if arg <= near_ruin_threshold:
                    near_ruin_weight += w
                acc += w * math.log(arg)
        ruin_prob = ruin_weight / mass
        near_ruin_prob = near_ruin_weight / mass
        exact_val = float("-inf") if ruin_weight > 0.0 else acc / mass
        return (exact_val, ruin_prob, near_ruin_prob)

    theta_grid: list[float] = []
    quadratic: list[float] = []
    exact: list[float] = []
    ruin_mass_grid: list[float] = []
    near_ruin_mass_grid: list[float] = []
    lo_t = min(-0.2, theta_star - 0.55)
    hi_t = max(0.8, theta_star + 0.55)
    m = 81
    for j in range(m):
        theta = lo_t + ((hi_t - lo_t) * j) / (m - 1)
        theta_grid.append(theta)
        quadratic.append(theta * drift - 0.5 * theta * theta * variance)
        ex_val, r_prob, nr_prob = eval_theta(theta)
        exact.append(ex_val)
        ruin_mass_grid.append(r_prob)
        near_ruin_mass_grid.append(nr_prob)

    star_ex, star_ruin, star_near = eval_theta(theta_star)
    ruin_boundary_lo = -1.0 / hi if hi > 0.0 else float("-inf")
    ruin_boundary_hi = -1.0 / lo if lo < 0.0 else float("inf")

    return {
        "theta_grid": theta_grid,
        "quadratic": quadratic,
        "exact": exact,
        "ruin_mass_grid": ruin_mass_grid,
        "near_ruin_mass_grid": near_ruin_mass_grid,
        "theta_star": theta_star,
        "quad_at_star": theta_star * drift - 0.5 * theta_star * theta_star * variance,
        "exact_at_star": star_ex,
        "ruin_mass": star_ruin,
        "near_ruin_mass": star_near,
        "ruin_boundary_lo": ruin_boundary_lo,
        "ruin_boundary_hi": ruin_boundary_hi,
        "near_ruin_threshold": near_ruin_threshold,
    }


# ============================================================================
# Kernel Problem Class & Three-Stage Vertical Rewrite Morphisms (Section 5)
# ============================================================================

@dataclass
class ResizingProblem:
    """Bounded-shock frictionless inventory resizing problem P_resize^{bdd} (Definition 2.3 & Section 5)."""
    N: int = 10
    W0: float = 100.0
    Q0: float = 0.0
    P0: float = 50.0
    f_bar: float = 0.12
    sigma2_bar: float = 0.25
    sigma_min2: float = 0.05
    xi_max: float = 0.5
    eta_margin: float = 0.2
    third_abs_moment: float = 0.08
    terminal_potential: str = "mtm"         # 'mtm' (Psi = Q_N P_N) | 'none'
    policies: str = "adapted"               # 'adapted' | 'anticipative'
    target: str = "log_wealth"              # 'log_wealth' | 'linear_wealth'
    allow_surrogate_slack: bool = True      # whether Stage 2b third-moment slack is permitted
    integrable: bool = True
    form: str = "raw"                       # 'raw' -> 'adv' -> 'cond' -> 'hamiltonian'
    kind: str = "resizing"

    @property
    def lambda_max(self) -> float:
        return (1.0 - self.eta_margin) / self.xi_max

    def equivalent(self, s1: dict[str, Any], s2: dict[str, Any]) -> bool:
        return abs(s1.get("theta_star", 0.0) - s2.get("theta_star", 0.0)) <= 1e-9


def _ok(cond: bool, yes: str, no: str) -> tuple[bool, str]:
    return (True, yes) if cond else (False, no)


def _at_form(form: str):
    return lambda P: _ok(
        getattr(P, "kind", None) == "resizing" and P.form == form,
        f"problem in '{form}' form",
        f"problem is in '{getattr(P, 'form', '?')}' form, rule requires '{form}'",
    )


MTM_SHIFT = Transformation(
    name="MTM_SHIFT",
    preserves=frozenset({"sound", "complete", "full"}),
    applies=_at_form("raw"),
    obligations=lambda P: [
        Obligation(
            "mark-to-market terminal potential (Psi(W_N, Q_N) = Q_N P_N)",
            S,
            lambda P: _ok(
                P.terminal_potential == "mtm",
                "Phi(W_t, Q_t) = Q_t P_t matches terminal Psi(W_N, Q_N) = Q_N P_N for all free Q_N in R",
                "terminal potential does not match mark-to-market Q_N P_N: boundary residual eps_1 != 0",
            ),
        ),
        Obligation(
            "telescoping potential identity (Theorem 5.2(1))",
            L,
            lambda P: (True, "exact pathwise telescoping sum with eps_1 = 0"),
            lemma="Theorem 5.2(1)",
        ),
    ],
    forward=lambda P: replace(P, form="adv"),
    lift=lambda P, s: {**s, "potential_offset": P.Q0 * P.P0},
    slack=lambda P: 0.0,
)

ADAPT_INCREMENTS = Transformation(
    name="ADAPT_INCREMENTS",
    preserves=frozenset({"sound", "complete", "full"}),
    applies=_at_form("adv"),
    obligations=lambda P: [
        Obligation(
            "admissible policies are F_t-adapted (Axiom 1)",
            S,
            lambda P: _ok(
                P.policies == "adapted",
                "u_t = Q_{t+1} is strictly F_t-measurable",
                "anticipative policies violate Axiom 1: E[u_t xi_{t+1} | F_t] != u_t E[xi_{t+1} | F_t]",
            ),
        ),
        Obligation(
            "shocks have finite third absolute moment",
            D,
            lambda P: (not P.integrable, "world model declares shocks non-integrable"),
        ),
        Obligation(
            "tower property of conditional expectation (Theorem 5.2(2a))",
            L,
            lambda P: (True, "L^2 martingale difference orthogonality"),
            lemma="Theorem 5.2(2a)",
        ),
    ],
    forward=lambda P: replace(P, form="projected"),
    slack=lambda P: 0.0,
)

CUMULANT_QUAD = Transformation(
    name="CUMULANT_QUAD",
    preserves=frozenset({"sound", "complete", "full"}),
    applies=_at_form("projected"),
    obligations=lambda P: [
        Obligation(
            "strict survival margin (1 - Lambda_max * xi_max >= eta_margin > 0)",
            S,
            lambda P: _ok(
                0.0 < P.eta_margin < 1.0 and P.xi_max > 0.0,
                f"eta_margin={P.eta_margin:.3f} in (0, 1) guarantees 1 + theta*xi >= eta_margin > 0",
                f"invalid survival margin eta_margin={P.eta_margin} or xi_max={P.xi_max}",
            ),
        ),
        Obligation(
            "positive conditional variance floor (sigma2_bar >= sigma_min2 > 0)",
            S,
            lambda P: _ok(
                P.sigma2_bar >= P.sigma_min2 > 0.0,
                f"sigma2_bar={P.sigma2_bar:.4f} >= sigma_min2={P.sigma_min2:.4f}",
                f"sigma2_bar={P.sigma2_bar:.4f} falls below sigma_min2={P.sigma_min2:.4f}",
            ),
        ),
        Obligation(
            "exact zero-slack preservation requires linear_wealth or declared third-moment surrogate slack (Theorem 2.4)",
            S,
            lambda P: _ok(
                P.target == "linear_wealth" or P.allow_surrogate_slack,
                "third-moment surrogate slack bound enabled via Theorem 2.4(2)",
                "Theorem 2.4(1) counterexample refutes exact zero-slack argmax equality between log_wealth and quadratic surrogate",
            ),
            required=False,
            for_property="full",
        ),
        Obligation(
            "bounded shock support |xi_{t+1}| <= xi_max a.s. (Definition 2.3)",
            D,
            lambda P: (P.xi_max <= 0.0 or not math.isfinite(P.xi_max), "unbounded shock support violates Definition 2.3(2)"),
        ),
        Obligation(
            "Lagrange third-order Taylor remainder & variational action-gap bound (Theorem 2.4(2) & Theorem 5.2(2b))",
            L,
            lambda P: (True, "eps_2b <= (Lambda_max^3 * E[|xi|^3]) / (3 * eta_margin^3)"),
            lemma="Theorem 2.4(2) & Theorem 5.2(2b)",
        ),
    ],
    forward=lambda P: replace(P, form="cond"),
    slack=lambda P: (
        0.0
        if P.target == "linear_wealth"
        else ((P.lambda_max ** 3) * P.third_abs_moment) / (3.0 * (P.eta_margin ** 3))
    ),
)

HAMILTONIAN_COMPLETE = Transformation(
    name="HAMILTONIAN_COMPLETE",
    preserves=frozenset({"sound", "complete", "full"}),
    applies=_at_form("cond"),
    obligations=lambda P: [
        Obligation(
            "strictly positive-definite scalar curvature phi_t = sigma2_bar / (2 W_t^2) > 0 (Axiom 2)",
            S,
            lambda P: _ok(P.W0 > 0.0 and P.sigma2_bar > 0.0, "phi_t > 0", "curvature is non-positive"),
        ),
        Obligation(
            "exact completion of the square & Moreau orthogonal projection (Lemma 2.2 & Theorem 5.2(3))",
            L,
            lambda P: (True, "Q_t^{quad}(theta) = Q_t^{quad}(theta_unc*) - (sigma^2/2)(theta - theta_unc*)^2"),
            lemma="Lemma 2.2 & Theorem 5.2(3)",
        ),
    ],
    forward=lambda P: replace(P, form="hamiltonian"),
    slack=lambda P: 0.0,
)


def _solve_kelly_clamped(P: ResizingProblem) -> Explicit:
    theta_unc = P.f_bar / P.sigma2_bar
    theta_star = clamp(theta_unc, -P.lambda_max, P.lambda_max)
    q_star = P.W0 * theta_star
    act_gap_bound = constrained_action_gap_bound(P.lambda_max, P.third_abs_moment, P.sigma2_bar, P.eta_margin)
    obj_bounds = exact_log_objective_gap_bounds(P.lambda_max, P.third_abs_moment, P.sigma2_bar, P.eta_margin)
    return Explicit([
        {
            "theta_unc": theta_unc,
            "theta_star": theta_star,
            "Q_next_star": q_star,
            "action_gap_bound": act_gap_bound,
            "objective_gap_bound": obj_bounds["combined_bound"],
            "variational_obj_bound": obj_bounds["variational_bound"],
            "taylor_obj_bound": obj_bounds["taylor_bound"],
        }
    ])


def _verify_clamped_solution(P: ResizingProblem, sol: dict[str, Any]) -> tuple[bool, str]:
    th = sol["theta_star"]
    th_unc = sol["theta_unc"]
    if abs(th) > P.lambda_max + 1e-12:
        return (False, f"exposure {th} exceeds leverage cap {P.lambda_max}")
    # First-order projection variational inequality: (th_unc - th) * (v - th) <= 0 for all v in [-Lambda, Lambda]
    for v in (-P.lambda_max, 0.0, +P.lambda_max):
        if (th_unc - th) * (v - th) > 1e-12:
            return (False, f"projection variational inequality violated at test point v={v}")
    return (True, "clamped exposure satisfies projection variational inequality and survival barrier")


SOLVE_KELLY_CLAMPED = Transformation(
    name="SOLVE_KELLY_CLAMPED",
    preserves=frozenset({"sound", "complete", "full"}),
    applies=_at_form("hamiltonian"),
    obligations=lambda P: [
        Obligation(
            "compact convex admissible exposure interval [-Lambda_max, +Lambda_max]",
            S,
            lambda P: _ok(P.lambda_max > 0.0, f"Lambda_max={P.lambda_max:.4f} > 0", "empty or degenerate exposure interval"),
        ),
        Obligation(
            "clamped exposure satisfies Moreau projection variational inequality",
            V,
            _verify_clamped_solution,
        ),
    ],
    forward=_solve_kelly_clamped,
    terminal=True,
    slack=lambda P: 0.0,
)

VERTICAL_CHAIN = (MTM_SHIFT, ADAPT_INCREMENTS, CUMULANT_QUAD, HAMILTONIAN_COMPLETE, SOLVE_KELLY_CLAMPED)
