"""Section 7: Epistemic Certificate Algebra (K) — Definition 7.1, Lemma 7.1a, and Proposition 7.2.

Implements:
- ExactCert, MomentCert, ConfidenceRadiusCert (high-probability ball without an a.s. diameter claim),
  EventCert (high-probability ball + verified compact-domain almost-sure bound M_max < inf), UnestablishedCert
- Compact input moment box D_bdd = [-xi_max, xi_max] x [sigma_min2, xi_max^2] projection & diameter M_{in,max}
- Lemma 7.1a(1): Repaired (1/2)-net sub-Gaussian Euclidean norm concentration radius r_delta^{subG}
- Lemma 7.1a(2): Expected curvature-weighted squared control-error bound (`event_to_quadratic_regret_bound`)
  and exact constrained quadratic objective-regret bound (`event_to_constrained_quadratic_objective_regret_bound`)
- Proposition 7.2(2): Two-stage ratio pushforward:
    Stage A (`kelly_ratio_certificate_pushforward`): raw ratio f/v -> CONF_RADIUS(r_raw, delta)
    Stage B (`project_kelly_exposure_certificate`): clamping Pi_{[-Lambda_max, Lambda_max]}(f/v) -> EVENT(min(r_raw, 2*Lambda_max), delta, 2*Lambda_max)
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Union


@dataclass(frozen=True)
class ExactCert:
    status: str = "EXACT"


@dataclass(frozen=True)
class MomentCert:
    eps: float
    variance: float
    nu_subg: float = 0.0
    dimension: int = 1
    status: str = "MOMENT"


@dataclass(frozen=True)
class ConfidenceRadiusCert:
    """High-probability error radius P(||z_hat - z*||_2 <= r) >= 1 - delta without an almost-sure diameter claim."""
    r: float
    delta: float
    L_hat_o: float = 1.0
    M_hat_Ho: float = 0.0
    status: str = "CONF_RADIUS"


@dataclass(frozen=True)
class EventCert:
    """Bounded-domain high-probability certificate: P(||z_hat - z*||_2 <= r) >= 1 - delta AND ||z_hat - z*||_2 <= m_max a.s."""
    r: float
    delta: float
    m_max: float
    L_hat_o: float = 1.0
    M_hat_Ho: float = 0.0
    status: str = "EVENT"


@dataclass(frozen=True)
class UnestablishedCert:
    code: str = "DOMAIN_VIOLATION"
    status: str = "UNESTABLISHED"


Certificate = Union[ExactCert, MomentCert, ConfidenceRadiusCert, EventCert, UnestablishedCert]


def sub_gaussian_event_radius(
    eps: float,
    dimension: int,
    nu: float,
    delta: float,
) -> dict[str, float]:
    """Lemma 7.1a(1) (v2.2.2-RC Repair of Finding A):
    Valid (1/2)-net sub-Gaussian Euclidean norm concentration radius in R^d:
      r_delta^{subG}(eps, d, nu) = eps + nu * sqrt(8 * (d * ln 5 + ln(1 / delta)))
    and its subadditive upper bound:
      eps + sqrt(8 * ln 5) * nu * sqrt(d) + sqrt(8) * nu * sqrt(ln(1 / delta)).
    """
    exact_net_radius = eps + nu * math.sqrt(8.0 * (dimension * math.log(5.0) + math.log(1.0 / delta)))
    additive_upper_bound = (
        eps
        + math.sqrt(8.0 * math.log(5.0)) * nu * math.sqrt(dimension)
        + math.sqrt(8.0) * nu * math.sqrt(math.log(1.0 / delta))
    )
    return {
        "exact_net_radius": exact_net_radius,
        "additive_upper_bound": additive_upper_bound,
    }


def compact_moment_box_diameter(xi_max: float, sigma_min2: float) -> float:
    """Computes the Euclidean diameter M_{in,max} of the compact moment box
    D_bdd(Drift (x) Var) = [-xi_max, xi_max] x [sigma_min2, xi_max^2] under bounded shock support |xi| <= xi_max.
    """
    if not math.isfinite(xi_max) or not math.isfinite(sigma_min2):
        raise ValueError(f"Non-finite compact moment box parameters: xi_max={xi_max}, sigma_min2={sigma_min2}")
    v_max = xi_max * xi_max
    if xi_max <= 0.0 or sigma_min2 <= 0.0 or v_max < sigma_min2:
        raise ValueError(f"Invalid compact moment box parameters: xi_max={xi_max}, sigma_min2={sigma_min2} (requires xi_max^2 >= sigma_min2 > 0)")
    return math.hypot(2.0 * xi_max, v_max - sigma_min2)


def clamp_moment_pair_to_compact_box(
    f_hat: float,
    v_hat: float,
    xi_max: float,
    sigma_min2: float,
) -> dict[str, Any]:
    """Orthogonally projects a finite input moment estimate (f_hat, v_hat) onto the compact moment box
    D_bdd = [-xi_max, xi_max] x [sigma_min2, xi_max^2].
    Rejects non-finite box parameters (`INVALID_MOMENT_BOX`) and non-finite moment observations
    (`NON_FINITE_OBSERVATION`) before projection so NaN or +/-inf cannot be silently clamped to box endpoints.
    """
    if not math.isfinite(xi_max) or not math.isfinite(sigma_min2):
        return {"status": "UNESTABLISHED", "code": "INVALID_MOMENT_BOX"}
    v_max = xi_max * xi_max
    if xi_max <= 0.0 or sigma_min2 <= 0.0 or v_max < sigma_min2 or not math.isfinite(v_max):
        return {"status": "UNESTABLISHED", "code": "INVALID_MOMENT_BOX"}
    if not math.isfinite(f_hat) or not math.isfinite(v_hat):
        return {"status": "UNESTABLISHED", "code": "NON_FINITE_OBSERVATION"}
    f_clamped = max(-xi_max, min(xi_max, f_hat))
    v_clamped = max(sigma_min2, min(v_max, v_hat))
    m_in_max = math.hypot(2.0 * xi_max, v_max - sigma_min2)
    return {
        "status": "OK",
        "f_clamped": f_clamped,
        "v_clamped": v_clamped,
        "m_in_max": m_in_max,
    }


def moment_to_confidence_radius(cert: MomentCert, delta: float) -> Union[ConfidenceRadiusCert, UnestablishedCert]:
    """Converts a MomentCert with sub-Gaussian parameter nu_subg >= 0 into a ConfidenceRadiusCert via Lemma 7.1a(1)
    (high-probability ball only; does NOT assert an almost-sure diameter M_max).
    """
    if (
        not math.isfinite(delta)
        or not (0.0 < delta < 1.0)
        or not math.isfinite(cert.eps)
        or cert.eps < 0.0
        or not math.isfinite(cert.nu_subg)
        or cert.nu_subg < 0.0
        or cert.dimension < 1
    ):
        return UnestablishedCert("TAIL_FAILURE")
    rad = sub_gaussian_event_radius(cert.eps, cert.dimension, cert.nu_subg, delta)["exact_net_radius"]
    return ConfidenceRadiusCert(r=rad, delta=delta)


def moment_to_event(
    cert: MomentCert,
    delta: float,
    m_max: float | None = None,
) -> Union[EventCert, UnestablishedCert]:
    """Upgrades a MomentCert into a bounded EventCert via Lemma 7.1a(1) ONLY when an explicit finite
    almost-sure error bound m_max > 0 (e.g., from a compact domain clamp) is supplied.
    """
    if (
        not math.isfinite(delta)
        or not (0.0 < delta < 1.0)
        or not math.isfinite(cert.eps)
        or cert.eps < 0.0
        or not math.isfinite(cert.nu_subg)
        or cert.nu_subg < 0.0
        or cert.dimension < 1
    ):
        return UnestablishedCert("TAIL_FAILURE")
    if m_max is None or not math.isfinite(m_max) or m_max <= 0.0:
        return UnestablishedCert("MISSING_AS_BOUND")
    rad = sub_gaussian_event_radius(cert.eps, cert.dimension, cert.nu_subg, delta)["exact_net_radius"]
    return EventCert(r=min(rad, m_max), delta=delta, m_max=m_max)


def event_to_quadratic_regret_bound(
    lambda_max_phi: float,
    r: float,
    delta: float,
    m_max: float,
) -> float:
    """Lemma 7.1a(2a): Expected curvature-weighted squared control-error bound from EVENT(r, delta, M_max):
      E[||u_hat - u*||_Phi^2 | x_t, w_t] <= lambda_max(Phi) * ((1 - delta) * r^2 + delta * M_max^2).
    Equals the constrained quadratic objective regret when the constrained optimum u* is interior;
    when u* lies on the boundary of U(x_t), see `event_to_constrained_quadratic_objective_regret_bound`.
    """
    if (
        not math.isfinite(lambda_max_phi)
        or lambda_max_phi < 0.0
        or not math.isfinite(r)
        or r < 0.0
        or not math.isfinite(m_max)
        or m_max < 0.0
        or not math.isfinite(delta)
        or not (0.0 <= delta <= 1.0)
    ):
        raise ValueError(f"event_to_quadratic_regret_bound requires finite non-negative parameters and delta in [0, 1], got lambda_max_phi={lambda_max_phi}, r={r}, m_max={m_max}, delta={delta}")
    return lambda_max_phi * ((1.0 - delta) * (r * r) + delta * (m_max * m_max))


def event_to_constrained_quadratic_objective_regret_bound(
    sigma2: float,
    r: float,
    delta: float,
    m_max: float,
    boundary_slope_max: float = 0.0,
) -> dict[str, float]:
    """Lemma 7.1a(2b) & Lemma 2.2(1): Exact constrained quadratic objective-regret bound on [-Lambda_max, Lambda_max].
    For any theta in [-Lambda_max, Lambda_max] and constrained optimum theta_quad* = Pi_{[-Lambda_max, Lambda_max]}(f / sigma^2),
    completing the square around theta_quad* yields the exact identity:
      Q_t^{quad}(theta_quad*) - Q_t^{quad}(theta)
        = (sigma^2 / 2) * (theta - theta_quad*)^2 + g_bnd * |theta - theta_quad*|,
    where g_bnd := |f - sigma^2 * theta_quad*| = max(0, |f| - sigma^2 * Lambda_max) <= boundary_slope_max
    (with g_bnd == 0 whenever theta_quad* is interior).
    Taking expectations over EVENT(r, delta, M_max) yields:
      E[Q_t^{quad}(theta_quad*) - Q_t^{quad}(theta_hat)]
        <= (sigma^2 / 2) * ((1 - delta) * r^2 + delta * M_max^2)
           + boundary_slope_max * ((1 - delta) * r + delta * M_max).
    """
    if (
        not math.isfinite(sigma2)
        or sigma2 < 0.0
        or not math.isfinite(r)
        or r < 0.0
        or not math.isfinite(m_max)
        or m_max < 0.0
        or not math.isfinite(delta)
        or not (0.0 <= delta <= 1.0)
        or not math.isfinite(boundary_slope_max)
        or boundary_slope_max < 0.0
    ):
        raise ValueError("Invalid parameters for event_to_constrained_quadratic_objective_regret_bound")
    control_error_term = event_to_quadratic_regret_bound(0.5 * sigma2, r, delta, m_max)
    boundary_slope_term = boundary_slope_max * ((1.0 - delta) * r + delta * m_max)
    return {
        "quadratic_control_error_bound": control_error_term,
        "boundary_clipping_regret_bound": boundary_slope_term,
        "constrained_quadratic_objective_regret_bound": control_error_term + boundary_slope_term,
    }


def exact_kelly_hessian_spectral_norm(f: float, v: float) -> float:
    """Exact spectral norm ||D^2 o(f, v)||_op for o(f, v) = f / v at point (f, v)."""
    b = 1.0 / (v * v)
    c = abs((2.0 * f) / (v * v * v))
    return 0.5 * (c + math.sqrt(c * c + 4.0 * b * b))


def kelly_ratio_certificate_pushforward(
    f_hat: float,
    v_hat: float,
    r: float,
    sigma_min2: float,
    delta: float = 0.05,
) -> dict[str, Any]:
    """Proposition 7.2(2) Stage A (Unclamped Ratio Confidence Pushforward):
    Computes conservative analytical box upper bounds L_hat_o and M_hat_{H,o} on the observable
    Euclidean ball B_{2r}(f_hat, v_hat) for the raw ratio o(f, v) = f / v on R x [sigma_min2, inf).
    Returns `status="CONF_RADIUS"` with high-probability radius `r_raw` without claiming an
    action-interval almost-sure bound `2 * Lambda_max`.
    """
    if (
        not math.isfinite(f_hat)
        or not math.isfinite(v_hat)
        or not math.isfinite(r)
        or not math.isfinite(sigma_min2)
        or not math.isfinite(delta)
        or r < 0.0
        or sigma_min2 <= 0.0
        or not (0.0 < delta < 1.0)
    ):
        return {"status": "UNESTABLISHED", "code": "DOMAIN_VIOLATION"}
    v_min_ball = v_hat - 2.0 * r
    if v_min_ball < sigma_min2:
        return {"status": "UNESTABLISHED", "code": "DOMAIN_VIOLATION"}
    f_max_ball = abs(f_hat) + 2.0 * r
    l_hat_o = math.hypot(1.0 / v_min_ball, f_max_ball / (v_min_ball * v_min_ball))
    b = 1.0 / (v_min_ball * v_min_ball)
    c = (2.0 * f_max_ball) / (v_min_ball * v_min_ball * v_min_ball)
    m_hat_ho = 0.5 * (c + math.sqrt(c * c + 4.0 * b * b))
    r_raw = l_hat_o * r + 0.5 * m_hat_ho * r * r
    return {
        "status": "CONF_RADIUS",
        "theta_raw": f_hat / v_hat,
        "r_raw": r_raw,
        "r_out": r_raw,
        "L_hat_o": l_hat_o,
        "M_hat_Ho": m_hat_ho,
        "delta": delta,
    }


def project_kelly_exposure_certificate(
    raw_cert: dict[str, Any],
    lambda_max: float,
) -> dict[str, Any]:
    """Proposition 7.2(2) Stage B (Clamped Action Projection to Bounded EVENT Certificate):
    Applies the orthogonal interval projection Pi_{[-Lambda_max, Lambda_max]} to the unclamped
    ratio certificate from Stage A.
    """
    if raw_cert.get("status") == "UNESTABLISHED":
        return dict(raw_cert)
    if raw_cert.get("status") != "CONF_RADIUS" or not math.isfinite(lambda_max) or lambda_max <= 0.0:
        return {"status": "UNESTABLISHED", "code": "INVALID_PROJECTION_CONTRACT"}
    theta_raw = float(raw_cert["theta_raw"])
    r_raw = float(raw_cert["r_raw"])
    delta = float(raw_cert["delta"])
    if not (math.isfinite(theta_raw) and math.isfinite(r_raw) and math.isfinite(delta)) or r_raw < 0.0 or not (0.0 < delta < 1.0):
        return {"status": "UNESTABLISHED", "code": "INVALID_PROJECTION_CONTRACT"}
    m_out_max = 2.0 * lambda_max
    theta_clamped = max(-lambda_max, min(lambda_max, theta_raw))
    r_proj = min(r_raw, m_out_max)
    return {
        "status": "EVENT",
        "theta_raw": theta_raw,
        "theta_clamped": theta_clamped,
        "r_raw": r_raw,
        "r_out": r_proj,
        "delta": delta,
        "m_max": m_out_max,
        "L_hat_o": raw_cert["L_hat_o"],
        "M_hat_Ho": raw_cert["M_hat_Ho"],
    }


def serial_scalar(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """Proposition 7.2: Serial composition of scalar moment/event bounds."""
    sd = math.sqrt(max(0.0, a["v"])) + math.sqrt(max(0.0, b["v"]))
    return {
        "eps": a["eps"] + b["eps"],
        "v": sd * sd,
        "delta": min(1.0, a["delta"] + b["delta"]),
    }


def push_scalar(
    inp: dict[str, float],
    jacobian: float,
    intrinsic_bias: float,
    hessian_bound: float = 0.0,
    kurtosis_factor: float = 3.0,
) -> dict[str, float]:
    """Proposition 7.2(3): Second-order scalar moment pushforward comparing Gaussian Isserlis
    variance (`v_gaussian_quad`) vs. distribution-free fourth-moment Cauchy-Schwarz bound (`v_general_fourth_moment`).
    """
    v = max(0.0, inp["v"])
    k4 = max(1.0, kurtosis_factor)
    second_order_bias = 0.5 * hessian_bound * (inp["eps"] * inp["eps"] + v)
    second_order_var = 0.5 * hessian_bound * hessian_bound * v * v
    v_gaussian_quad = jacobian * v * jacobian + second_order_var
    linear_sd = abs(jacobian) * math.sqrt(v)
    quad_rms = 0.5 * hessian_bound * math.sqrt(k4) * v
    v_general_fourth_moment = (linear_sd + quad_rms) ** 2
    return {
        "eps": intrinsic_bias + abs(jacobian) * inp["eps"] + second_order_bias,
        "v": v_gaussian_quad,
        "v_gaussian_quad": v_gaussian_quad,
        "v_general_fourth_moment": v_general_fourth_moment,
        "delta": min(1.0, inp["delta"]),
    }


def push_split_reconverge(
    source_var: float,
    j1: float,
    j2: float,
    correlation: float,
    hessian_norm: float,
    kurtosis_factor: float = 3.0,
) -> dict[str, float]:
    """Proposition 7.2(1) & 7.2(3): Tree-multicategory pushforward vs. Copy-Discard PROP pushforward."""
    v = max(0.0, source_var)
    k4 = max(1.0, kurtosis_factor)
    naive_tree_var = j1 * j1 * v + j2 * j2 * v
    prop_true_linear_var = max(0.0, j1 * j1 * v + j2 * j2 * v + 2.0 * j1 * j2 * correlation * v)
    trace_vin = 2.0 * v
    hessian_guard = 0.5 * hessian_norm * hessian_norm * trace_vin * trace_vin
    prop_certified_var = prop_true_linear_var + hessian_guard
    quad_rms = 0.5 * hessian_norm * math.sqrt(k4) * trace_vin
    prop_fourth_moment_certified_var = (math.sqrt(prop_true_linear_var) + quad_rms) ** 2
    return {
        "naive_tree_var": naive_tree_var,
        "prop_true_linear_var": prop_true_linear_var,
        "hessian_guard": hessian_guard,
        "prop_certified_var": prop_certified_var,
        "prop_fourth_moment_certified_var": prop_fourth_moment_certified_var,
        "underestimation_gap": max(0.0, prop_certified_var - naive_tree_var),
    }
