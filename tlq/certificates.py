"""Section 7: Epistemic Certificate Algebra (K) — Definition 7.1, Lemma 7.1a, and Proposition 7.2.

Implements:
- ExactCert, MomentCert, EventCert, UnestablishedCert
- Lemma 7.1a(1): Repaired (1/2)-net sub-Gaussian Euclidean norm concentration radius r_delta^{subG}
- Lemma 7.1a(2): Expected quadratic control regret bound from EVENT(r, delta, M_max)
- Proposition 7.2(1)-(3): Guarded KellyTarget pushforward, serial composition, and 4th-moment Cauchy-Schwarz bounds
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
class EventCert:
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


Certificate = Union[ExactCert, MomentCert, EventCert, UnestablishedCert]


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


def moment_to_event(cert: MomentCert, delta: float, m_max: float) -> EventCert | UnestablishedCert:
    """Converts a MomentCert with sub-Gaussian parameter nu_subg > 0 into an EventCert via Lemma 7.1a(1)."""
    if delta <= 0.0 or delta >= 1.0 or cert.nu_subg < 0.0:
        return UnestablishedCert("TAIL_FAILURE")
    rad = sub_gaussian_event_radius(cert.eps, cert.dimension, cert.nu_subg, delta)["exact_net_radius"]
    return EventCert(r=rad, delta=delta, m_max=m_max)


def event_to_quadratic_regret_bound(
    lambda_max_phi: float,
    r: float,
    delta: float,
    m_max: float,
) -> float:
    """Lemma 7.1a(2): Expected quadratic control regret upper bound from EVENT(r, delta, M_max):
      E[||u_hat - u*||_Phi^2 | x_t, w_t] <= lambda_max(Phi) * ((1 - delta) * r^2 + delta * M_max^2).
    """
    return lambda_max_phi * ((1.0 - delta) * (r * r) + delta * (m_max * m_max))


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
    lambda_max: float = 1.0,
) -> dict[str, Any]:
    """Proposition 7.2(2): Conservative analytical box upper bounds L_hat_o and M_hat_{H,o}
    on the observable Euclidean ball B_{2r}(f_hat, v_hat) for o(f, v) = f / v on R x [sigma_min2, inf).
    """
    v_min_ball = v_hat - 2.0 * r
    if v_min_ball < sigma_min2:
        return {"status": "UNESTABLISHED", "code": "DOMAIN_VIOLATION"}
    f_max_ball = abs(f_hat) + 2.0 * r
    l_hat_o = math.hypot(1.0 / v_min_ball, f_max_ball / (v_min_ball * v_min_ball))
    b = 1.0 / (v_min_ball * v_min_ball)
    c = (2.0 * f_max_ball) / (v_min_ball * v_min_ball * v_min_ball)
    m_hat_ho = 0.5 * (c + math.sqrt(c * c + 4.0 * b * b))
    r_out = l_hat_o * r + 0.5 * m_hat_ho * r * r
    return {
        "status": "EVENT",
        "r_out": r_out,
        "L_hat_o": l_hat_o,
        "M_hat_Ho": m_hat_ho,
        "delta": delta,
        "m_max": 2.0 * lambda_max,
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
