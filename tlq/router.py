"""Sections 8 and 9: Guarded Horizontal Hypergraph Routing & End-to-End Composition.

Implements:
- Anytime-valid confidence schedule delta_k = 6 * delta / (pi^2 * k^2)
- All-pairs Azuma-Hoeffding concentration width kappa_search (Definition 8.1(2) & Theorem 8.4c)
- Factored score increment and self-normalized predictable-variation bound kappa_pred (Proposition 8.4e)
- Guard 1 (Immediate Epistemic Collapse) & Guard 2 (Certified Expansion with Symmetric Dwell Reset, Theorem 8.4b)
- Post-switch regime-shift drift diagnostic D_regime(t_j) (Theorem 8.4d)
- End-to-End Vertical-to-Horizontal Composition (Theorem 9.1)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any
from .vertical import (
    clamp,
    control_loss,
    null_loss,
    ema_theta_variance,
    ema_sharpe_hurdle,
    squared_sharpe,
    exact_log_objective_gap_bounds,
)
from .certificates import (
    sub_gaussian_event_radius,
    kelly_ratio_certificate_pushforward,
    event_to_quadratic_regret_bound,
)


def anytime_delta(base_delta: float, eval_index: int) -> float:
    k = max(1, int(eval_index))
    return (6.0 * base_delta) / (math.pi * math.pi * k * k)


def search_penalty(candidates: int, delta: float, window: int) -> float:
    if window <= 0 or delta <= 0.0 or candidates <= 0:
        return float("inf")
    return (2.0 * math.log(candidates / delta)) / window


def all_pairs_search_penalty(
    m_total: int,
    window_length: int,
    delta_k: float,
    lambda_max: float,
    xi_max: float,
) -> dict[str, float]:
    """Definition 8.1(2) & Theorem 8.4c:
      B_max = 2 * Lambda_max * xi_max + 0.5 * Lambda_max^2 * xi_max^2
      kappa_search(M_total, L, delta_k) = 2 * B_max * sqrt(2 * ln(M_total^2 / delta_k) / L).
    """
    b_max = 2.0 * lambda_max * xi_max + 0.5 * (lambda_max * lambda_max) * (xi_max * xi_max)
    kappa_search = 2.0 * b_max * math.sqrt((2.0 * math.log((m_total * m_total) / delta_k)) / window_length)
    return {"b_max": b_max, "kappa_search": kappa_search}


def factored_score_increment(theta_a: float, theta_b: float, xi: float) -> float:
    """Proposition 8.4e(1): d_tau(G_a, G_b) = (a - b) * (0.5 * (a + b) * xi^2 - xi)."""
    return (theta_a - theta_b) * (0.5 * (theta_a + theta_b) * (xi * xi) - xi)


def predictable_variation_penalty(
    m_total: int,
    window_length: int,
    delta_k: float,
    lambda_max: float,
    xi_max: float,
    sum_sq_exposure_diff: float,
    rho: float | None = None,
) -> dict[str, float]:
    """Proposition 8.4e(2)-(3): Self-normalized predictable-variation concentration width via
    Gaussian supermartingale mixture over V_hat_L^{obs} = B_lip^2 * sum (a_{tau-1} - b_{tau-1})^2.
    """
    b_lip = xi_max + lambda_max * (xi_max * xi_max)
    v_obs = (b_lip * b_lip) * sum_sq_exposure_diff
    rho_used = rho if rho is not None else window_length * (b_lip * b_lip) * (lambda_max * lambda_max)
    kappa_pred = (1.0 / window_length) * math.sqrt(
        2.0
        * (rho_used + v_obs)
        * math.log(((m_total * m_total) / (2.0 * delta_k)) * math.sqrt(1.0 + v_obs / rho_used))
    )
    return {
        "b_lip": b_lip,
        "v_obs": v_obs,
        "rho_used": rho_used,
        "kappa_pred": kappa_pred,
    }


def max_switches_bound(horizon_n: int, tau_dwell: int) -> int:
    """Theorem 8.4b: N_switch([0, N-1]) <= 2 * floor(N / (tau_dwell + 1)) + 2."""
    return 2 * (horizon_n // (tau_dwell + 1)) + 2


def evaluate_gate(
    alpha: float,
    drift: float,
    variance: float,
    bias: float,
    candidates: int,
    delta: float,
    window: int,
    hysteresis: float,
    eval_step: int | None = None,
    hac_factor: float = 1.0,
) -> dict[str, Any]:
    estimator_variance = ema_theta_variance(alpha, variance) * hac_factor
    loss_active = control_loss(variance, bias, estimator_variance)
    loss_null = null_loss(drift, variance)
    population_gap = loss_null - loss_active
    delta_k = anytime_delta(delta, eval_step) if eval_step is not None else delta
    penalty = search_penalty(candidates, delta_k, window)
    net_gain = population_gap - penalty
    sharpe2 = squared_sharpe(drift, variance)
    hurdle = variance * estimator_variance + variance * bias * bias
    return {
        "sharpe2": sharpe2,
        "hurdle": hurdle,
        "estimator_variance": estimator_variance,
        "loss_active": loss_active,
        "loss_null": loss_null,
        "population_gap": population_gap,
        "penalty": penalty,
        "net_gain": net_gain,
        "delta_k": delta_k,
        "population_trades": sharpe2 > hurdle,
        "gate_opens": net_gain > hysteresis,
    }


@dataclass
class GuardedRouter:
    """Context-dependent horizontal router G_t =>_{C_t} G_{t+1} (Definition 8.3 & Theorem 8.4b)."""
    tau_dwell: int = 5
    tau_hyst: float = 0.015
    active_graph: str = "G_null"
    tau_elapsed: int = 0
    switch_epochs: list[tuple[int, str, str, str]] = field(default_factory=list)

    def step(
        self,
        t: int,
        active_cert_valid: bool,
        collapse_gain_to_null: float,
        candidate_gains: dict[str, tuple[bool, float]],
    ) -> str:
        """Executes Definition 8.3 priority rules at epoch t and updates tau_elapsed symmetrically."""
        # Guard 1: Immediate Epistemic Shield (Collapse Rule G_t =>^- G_null)
        if self.active_graph != "G_null" and (not active_cert_valid or collapse_gain_to_null > 0.0):
            prev = self.active_graph
            self.active_graph = "G_null"
            self.tau_elapsed = 0
            self.switch_epochs.append((t, prev, "G_null", "COLLAPSE"))
            return self.active_graph

        # Guard 2: Certified Structural Expansion (Grafting Rule G_t =>^+ G_new)
        if self.tau_elapsed >= self.tau_dwell:
            best_cand = None
            best_gain = self.tau_hyst
            for name, (cert_ok, gain) in candidate_gains.items():
                if name != self.active_graph and cert_ok and gain > best_gain:
                    best_gain = gain
                    best_cand = name
            if best_cand is not None:
                prev = self.active_graph
                self.active_graph = best_cand
                self.tau_elapsed = 0
                self.switch_epochs.append((t, prev, best_cand, "EXPANSION"))
                return self.active_graph

        # Rule 3: Inertial Persistence
        self.tau_elapsed += 1
        return self.active_graph


def compose_end_to_end_certificate(
    f_hat: float,
    v_hat: float,
    eps_moment: float,
    nu_subg: float,
    delta: float,
    sigma_min2: float,
    sigma2_true: float,
    lambda_max: float,
    eta_margin: float,
    third_abs_moment: float,
) -> dict[str, Any]:
    """Theorem 9.1 (Package B End-to-End Composition):
    Chains Lemma 7.1a(1) -> Proposition 7.2(2) -> Lemma 7.1a(2) -> Theorem 2.4(2) + Theorem 5.2.
    """
    rad_in = sub_gaussian_event_radius(eps_moment, 2, nu_subg, delta)["exact_net_radius"]
    push = kelly_ratio_certificate_pushforward(f_hat, v_hat, rad_in, sigma_min2, delta=delta, lambda_max=lambda_max)
    if push["status"] == "UNESTABLISHED":
        return {
            "status": "UNESTABLISHED",
            "code": push["code"],
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }
    r_out = push["r_out"]
    quad_loss_bound = event_to_quadratic_regret_bound(0.5 * sigma2_true, r_out, delta, 2.0 * lambda_max)
    surrogate_bounds = exact_log_objective_gap_bounds(lambda_max, third_abs_moment, sigma2_true, eta_margin)
    theta_clamped = clamp(f_hat / v_hat, -lambda_max, lambda_max)
    return {
        "status": "CERTIFIED",
        "r_in": rad_in,
        "r_out": r_out,
        "theta_clamped": theta_clamped,
        "quadratic_control_loss_bound": quad_loss_bound,
        "surrogate_stage2b_remainder": surrogate_bounds["taylor_bound"],
        "total_exact_log_suboptimality_bound": quad_loss_bound + surrogate_bounds["taylor_bound"],
    }
