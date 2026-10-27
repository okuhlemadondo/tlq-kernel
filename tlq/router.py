"""Sections 8 and 9: Guarded Horizontal Hypergraph Routing & End-to-End Composition.

Implements:
- Anytime-valid confidence schedule delta_k = 6 * delta / (pi^2 * k^2)
- All-pairs Azuma-Hoeffding concentration width kappa_search (Definition 8.1(2) & Theorem 8.4c)
- Factored score increment and self-normalized predictable-variation bound kappa_pred (Proposition 8.4e)
- Guard 1 (Immediate Epistemic Collapse) & Guard 2 (Certified Expansion with Symmetric Dwell Reset, Theorem 8.4b)
- Post-switch regime-shift drift diagnostic D_regime(t_j) (Theorem 8.4d)
- End-to-End Vertical-to-Horizontal Composition (Theorem 9.1) separating squared control error
  from constrained quadratic objective regret (including the boundary normal-cone slope term).
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
    clamp_moment_pair_to_compact_box,
    kelly_ratio_certificate_pushforward,
    project_kelly_exposure_certificate,
    event_to_constrained_quadratic_objective_regret_bound,
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
    xi_max: float | None = None,
    f_true_abs_max: float | None = None,
) -> dict[str, Any]:
    """Theorem 9.1 (Package B End-to-End Composition with Explicit Separation of Squared Control Error
    vs. Constrained Quadratic Objective Regret & Safety-Cap Contract Validation):
    0. Entry-Point Contract Validation:
       Verifies 0 < eta_margin < 1, lambda_max > 0, xi_max > 0, and the bounded-shock pathwise survival
       relationship lambda_max * xi_max <= 1 - eta_margin (returning UNESTABLISHED(SAFETY_CAP_VIOLATION)
       and falling back to G_null if violated). Also validates 0 < delta < 1, eps_moment >= 0, nu_subg >= 0,
       sigma_min2 <= sigma2_true <= xi_max^2, third_abs_moment >= 0, and (when supplied as an explicit
       caller-assumed prior drift envelope |f_t| <= f_true_abs_max) 0 <= f_true_abs_max <= xi_max.
    1. Stage 1 (Compact Input Moment Box Projection & Input EVENT Certificate):
       Derives xi_max = (1 - eta_margin) / lambda_max (if not passed), checks xi_max^2 >= sigma_min2 > 0,
       projects (f_hat, v_hat) onto D_bdd = [-xi_max, xi_max] x [sigma_min2, xi_max^2], and combines
       Lemma 7.1a(1)'s sub-Gaussian radius r_in with D_bdd's finite diameter M_{in,max} to form a valid
       input EVENT(min(r_in, M_{in,max}), delta, M_{in,max}) certificate.
    2. Stage 2 (Unclamped Ratio Confidence Pushforward):
       Applies `kelly_ratio_certificate_pushforward` on B_{2*r_in_eff}(f_clamped, v_clamped), producing
       either UNESTABLISHED(DOMAIN_VIOLATION) or a raw-ratio CONF_RADIUS(r_raw, delta) certificate.
    3. Stage 3 (Clamped Exposure Projection to Output EVENT Certificate):
       Applies `project_kelly_exposure_certificate` (Pi_{[-Lambda_max, Lambda_max]}), using Moreau
       projection non-expansiveness (Lemma 2.2(3)) to contract the good-event radius to
       r_out = min(r_raw, 2 * Lambda_max) and establish the almost-sure bound M_{out,max} = 2 * Lambda_max
       relative to the CONSTRAINED oracle theta_quad* = Pi_{[-Lambda_max, Lambda_max]}(f_t / sigma_t^2).
    4. Stage 4 (Squared Control Error, Boundary Clipping Slope Term, & Exact-Log Suboptimality):
       Uses the exact constrained quadratic identity:
         Q_t^{quad}(theta_quad*) - Q_t^{quad}(theta_t)
           = (sigma_t^2 / 2) * (theta_t - theta_quad*)^2 + g_{bnd,t} * |theta_t - theta_quad*|,
       where g_{bnd,t} = max(0, |f_t| - sigma_t^2 * Lambda_max) <= max(0, f_abs_bound - sigma_t^2 * Lambda_max)
       (vanishing identically whenever theta_quad* is interior), and adds the Stage 2b Taylor remainder.
       Note: When optional `f_true_abs_max` is omitted, the unconditional bounded-shock support bound
       |f_t| <= xi_max is used; when supplied, |f_t| <= f_true_abs_max is a caller-supplied prior assumption.
    """
    if (
        not math.isfinite(lambda_max)
        or lambda_max <= 0.0
        or not math.isfinite(eta_margin)
        or not (0.0 < eta_margin < 1.0)
    ):
        return {
            "status": "UNESTABLISHED",
            "code": "SAFETY_CAP_VIOLATION",
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }
    xi_max_eff = xi_max if xi_max is not None else (1.0 - eta_margin) / lambda_max
    if (
        not math.isfinite(xi_max_eff)
        or xi_max_eff <= 0.0
        or lambda_max * xi_max_eff > (1.0 - eta_margin) + 1e-12
    ):
        return {
            "status": "UNESTABLISHED",
            "code": "SAFETY_CAP_VIOLATION",
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }
    if (
        not math.isfinite(delta)
        or not (0.0 < delta < 1.0)
        or not math.isfinite(eps_moment)
        or eps_moment < 0.0
        or not math.isfinite(nu_subg)
        or nu_subg < 0.0
    ):
        return {
            "status": "UNESTABLISHED",
            "code": "TAIL_FAILURE",
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }
    box = clamp_moment_pair_to_compact_box(f_hat, v_hat, xi_max_eff, sigma_min2)
    if box["status"] == "UNESTABLISHED":
        return {
            "status": "UNESTABLISHED",
            "code": box["code"],
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }
    if (
        not math.isfinite(sigma2_true)
        or sigma2_true < sigma_min2
        or sigma2_true > (xi_max_eff * xi_max_eff) + 1e-12
        or not math.isfinite(third_abs_moment)
        or third_abs_moment < 0.0
        or (
            f_true_abs_max is not None
            and (not math.isfinite(f_true_abs_max) or f_true_abs_max < 0.0 or f_true_abs_max > xi_max_eff + 1e-12)
        )
    ):
        return {
            "status": "UNESTABLISHED",
            "code": "INVALID_MOMENT_ENVELOPE",
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }
    f_clamped = box["f_clamped"]
    v_clamped = box["v_clamped"]
    m_in_max = box["m_in_max"]

    rad_in_unbounded = sub_gaussian_event_radius(eps_moment, 2, nu_subg, delta)["exact_net_radius"]
    r_in_eff = min(rad_in_unbounded, m_in_max)

    raw_push = kelly_ratio_certificate_pushforward(f_clamped, v_clamped, r_in_eff, sigma_min2, delta=delta)
    if raw_push["status"] == "UNESTABLISHED":
        return {
            "status": "UNESTABLISHED",
            "code": raw_push["code"],
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }

    proj_cert = project_kelly_exposure_certificate(raw_push, lambda_max)
    if proj_cert["status"] == "UNESTABLISHED":
        return {
            "status": "UNESTABLISHED",
            "code": proj_cert["code"],
            "fallback_graph": "G_null",
            "fallback_exposure": 0.0,
        }

    r_raw = proj_cert["r_raw"]
    r_out = proj_cert["r_out"]
    m_out_max = proj_cert["m_max"]

    f_abs_bound = xi_max_eff if f_true_abs_max is None else min(xi_max_eff, max(0.0, f_true_abs_max))
    boundary_slope_max = max(0.0, f_abs_bound - sigma2_true * lambda_max)

    regret_decomp = event_to_constrained_quadratic_objective_regret_bound(
        sigma2_true,
        r_out,
        delta,
        m_out_max,
        boundary_slope_max=boundary_slope_max,
    )
    quad_control_err = regret_decomp["quadratic_control_error_bound"]
    boundary_regret = regret_decomp["boundary_clipping_regret_bound"]
    constrained_quad_obj_regret = regret_decomp["constrained_quadratic_objective_regret_bound"]

    surrogate_bounds = exact_log_objective_gap_bounds(lambda_max, third_abs_moment, sigma2_true, eta_margin)
    return {
        "status": "CERTIFIED",
        "f_clamped": f_clamped,
        "v_clamped": v_clamped,
        "m_in_max": m_in_max,
        "r_in": r_in_eff,
        "raw_ratio_status": raw_push["status"],
        "r_raw": r_raw,
        "projected_status": proj_cert["status"],
        "r_out": r_out,
        "m_out_max": m_out_max,
        "theta_raw": proj_cert["theta_raw"],
        "theta_clamped": proj_cert["theta_clamped"],
        "boundary_slope_max": boundary_slope_max,
        "quadratic_control_error_bound": quad_control_err,
        "quadratic_control_loss_bound": quad_control_err,
        "boundary_clipping_regret_bound": boundary_regret,
        "constrained_quadratic_objective_regret_bound": constrained_quad_obj_regret,
        "surrogate_stage2b_remainder": surrogate_bounds["taylor_bound"],
        "total_exact_log_suboptimality_bound": constrained_quad_obj_regret + surrogate_bounds["taylor_bound"],
    }
