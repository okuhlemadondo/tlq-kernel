"""Tests for Sections 2, 5, and 6: Theorem 2.4(2) Surrogate Gap Bounds & Ruin-Boundary Semantics (Checks 1A, 1B, 6B)."""
from __future__ import annotations
import math
from tlq.vertical import (
    clamp,
    constrained_action_gap_bound,
    exact_log_objective_gap_bounds,
    log_obj,
    log_grad,
    solve_constrained_log_optimum,
    truncated_log_compare,
)


def _mean(xs: list[float], ps: list[float]) -> float:
    return sum(x * p for x, p in zip(xs, ps))


def _moment(xs: list[float], ps: list[float], k: int) -> float:
    return sum((abs(x) ** k) * p for x, p in zip(xs, ps))


def test_1a_skewed_two_point_shock_counterexample():
    """Check 1A: Skewed two-point shock has theta_log* = 0.50 != theta_quad* = 0.40."""
    xs = [-0.5, 1.0]
    ps = [0.5, 0.5]
    f = _mean(xs, ps)                       # 0.25
    sigma2 = _mean([x * x for x in xs], ps) # 0.625
    lambda_max = 0.8

    theta_quad = clamp(f / sigma2, -lambda_max, lambda_max)
    theta_log = solve_constrained_log_optimum(xs, ps, lambda_max)

    assert abs(theta_quad - 0.4) <= 1e-8
    assert abs(theta_log - 0.5) <= 1e-8
    assert abs(theta_quad - theta_log) > 0.09


def test_1b_theorem_2_4_variational_and_objective_bounds_across_800_distributions():
    """Check 1B: Verifies intermediate variational step, action-distance bound, and both objective-gap bounds."""
    seed = 0x6D2B79F5

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / (2 ** 32)

    interior_cases = 0
    boundary_cases = 0

    for _k in range(800):
        n = 2 + int(rnd() * 6)
        xi_max = 0.8
        xs = [(2.0 * rnd() - 1.0) * xi_max for _ in range(n)]
        raw = [0.05 + rnd() for _ in range(n)]
        z = sum(raw)
        ps = [p / z for p in raw]

        f = _mean(xs, ps)
        sigma2 = _mean([x * x for x in xs], ps)
        if sigma2 < 0.01:
            continue

        eta_margin = 0.15 + 0.5 * rnd()
        lambda_max = (1.0 - eta_margin) / xi_max

        theta_quad = clamp(f / sigma2, -lambda_max, lambda_max)
        theta_log = solve_constrained_log_optimum(xs, ps, lambda_max)

        if abs(abs(theta_quad) - lambda_max) < 1e-9 or abs(abs(theta_log) - lambda_max) < 1e-9:
            boundary_cases += 1
        else:
            interior_cases += 1

        d = theta_log - theta_quad
        m3 = _moment(xs, ps, 3)
        grad_diff_bound = (lambda_max * lambda_max * m3) / eta_margin

        g_prime_log_l = log_grad(theta_log, xs, ps)
        g_prime_quad_l = f - sigma2 * theta_log
        lhs_var = sigma2 * d * d
        mid_var = (g_prime_log_l - g_prime_quad_l) * d
        rhs_var = abs(g_prime_log_l - g_prime_quad_l) * abs(d)
        assert lhs_var <= mid_var + 1e-12
        assert mid_var <= rhs_var + 1e-12

        rhs_action = constrained_action_gap_bound(lambda_max, m3, sigma2, eta_margin)
        assert abs(d) <= rhs_action + 1e-10

        actual_obj_gap = log_obj(theta_log, xs, ps) - log_obj(theta_quad, xs, ps)
        assert actual_obj_gap >= -1e-12

        bounds = exact_log_objective_gap_bounds(lambda_max, m3, sigma2, eta_margin)
        assert abs(bounds["variational_bound"] - (grad_diff_bound * grad_diff_bound) / sigma2) <= 1e-14
        assert actual_obj_gap <= bounds["combined_bound"] + 1e-10

    assert interior_cases > 100
    assert boundary_cases > 50


def test_6b_truncated_log_compare_ruin_boundary_and_safe_interior_semantics():
    """Check 6B: Verifies safe-interior finiteness, exact -inf on ruin crossing, and ruinMass vs nearRuinMass."""
    safe_case = truncated_log_compare(0.35, 1.0)
    assert math.isfinite(safe_case["exact_at_star"])
    assert safe_case["ruin_mass"] == 0.0
    assert safe_case["near_ruin_mass"] == 0.0
    assert abs(safe_case["ruin_boundary_hi"] - 1.0 / 2.65) <= 1e-12

    for theta, exact_val, r_mass in zip(
        safe_case["theta_grid"], safe_case["exact"], safe_case["ruin_mass_grid"]
    ):
        if theta > safe_case["ruin_boundary_hi"] + 1e-9 or theta < safe_case["ruin_boundary_lo"] - 1e-9:
            assert exact_val == float("-inf")
            assert r_mass > 0.0
        elif safe_case["ruin_boundary_lo"] + 1e-9 < theta < safe_case["ruin_boundary_hi"] - 1e-9:
            assert math.isfinite(exact_val)
            assert r_mass == 0.0

    ruin_at_star = truncated_log_compare(0.45, 1.0)
    assert ruin_at_star["theta_star"] > ruin_at_star["ruin_boundary_hi"]
    assert ruin_at_star["ruin_mass"] > 0.0
    assert ruin_at_star["exact_at_star"] == float("-inf")

    probe = truncated_log_compare(0.35, 1.0, 0.25)
    assert math.isfinite(probe["exact_at_star"])
    assert probe["ruin_mass"] == 0.0
    assert probe["near_ruin_mass"] > 0.0
