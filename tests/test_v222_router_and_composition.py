"""Tests for Sections 8 and 9: All-Pairs Router Concentration, Guard 2 Negative Test, Predictable Variation, and Theorem 9.1 (Checks 4A, 4B, 5A, 5B)."""
from __future__ import annotations
import math
from tlq.vertical import clamp
from tlq.router import (
    anytime_delta,
    all_pairs_search_penalty,
    factored_score_increment,
    predictable_variation_penalty,
    max_switches_bound,
    GuardedRouter,
    compose_end_to_end_certificate,
)


def _mean(xs: list[float], ps: list[float]) -> float:
    return sum(x * p for x, p in zip(xs, ps))


def test_4a_all_pairs_router_concentration_across_overlapping_windows():
    """Check 4A: Simulation regression of Theorem 8.4c across 9 overlapping windows and 5 candidate routers."""
    seed = 42424242

    def rand() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / (2 ** 32)

    alphas = [0.0, 0.08, 0.18, 0.35, 0.65]
    m_total = len(alphas)
    xi_max = 0.5
    eta_margin = 0.2
    lambda_max = (1.0 - eta_margin) / xi_max
    sigma_min2 = 0.05

    window_l = 60
    horizon = 180
    eval_epochs = [60, 75, 90, 105, 120, 135, 150, 165, 180]
    delta = 0.05
    num_trajectories = 400

    trajectory_violations = 0

    for _tr in range(num_trajectories):
        f_path: list[float] = []
        sigma2_path: list[float] = []
        xi_path: list[float] = []

        ema_drift = [0.0] * m_total
        ema_var = [0.12] * m_total
        theta_history: list[list[float]] = [[] for _ in range(m_total)]

        for t in range(horizon):
            f_t = 0.12 if t < 100 else -0.05
            step_amp = 0.32
            sigma2_t = f_t * f_t + step_amp * step_amp
            eps_t = -step_amp if rand() < 0.5 else +step_amp
            xi_t = f_t + eps_t

            f_path.append(f_t)
            sigma2_path.append(sigma2_t)
            xi_path.append(xi_t)

            for g in range(m_total):
                v_hat = max(ema_var[g], sigma_min2)
                raw_theta = 0.0 if g == 0 else ema_drift[g] / v_hat
                theta_history[g].append(clamp(raw_theta, -lambda_max, lambda_max))

            for g in range(1, m_total):
                a = alphas[g]
                ema_drift[g] = (1.0 - a) * ema_drift[g] + a * xi_t
                ema_var[g] = (1.0 - a) * ema_var[g] + a * (xi_t * xi_t)

        any_eval_violated = False

        for idx, t_k in enumerate(eval_epochs):
            delta_k = anytime_delta(delta, idx + 1)
            kappa = all_pairs_search_penalty(m_total, window_l, delta_k, lambda_max, xi_max)["kappa_search"]

            for a in range(m_total):
                for b in range(a + 1, m_total):
                    emp_diff_sum = 0.0
                    true_loss_diff_sum = 0.0
                    for tau in range(t_k - window_l, t_k):
                        th_a = theta_history[a][tau]
                        th_b = theta_history[b][tau]
                        xi = xi_path[tau]
                        f = f_path[tau]
                        s2 = sigma2_path[tau]
                        th_star = f / s2

                        emp_diff_sum += factored_score_increment(th_a, th_b, xi)
                        loss_a = 0.5 * s2 * ((th_a - th_star) ** 2)
                        loss_b = 0.5 * s2 * ((th_b - theta_history[b][tau] + th_b - th_star) ** 2)
                        loss_b = 0.5 * s2 * ((th_b - th_star) ** 2)
                        true_loss_diff_sum += loss_a - loss_b

                    if abs(emp_diff_sum / window_l - true_loss_diff_sum / window_l) > kappa:
                        any_eval_violated = True

        if any_eval_violated:
            trajectory_violations += 1

    empirical_rate = trajectory_violations / num_trajectories
    assert empirical_rate <= delta


def test_4b_theorem_8_4d_realized_sequence_guard_and_post_flip_negative_test():
    """Check 4B: Simulates L=2400 realized shock sequence, fires Guard 2 under exact kappa_search, and verifies post-flip loss reversal."""
    m_total = 2
    xi_max = 0.5
    eta_margin = 0.2
    lambda_max = (1.0 - eta_margin) / xi_max
    delta = 0.05
    delta_1 = anytime_delta(delta, 1)
    window_l = 2400
    tau_dwell = 5
    tau_hyst = 0.015
    potential_jump = 0.0
    alpha = 0.05

    kappa_search = all_pairs_search_penalty(m_total, window_l, delta_1, lambda_max, xi_max)["kappa_search"]

    seed = 0x2468ACE1

    def rand() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / (2 ** 32)

    f_hist = 0.4
    sigma2 = 0.25
    theta_star_hist = f_hist / sigma2

    f_hat_ema = 0.4
    empirical_score_diff_sum = 0.0
    population_loss_diff_sum = 0.0

    for _tau in range(window_l):
        theta_old = 0.0
        theta_new = clamp(f_hat_ema / sigma2, -lambda_max, lambda_max)
        xi_tau = +0.5 if rand() < 0.9 else -0.5

        empirical_score_diff_sum += factored_score_increment(theta_old, theta_new, xi_tau)
        loss_old = 0.5 * sigma2 * ((theta_old - theta_star_hist) ** 2)
        loss_new = 0.5 * sigma2 * ((theta_new - theta_star_hist) ** 2)
        population_loss_diff_sum += loss_old - loss_new

        f_hat_ema = (1.0 - alpha) * f_hat_ema + alpha * xi_tau

    emp_delta_l = empirical_score_diff_sum / window_l
    pop_delta_l = population_loss_diff_sum / window_l

    assert abs(emp_delta_l - pop_delta_l) <= kappa_search

    delta_g_tj = emp_delta_l - kappa_search - potential_jump / tau_dwell
    assert delta_g_tj > tau_hyst
    assert pop_delta_l > tau_hyst

    f_fut = -0.4
    theta_star_fut = f_fut / sigma2
    fut_loss_old_sum = 0.0
    fut_loss_new_sum = 0.0

    for _step in range(tau_dwell):
        theta_new = clamp(f_hat_ema / sigma2, -lambda_max, lambda_max)
        loss_old = 0.5 * sigma2 * ((0.0 - theta_star_fut) ** 2)
        loss_new = 0.5 * sigma2 * ((theta_new - theta_star_fut) ** 2)
        fut_loss_old_sum += loss_old
        fut_loss_new_sum += loss_new

        xi_fut = -0.5 if rand() < 0.9 else +0.5
        f_hat_ema = (1.0 - alpha) * f_hat_ema + alpha * xi_fut

    fut_avg_gain = (fut_loss_old_sum - fut_loss_new_sum) / tau_dwell
    d_regime = abs(fut_avg_gain - pop_delta_l)

    assert d_regime > tau_hyst
    assert fut_loss_new_sum + potential_jump > fut_loss_old_sum


def test_5a_factored_score_identity_and_predictable_variation_envelope():
    """Check 5A: Verifies factored score identity, falsifies naive xi_max^2*(a-b)^2 proxy, and checks B_lip^2*(a-b)^2."""
    xi_max = 0.5
    eta_margin = 0.2
    lambda_max = (1.0 - eta_margin) / xi_max
    b_lip = xi_max + lambda_max * (xi_max ** 2)

    for a in (-1.4, -0.3, 0.0, 0.7, 1.6):
        for b in (-1.5, -0.1, 0.4, 1.2, 1.6):
            for xi in (-0.5, -0.2, 0.15, 0.5):
                s_a = 0.5 * a * a * xi * xi - a * xi
                s_b = 0.5 * b * b * xi * xi - b * xi
                factored = factored_score_increment(a, b, xi)
                assert abs((s_a - s_b) - factored) <= 1e-14

    a = -1.6
    b = -1.5
    xs = [0.5, -0.25]
    ps = [0.5, 0.5]
    z_vals = [factored_score_increment(a, b, xi) for xi in xs]
    z_mean = _mean(z_vals, ps)
    true_cond_var = _mean([(z - z_mean) ** 2 for z in z_vals], ps)
    naive_proxy = ((a - b) ** 2) * (xi_max ** 2)
    rigorous_pred_bound = ((a - b) ** 2) * (b_lip ** 2)

    assert true_cond_var > naive_proxy
    assert true_cond_var <= rigorous_pred_bound + 1e-14


def test_5b_gaussian_mixture_supermartingale_and_self_normalized_kappa_pred():
    """Check 5B: Tests Gaussian-mixture supermartingale E[M_bar] <= 1.05 and all-pairs self-normalized kappa_pred."""
    seed = 0x13572468

    def rand() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / (2 ** 32)

    alphas = [0.0, 0.1, 0.25, 0.5]
    m_total = len(alphas)
    xi_max = 0.5
    eta_margin = 0.2
    lambda_max = (1.0 - eta_margin) / xi_max
    b_lip = xi_max + lambda_max * (xi_max ** 2)
    window_l = 60
    rho = window_l * (b_lip ** 2) * 0.25
    delta = 0.05
    delta_1 = anytime_delta(delta, 1)
    num_trials = 400

    mixture_sum = 0.0
    violations = 0

    for _tr in range(num_trials):
        ema = [0.0] * m_total
        sum_x = [[0.0] * m_total for _ in range(m_total)]
        sum_sq_diff = [[0.0] * m_total for _ in range(m_total)]

        for _t in range(window_l):
            f_t = 0.08
            step_amp = 0.35
            sigma2_t = f_t * f_t + step_amp * step_amp
            xi_t = f_t + (-step_amp if rand() < 0.5 else +step_amp)
            theta_star = f_t / sigma2_t

            thetas = [0.0 if g == 0 else clamp(ema[g] / sigma2_t, -lambda_max, lambda_max) for g in range(m_total)]

            for a in range(m_total):
                for b in range(a + 1, m_total):
                    th_a = thetas[a]
                    th_b = thetas[b]
                    d_tau = factored_score_increment(th_a, th_b, xi_t)
                    true_diff = 0.5 * sigma2_t * ((th_a - theta_star) ** 2) - 0.5 * sigma2_t * ((th_b - theta_star) ** 2)
                    x_tau = d_tau - true_diff
                    sum_x[a][b] += x_tau
                    sum_sq_diff[a][b] += (th_a - th_b) ** 2

            for g in range(1, m_total):
                ema[g] = (1.0 - alphas[g]) * ema[g] + alphas[g] * xi_t

        trial_violated = False
        for a in range(m_total):
            for b in range(a + 1, m_total):
                s_val = sum_x[a][b]
                pred_res = predictable_variation_penalty(
                    m_total, window_l, delta_1, lambda_max, xi_max, sum_sq_diff[a][b], rho=rho
                )
                v_obs = pred_res["v_obs"]
                m_bar = math.sqrt(rho / (rho + v_obs)) * math.exp((s_val * s_val) / (2.0 * (rho + v_obs)))
                if a == 0 and b == 1:
                    mixture_sum += m_bar
                if abs(s_val) / window_l > pred_res["kappa_pred"]:
                    trial_violated = True

        if trial_violated:
            violations += 1

    avg_m_bar = mixture_sum / num_trials
    emp_viol_rate = violations / num_trials
    assert avg_m_bar <= 1.05
    assert emp_viol_rate <= delta


def test_theorem_8_4b_switching_frequency_and_theorem_9_1_four_stage_composition():
    """Verifies Theorem 8.4b symmetric dwell reset switching bound and Theorem 9.1 4-stage composition (Findings 1 & 2)."""
    router = GuardedRouter(tau_dwell=4, tau_hyst=0.01)
    horizon_n = 50
    for t in range(horizon_n):
        if t % 2 == 0:
            router.step(t, active_cert_valid=True, collapse_gain_to_null=0.0, candidate_gains={"G_ema": (True, 0.05)})
        else:
            router.step(t, active_cert_valid=False, collapse_gain_to_null=0.1, candidate_gains={})

    assert len(router.switch_epochs) <= max_switches_bound(horizon_n, tau_dwell=4)

    # Theorem 9.1 composition (Boundary-clamped regime): xi_max = (1 - 0.2) / 0.8 = 1.0, so D_bdd = [-1.0, 1.0] x [0.20, 1.00]
    cert_ok = compose_end_to_end_certificate(
        f_hat=1.25,       # exceeds xi_max=1.0 -> clamped to 1.0 in Stage 1
        v_hat=0.45,
        eps_moment=0.005,
        nu_subg=0.01,
        delta=0.05,
        sigma_min2=0.20,
        sigma2_true=0.45,
        lambda_max=0.8,
        eta_margin=0.2,
        third_abs_moment=0.04,
    )
    assert cert_ok["status"] == "CERTIFIED"
    assert abs(cert_ok["f_clamped"] - 1.0) <= 1e-12
    assert abs(cert_ok["m_in_max"] - math.hypot(2.0, 1.0 - 0.20)) <= 1e-12
    assert cert_ok["raw_ratio_status"] == "CONF_RADIUS"
    assert cert_ok["projected_status"] == "EVENT"
    assert cert_ok["theta_raw"] > 0.8
    assert abs(cert_ok["theta_clamped"] - 0.8) <= 1e-12
    assert abs(cert_ok["m_out_max"] - 1.6) <= 1e-12
    assert cert_ok["r_out"] <= cert_ok["m_out_max"]
    # Verify exact separation of squared control error vs. constrained quadratic objective regret
    expected_bnd_slope = max(0.0, 1.0 - 0.45 * 0.8)  # 0.64 > 0
    assert abs(cert_ok["boundary_slope_max"] - expected_bnd_slope) <= 1e-12
    assert cert_ok["boundary_clipping_regret_bound"] > 0.0
    assert abs(
        cert_ok["constrained_quadratic_objective_regret_bound"]
        - (cert_ok["quadratic_control_error_bound"] + cert_ok["boundary_clipping_regret_bound"])
    ) <= 1e-12
    assert abs(
        cert_ok["total_exact_log_suboptimality_bound"]
        - (cert_ok["constrained_quadratic_objective_regret_bound"] + cert_ok["surrogate_stage2b_remainder"])
    ) <= 1e-12

    # Verify exact algebraic identity: Q_quad(theta_quad*) - Q_quad(theta) == (sigma^2/2)*(theta - theta_quad*)^2 + g_bnd*|theta - theta_quad*|
    for f_true in (-0.95, -0.36, 0.0, 0.20, 0.85):
        sigma2_val = 0.45
        l_max = 0.8
        theta_quad_star = clamp(f_true / sigma2_val, -l_max, l_max)
        g_bnd = max(0.0, abs(f_true) - sigma2_val * l_max)
        for theta_test in (-0.8, -0.35, 0.0, 0.4, 0.8):
            q_star = f_true * theta_quad_star - 0.5 * sigma2_val * (theta_quad_star ** 2)
            q_test = f_true * theta_test - 0.5 * sigma2_val * (theta_test ** 2)
            exact_rhs = 0.5 * sigma2_val * ((theta_test - theta_quad_star) ** 2) + g_bnd * abs(theta_test - theta_quad_star)
            assert abs((q_star - q_test) - exact_rhs) <= 1e-14

    # Interior regime (f_true_abs_max <= sigma2_true * lambda_max => boundary_slope_max == 0, objective regret == control error)
    cert_interior = compose_end_to_end_certificate(
        f_hat=0.15,
        v_hat=0.45,
        eps_moment=0.005,
        nu_subg=0.01,
        delta=0.05,
        sigma_min2=0.20,
        sigma2_true=0.45,
        lambda_max=0.8,
        eta_margin=0.2,
        third_abs_moment=0.04,
        f_true_abs_max=0.30,  # 0.30 <= 0.45 * 0.8 = 0.36 -> interior optimum
    )
    assert cert_interior["status"] == "CERTIFIED"
    assert cert_interior["boundary_slope_max"] == 0.0
    assert cert_interior["boundary_clipping_regret_bound"] == 0.0
    assert abs(
        cert_interior["constrained_quadratic_objective_regret_bound"]
        - cert_interior["quadratic_control_error_bound"]
    ) <= 1e-15

    # Stage 2 domain violation (v_clamped - 2*r_in < sigma_min2) triggers fallback to G_null
    cert_fail = compose_end_to_end_certificate(
        f_hat=0.12,
        v_hat=0.21,
        eps_moment=0.02,
        nu_subg=0.05,
        delta=0.05,
        sigma_min2=0.20,
        sigma2_true=0.21,
        lambda_max=0.8,
        eta_margin=0.2,
        third_abs_moment=0.04,
    )
    assert cert_fail["status"] == "UNESTABLISHED"
    assert cert_fail["code"] == "DOMAIN_VIOLATION"
    assert cert_fail["fallback_graph"] == "G_null"
    assert cert_fail["fallback_exposure"] == 0.0

    # Stage 1 invalid moment box (xi_max^2 < sigma_min2) triggers INVALID_MOMENT_BOX fallback
    cert_bad_box = compose_end_to_end_certificate(
        f_hat=0.12,
        v_hat=0.25,
        eps_moment=0.005,
        nu_subg=0.01,
        delta=0.05,
        sigma_min2=0.50,
        sigma2_true=0.50,
        lambda_max=1.6,
        eta_margin=0.2,
        third_abs_moment=0.04,
        xi_max=0.5,       # lambda_max * xi_max = 0.8 <= 1 - 0.2, but xi_max^2 = 0.25 < sigma_min2 = 0.50
    )
    assert cert_bad_box["status"] == "UNESTABLISHED"
    assert cert_bad_box["code"] == "INVALID_MOMENT_BOX"

    # Entry-point safety-cap violation (lambda_max * xi_max = 1.0 > 1 - eta_margin = 0.8) triggers SAFETY_CAP_VIOLATION
    cert_bad_safety = compose_end_to_end_certificate(
        f_hat=0.12,
        v_hat=0.45,
        eps_moment=0.005,
        nu_subg=0.01,
        delta=0.05,
        sigma_min2=0.20,
        sigma2_true=0.45,
        lambda_max=1.0,
        eta_margin=0.2,
        third_abs_moment=0.04,
        xi_max=1.0,       # 1.0 * 1.0 = 1.0 > 1 - 0.2 = 0.8 -> violates bounded-shock survival contract
    )
    assert cert_bad_safety["status"] == "UNESTABLISHED"
    assert cert_bad_safety["code"] == "SAFETY_CAP_VIOLATION"
    assert cert_bad_safety["fallback_graph"] == "G_null"
    assert cert_bad_safety["fallback_exposure"] == 0.0

    # Invalid caller-supplied moment envelope (f_true_abs_max > xi_max) triggers INVALID_MOMENT_ENVELOPE
    cert_bad_envelope = compose_end_to_end_certificate(
        f_hat=0.12,
        v_hat=0.45,
        eps_moment=0.005,
        nu_subg=0.01,
        delta=0.05,
        sigma_min2=0.20,
        sigma2_true=0.45,
        lambda_max=0.8,
        eta_margin=0.2,
        third_abs_moment=0.04,
        xi_max=1.0,
        f_true_abs_max=1.5,  # exceeds xi_max = 1.0
    )
    assert cert_bad_envelope["status"] == "UNESTABLISHED"
    assert cert_bad_envelope["code"] == "INVALID_MOMENT_ENVELOPE"

