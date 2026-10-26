"""Tests for Sections 3 and 7: CD(Sigma) Multicategory, Proposition 3.3, Lemma 7.1a, and Proposition 7.2 (Checks 2A, 3A, 3B, 6A, 6C, and Issue A/Finding 1-2 regressions)."""
from __future__ import annotations
import math
import pytest
from tlq.hypergraph import (
    Sort,
    WEALTH_SORT,
    DRIFT_SORT,
    make_var_sort,
    Hyperedge,
    verify_shift_register_impossibility,
)
from tlq.certificates import (
    MomentCert,
    sub_gaussian_event_radius,
    compact_moment_box_diameter,
    clamp_moment_pair_to_compact_box,
    moment_to_confidence_radius,
    moment_to_event,
    event_to_quadratic_regret_bound,
    exact_kelly_hessian_spectral_norm,
    kelly_ratio_certificate_pushforward,
    project_kelly_exposure_certificate,
    push_scalar,
    push_split_reconverge,
)
from tlq.vertical import (
    constrained_action_gap_bound,
    exact_log_objective_gap_bounds,
)


def test_2a_shift_register_impossibility_and_cd_sigma_port_grafting():
    """Check 2A: Verifies (10,5) and (8,6) at K=2, u_K > v_K >= 2 across all 127 words (0<=K<=6), and associative port-grafting."""
    res = verify_shift_register_impossibility(max_length=6)
    assert res["cartesian_target"] == (5.0, 6.0)
    assert res["order_sum_then_prod"] == (10.0, 5.0)
    assert res["order_prod_then_sum"] == (8.0, 6.0)
    assert res["words_checked"] == 127
    assert res["invariant_holds"] is True

    # Verify associative port-grafting (f o_i g) o_{i+j-1} h == f o_i (g o_j h) in CD(Sigma)
    var_sort = make_var_sort(0.05)
    f_edge = Hyperedge(
        "kelly_pos",
        (WEALTH_SORT, DRIFT_SORT, var_sort),
        (DRIFT_SORT,),
        lambda w, f, v: (w * (f / v),),
    )
    g_edge = Hyperedge(
        "blend_drift",
        (DRIFT_SORT, DRIFT_SORT),
        (DRIFT_SORT,),
        lambda f1, f2: (0.6 * f1 + 0.4 * f2,),
    )
    h_edge = Hyperedge(
        "scale_drift",
        (DRIFT_SORT,),
        (DRIFT_SORT,),
        lambda f0: (1.5 * f0,),
    )

    left_assoc = f_edge.graft(g_edge, 1).graft(h_edge, 1)
    right_assoc = f_edge.graft(g_edge.graft(h_edge, 0), 1)
    val_left = left_assoc.eval(100.0, 0.10, 0.05, 0.25)
    val_right = right_assoc.eval(100.0, 0.10, 0.05, 0.25)
    assert abs(val_left[0] - val_right[0]) <= 1e-14


def test_issue_a_sort_contract_rejects_same_name_incompatible_validators():
    """Issue A Regression: Same-named sorts with different domain validators or specs must be rejected at composition time."""
    var_strict = make_var_sort(0.20)
    var_loose = make_var_sort(0.05)
    assert var_strict.name == var_loose.name == "Var"
    assert var_strict != var_loose

    producer_loose = Hyperedge("est_var", (DRIFT_SORT,), (var_loose,), lambda x: (max(0.05, x * x),))
    consumer_strict = Hyperedge("inv_var", (var_strict,), (DRIFT_SORT,), lambda v: (1.0 / v,))

    with pytest.raises(TypeError):
        consumer_strict.graft(producer_loose, 0)

    with pytest.raises(TypeError):
        consumer_strict.compose(producer_loose)

    # Even if domain_spec is identical, distinct domain_check callables with the same name are rejected
    custom_a = Sort("Custom", lambda x: x > 0.0, domain_spec="pos")
    custom_b = Sort("Custom", lambda x: x > 1.0, domain_spec="pos")
    edge_a = Hyperedge("ea", (custom_a,), (custom_a,), lambda x: (x,))
    edge_b = Hyperedge("eb", (custom_b,), (custom_b,), lambda x: (x,))
    with pytest.raises(TypeError):
        edge_b.graft(edge_a, 0)
    with pytest.raises(TypeError):
        edge_b.compose(edge_a)


def test_3a_two_stage_ratio_conf_radius_and_projected_event_certificate():
    """Check 3A (Findings 1 & 2 Repaired): Verifies Stage A (`CONF_RADIUS` on unclamped ratio f/v),
    Stage B (`EVENT` on projected exposure in [-Lambda_max, Lambda_max]), boundary target v_star = sigma_min2,
    and gradient/Hessian spectral norm bounds across 2,000 balls.
    """
    sigma_min2 = 0.2
    lambda_max = 1.6

    near_floor = kelly_ratio_certificate_pushforward(0.3, 0.25, 0.04, sigma_min2)
    assert near_floor["status"] == "UNESTABLISHED"
    assert near_floor["code"] == "DOMAIN_VIOLATION"
    assert project_kelly_exposure_certificate(near_floor, lambda_max)["status"] == "UNESTABLISHED"

    # Boundary target v_star = sigma_min2
    boundary_v_star = sigma_min2
    f_star_bound = 0.3
    r_bound = 0.03
    f_hat_bound = 0.3
    v_hat_bound = boundary_v_star + 0.02
    assert math.hypot(f_hat_bound - f_star_bound, v_hat_bound - boundary_v_star) <= r_bound
    boundary_push = kelly_ratio_certificate_pushforward(f_hat_bound, v_hat_bound, r_bound, sigma_min2)
    assert boundary_push["status"] == "UNESTABLISHED"
    assert boundary_push["code"] == "DOMAIN_VIOLATION"

    # Unclamped ratio exceeding [-Lambda_max, Lambda_max]: Stage A is CONF_RADIUS (unclamped), Stage B projects to EVENT
    high_ratio_raw = kelly_ratio_certificate_pushforward(0.9, 0.35, 0.02, sigma_min2, delta=0.05)
    assert high_ratio_raw["status"] == "CONF_RADIUS"
    assert "m_max" not in high_ratio_raw
    assert high_ratio_raw["theta_raw"] > lambda_max

    high_ratio_proj = project_kelly_exposure_certificate(high_ratio_raw, lambda_max)
    assert high_ratio_proj["status"] == "EVENT"
    assert abs(high_ratio_proj["theta_clamped"] - lambda_max) <= 1e-12
    assert abs(high_ratio_proj["m_max"] - 2.0 * lambda_max) <= 1e-12
    assert high_ratio_proj["r_out"] <= 2.0 * lambda_max

    # Finding 1 check: moment_to_event requires explicit finite m_max, while moment_to_confidence_radius does not
    m_cert = MomentCert(eps=0.01, variance=0.04, nu_subg=0.1, dimension=2)
    conf_cert = moment_to_confidence_radius(m_cert, delta=0.05)
    assert conf_cert.status == "CONF_RADIUS"
    missing_bound = moment_to_event(m_cert, delta=0.05, m_max=None)
    assert missing_bound.status == "UNESTABLISHED"
    assert missing_bound.code == "MISSING_AS_BOUND"
    box_diam = compact_moment_box_diameter(xi_max=0.5, sigma_min2=0.05)
    ev_cert = moment_to_event(m_cert, delta=0.05, m_max=box_diam)
    assert ev_cert.status == "EVENT"
    assert abs(ev_cert.m_max - box_diam) <= 1e-14

    seed = 987654321

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / (2 ** 32)

    for _ in range(2000):
        f_star = -0.8 + 1.6 * rnd()
        v_star = 0.4 + 1.2 * rnd()
        r = 0.01 + 0.06 * rnd()

        angle = 2.0 * math.pi * rnd()
        dist = r * rnd()
        f_hat = f_star + dist * math.cos(angle)
        v_hat = v_star + dist * math.sin(angle)

        raw_cert = kelly_ratio_certificate_pushforward(f_hat, v_hat, r, sigma_min2)
        if raw_cert["status"] == "UNESTABLISHED":
            continue
        assert raw_cert["status"] == "CONF_RADIUS"

        for rad_scale in (0.0, 0.5, 1.0, 2.0):
            for step in range(8):
                ang = step * (math.pi / 4.0)
                fb = f_hat + rad_scale * r * math.cos(ang)
                vb = v_hat + rad_scale * r * math.sin(ang)
                grad_norm = math.hypot(1.0 / vb, fb / (vb * vb))
                hess_norm = exact_kelly_hessian_spectral_norm(fb, vb)
                assert raw_cert["L_hat_o"] >= grad_norm - 1e-12
                assert raw_cert["M_hat_Ho"] >= hess_norm - 1e-12

        actual_raw_err = abs(f_hat / v_hat - f_star / v_star)
        assert actual_raw_err <= raw_cert["r_raw"] + 1e-12

        proj_cert = project_kelly_exposure_certificate(raw_cert, lambda_max)
        assert proj_cert["status"] == "EVENT"
        theta_star_clamped = max(-lambda_max, min(lambda_max, f_star / v_star))
        actual_proj_err = abs(proj_cert["theta_clamped"] - theta_star_clamped)
        assert actual_proj_err <= proj_cert["r_out"] + 1e-12
        assert actual_proj_err <= proj_cert["m_max"] + 1e-12


def test_3b_lemma_7_1a_sub_gaussian_net_radius_repair():
    """Check 3B: Refutes v2.2.1 coefficient 2*nu*sqrt(d) and verifies repaired (1/2)-net radius r_delta^{subG}."""
    eps = 0.05
    nu = 0.25
    delta = 0.05

    for d in (1, 2, 5, 10, 50):
        res = sub_gaussian_event_radius(eps, d, nu, delta)
        exact_net_t = res["exact_net_radius"] - eps
        refuted_v221 = 2.0 * nu * math.sqrt(d) + 2.0 * nu * math.sqrt(math.log(1.0 / delta))
        repaired_add = res["additive_upper_bound"] - eps

        assert exact_net_t > refuted_v221
        assert exact_net_t <= repaired_add + 1e-14
        assert res["exact_net_radius"] <= res["additive_upper_bound"] + 1e-14


def test_6a_canonical_specification_functions_across_200_randomized_inputs():
    """Check 6A: Verifies exact formula parity across 200 randomized parameter tuples."""
    seed = 0x99887766

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / (2 ** 32)

    for _ in range(200):
        lambda_max = 0.5 + 2.0 * rnd()
        m3 = 0.01 + 0.5 * rnd()
        sigma2 = 0.1 + 1.5 * rnd()
        eta = 0.1 + 0.7 * rnd()

        expected_act = (lambda_max * lambda_max * m3) / (sigma2 * eta)
        assert abs(constrained_action_gap_bound(lambda_max, m3, sigma2, eta) - expected_act) <= 1e-14

        obj_bounds = exact_log_objective_gap_bounds(lambda_max, m3, sigma2, eta)
        expected_var = (((lambda_max * lambda_max * m3) / eta) ** 2) / sigma2
        expected_taylor = (2.0 * (lambda_max ** 3) * m3) / (3.0 * (eta ** 3))
        assert abs(obj_bounds["variational_bound"] - expected_var) <= 1e-14
        assert abs(obj_bounds["taylor_bound"] - expected_taylor) <= 1e-14
        assert abs(obj_bounds["combined_bound"] - min(expected_var, expected_taylor)) <= 1e-14

        eps = 0.1 * rnd()
        dim = 1 + int(rnd() * 20)
        nu = 0.05 + 0.5 * rnd()
        delta = 0.001 + 0.1 * rnd()
        rad = sub_gaussian_event_radius(eps, dim, nu, delta)
        assert rad["exact_net_radius"] <= rad["additive_upper_bound"] + 1e-14

        reg_bound = event_to_quadratic_regret_bound(0.5 * sigma2, rad["exact_net_radius"], delta, 2.0 * lambda_max)
        expected_reg = 0.5 * sigma2 * ((1.0 - delta) * (rad["exact_net_radius"] ** 2) + delta * ((2.0 * lambda_max) ** 2))
        assert abs(reg_bound - expected_reg) <= 1e-14


def test_6c_push_scalar_and_push_split_reconverge_fourth_moment_audit():
    """Check 6C: Falsifies unguarded Gaussian variance on skewed two-point error and verifies Cauchy-Schwarz 4th-moment bound."""
    p = 0.8
    q = 0.2
    e1 = math.sqrt(q / p)   # +0.5
    e2 = -math.sqrt(p / q)  # -2.0
    v = p * (e1 ** 2) + q * (e2 ** 2)
    m3 = p * (e1 ** 3) + q * (e2 ** 3)
    m4 = p * (e1 ** 4) + q * (e2 ** 4)
    kurtosis_factor = m4 / (v * v)

    j = 1.0
    h = 1.0
    y1 = j * e1 + 0.5 * h * (e1 ** 2)
    y2 = j * e2 + 0.5 * h * (e2 ** 2)
    y_mean = p * y1 + q * y2
    true_var_y = p * ((y1 - y_mean) ** 2) + q * ((y2 - y_mean) ** 2)

    pushed = push_scalar({"eps": 0.0, "v": v, "delta": 0.05}, j, 0.0, h, kurtosis_factor)
    assert m3 < 0.0
    assert true_var_y < pushed["v_gaussian_quad"]
    assert true_var_y <= pushed["v_general_fourth_moment"] + 1e-12

    y1_pos = j * (-e1) + 0.5 * h * ((-e1) ** 2)
    y2_pos = j * (-e2) + 0.5 * h * ((-e2) ** 2)
    y_mean_pos = p * y1_pos + q * y2_pos
    true_var_y_pos = p * ((y1_pos - y_mean_pos) ** 2) + q * ((y2_pos - y_mean_pos) ** 2)
    assert true_var_y_pos > pushed["v_gaussian_quad"]
    assert true_var_y_pos <= pushed["v_general_fourth_moment"] + 1e-12

    split_res = push_split_reconverge(
        source_var=0.25,
        j1=1.2,
        j2=0.8,
        correlation=0.75,
        hessian_norm=0.9,
        kurtosis_factor=kurtosis_factor,
    )
    assert split_res["prop_true_linear_var"] > split_res["naive_tree_var"]
    assert split_res["prop_certified_var"] >= split_res["prop_true_linear_var"]
    assert split_res["prop_fourth_moment_certified_var"] > split_res["prop_certified_var"]
