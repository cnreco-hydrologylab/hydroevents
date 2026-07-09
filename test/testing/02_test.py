# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:10:39 2026

@author: s.ortenzi
"""

import numpy as np
import pandas as pd

import pytest

from hydroevents import compute_mrc
from hydroevents.mrc import (
    extract_recession_segments,
    identify_inflection_points,
    transform_segments_to_log_space,
    filter_segments_by_linearity,
    keep_longest_segment_per_decreasing_branch,
    interpolate_valid_segments,
    align_overlapping_segments,
    apply_global_shift,
    get_lh_k_from_mrc,
    fit_maillet_model,
)


def make_recession_with_tail(alpha=0.05, q0=10.0, n_tail=25):
    dates = pd.date_range("2000-01-01", periods=n_tail + 4, freq="D")

    q_head = np.array([
        q0 * 1.35,
        q0 * 1.25,
        q0 * 1.05,
        q0,
    ])

    t = np.arange(n_tail)
    q_tail = q0 * np.exp(-alpha * t)

    q = np.concatenate([q_head, q_tail])

    return pd.DataFrame({
        "Date": dates,
        "Q": q
    })

def make_multiple_recessions_same_alpha(alpha=0.05):
    all_parts = []
    current_date = pd.Timestamp("2000-01-01")

    for q0 in [20.0, 18.0, 16.0, 14.0]:
        df_part = make_recession_with_tail(
            alpha=alpha,
            q0=q0,
            n_tail=25,
        )

        df_part["Date"] = pd.date_range(
            current_date,
            periods=len(df_part),
            freq="D"
        )

        all_parts.append(df_part)
        current_date = df_part["Date"].iloc[-1] + pd.Timedelta(days=5)

    return pd.concat(all_parts, ignore_index=True)

def test_mrc_estimates_known_alpha():
    expected_alpha = 0.05

    df = make_multiple_recessions_same_alpha(alpha=expected_alpha)

    results = compute_mrc(
        df,
        min_recession_length=10,
        recession_tolerance=0.0,
        min_baseflow_length=5,
        skip_after_inflection=2,
        r2_threshold=0.95,
    )

    fit = results["fit_results"]

    print("\nEstimated fit:")
    print(fit)

    assert fit is not None
    assert np.isclose(fit["alpha"], expected_alpha, atol=0.02)
    assert fit["R2"] > 0.90

def test_r2_filter_discards_poorly_linear_segments():
    dates = pd.date_range("2000-01-01", periods=20, freq="D")

    good_q = 10 * np.exp(-0.05 * np.arange(20))
    bad_q = np.array([
        10.0, 8.0, 9.0, 6.0, 7.5,
        5.0, 5.8, 4.0, 4.5, 3.5,
        3.8, 3.0, 3.4, 2.7, 3.0,
        2.4, 2.8, 2.2, 2.5, 2.0
    ])

    log_results = [
        {
            "baseflow_segment_ln": pd.DataFrame({
                "Date": dates,
                "Q": good_q,
                "ln_Q": np.log(good_q),
                "t": np.arange(20),
            })
        },
        {
            "baseflow_segment_ln": pd.DataFrame({
                "Date": dates,
                "Q": bad_q,
                "ln_Q": np.log(bad_q),
                "t": np.arange(20),
            })
        },
    ]

    valid_segments, discarded_segments = filter_segments_by_linearity(
        log_results,
        r2_min=0.95,
    )

    print("\nValid segments:", len(valid_segments))
    print("Discarded segments:", len(discarded_segments))

    assert len(valid_segments) == 1
    assert len(discarded_segments) == 1
    assert valid_segments[0]["R2"] >= 0.95
    assert discarded_segments[0]["R2"] < 0.95


def test_recession_tolerance_produces_longer_segments():
    dates = pd.date_range("2000-01-01", periods=12, freq="D")

    q = np.array([
        10.0,
        9.8,
        9.6,
        9.7,
        9.4,
        9.2,
        9.0,
        9.1,
        8.8,
        8.6,
        8.4,
        8.2,
    ])

    df = pd.DataFrame({"Date": dates, "Q": q})

    strict_segments = extract_recession_segments(
        df,
        min_length=3,
        recession_tolerance=0.0,
    )

    tolerant_segments = extract_recession_segments(
        df,
        min_length=3,
        recession_tolerance=0.15,
    )

    strict_max_len = max(len(s) for s in strict_segments)
    tolerant_max_len = max(len(s) for s in tolerant_segments)

    print("\nStrict segment lengths:", [len(s) for s in strict_segments])
    print("Tolerant segment lengths:", [len(s) for s in tolerant_segments])

    assert tolerant_max_len > strict_max_len

def make_multiple_recessions():
    all_parts = []
    start_date = pd.Timestamp("2000-01-01")

    alphas = [0.02, 0.04, 0.06, 0.20]
    q0_values = [20.0, 18.0, 16.0, 14.0]

    current_date = start_date

    for alpha, q0 in zip(alphas, q0_values):
        df_part = make_recession_with_tail(
            alpha=alpha,
            q0=q0,
            n_tail=25,
        )

        df_part["Date"] = pd.date_range(
            current_date,
            periods=len(df_part),
            freq="D"
        )

        all_parts.append(df_part)

        current_date = df_part["Date"].iloc[-1] + pd.Timedelta(days=5)

    df = pd.concat(all_parts, ignore_index=True)

    return df, alphas

def test_k_mean_and_median_with_multiple_recessions():
    df, expected_alphas = make_multiple_recessions()

    results = compute_mrc(
        df,
        min_recession_length=10,
        recession_tolerance=0.0,
        min_baseflow_length=5,
        skip_after_inflection=2,
        r2_threshold=0.95,
    )

    k_estimates = results["k_estimates"]

    print("\nK estimates:")
    print(k_estimates)

    assert k_estimates["mrc"] is not None
    assert k_estimates["mean"] is not None
    assert k_estimates["median"] is not None

    alpha_mean = k_estimates["mean"]["alpha"]
    alpha_median = k_estimates["median"]["alpha"]

    print("\nExpected alphas:")
    print(expected_alphas)

    print("\nEstimated alpha mean:", alpha_mean)
    print("Estimated alpha median:", alpha_median)

    assert alpha_mean > 0
    assert alpha_median > 0

    assert not np.isclose(alpha_mean, alpha_median)

    assert 0 < k_estimates["mean"]["k_day"] < 1
    assert 0 < k_estimates["median"]["k_day"] < 1

def test_identify_inflection_points_finds_head_tail_transition():
    seg = make_recession_with_tail(alpha=0.05, q0=10.0, n_tail=25)
    seg["decreasing_segment_id"] = 0

    results = identify_inflection_points(
        [seg], min_baseflow_length=5, skip_after_inflection=2
    )

    assert len(results) == 1
    baseflow_segment = results[0]["baseflow_segment"]
    assert baseflow_segment.iloc[0]["Q"] == pytest.approx(10.0)
    assert len(baseflow_segment) == 25


def test_identify_inflection_points_discards_short_tail_and_short_input():
    seg = make_recession_with_tail(alpha=0.05, q0=10.0, n_tail=3)
    seg["decreasing_segment_id"] = 0
    results = identify_inflection_points(
        [seg], min_baseflow_length=5, skip_after_inflection=2
    )
    assert results == []

    tiny = seg.iloc[:2]
    results_tiny = identify_inflection_points(
        [tiny], min_baseflow_length=5, skip_after_inflection=2
    )
    assert results_tiny == []


def test_identify_inflection_points_validates_skip_parameter():
    seg = make_recession_with_tail(alpha=0.05, q0=10.0, n_tail=25)
    seg["decreasing_segment_id"] = 0

    with pytest.raises(ValueError):
        identify_inflection_points(
            [seg], min_baseflow_length=5, skip_after_inflection=5
        )

    with pytest.raises(ValueError):
        identify_inflection_points(
            [seg], min_baseflow_length=5, skip_after_inflection=3
        )

def test_transform_segments_to_log_space_computes_ln_q_and_t():
    dates = pd.date_range("2000-01-01", periods=6, freq="D")
    Q = np.array([10.0, 8.0, 6.0, -1.0, 4.0, 3.0])  # un valore negativo da scartare
    baseflow_segment = pd.DataFrame({"Date": dates, "Q": Q})

    baseflow_results = [
        {"decreasing_segment_id": 0, "inflection_point": None, "baseflow_segment": baseflow_segment}
    ]
    log_results = transform_segments_to_log_space(baseflow_results)

    assert len(log_results) == 1
    seg_ln = log_results[0]["baseflow_segment_ln"]

    assert len(seg_ln) == 5
    assert np.allclose(seg_ln["ln_Q"], np.log(seg_ln["Q"]))
    assert seg_ln["t"].tolist() == [0, 1, 2, 4, 5]


def test_transform_segments_to_log_space_discards_non_positive_segments():
    dates = pd.date_range("2000-01-01", periods=6, freq="D")

    all_negative = pd.DataFrame({"Date": dates, "Q": -np.arange(1, 7, dtype=float)})
    single_positive = pd.DataFrame({"Date": dates, "Q": [10.0, -1, -2, -3, -4, -5]})

    baseflow_results = [
        {"decreasing_segment_id": 0, "inflection_point": None, "baseflow_segment": all_negative},
        {"decreasing_segment_id": 1, "inflection_point": None, "baseflow_segment": single_positive},
    ]
    log_results = transform_segments_to_log_space(baseflow_results)

    assert log_results == []

def test_filter_segments_by_linearity_discards_positive_slope_even_with_high_r2():
    t = np.arange(20)
    increasing_q = 2 * np.exp(0.05 * t)  # perfettamente lineare in log-spazio, ma crescente

    log_results = [
        {"baseflow_segment_ln": pd.DataFrame({"t": t, "ln_Q": np.log(increasing_q)})}
    ]

    valid, discarded = filter_segments_by_linearity(log_results, r2_min=0.95)

    assert valid == []
    assert len(discarded) == 1
    assert discarded[0]["R2"] == pytest.approx(1.0)
    assert discarded[0]["slope"] > 0

def test_keep_longest_segment_per_decreasing_branch():
    seg_a_short = pd.DataFrame({"t": [0, 1, 2]})
    seg_a_long = pd.DataFrame({"t": [0, 1, 2, 3, 4, 5]})
    seg_b = pd.DataFrame({"t": [0, 1, 2, 3]})

    valid_segments = [
        {"decreasing_segment_id": 0, "baseflow_segment_ln": seg_a_short, "tag": "a_short"},
        {"decreasing_segment_id": 0, "baseflow_segment_ln": seg_a_long, "tag": "a_long"},
        {"decreasing_segment_id": 1, "baseflow_segment_ln": seg_b, "tag": "b"},
    ]

    kept = keep_longest_segment_per_decreasing_branch(valid_segments)

    assert sorted(r["tag"] for r in kept) == ["a_long", "b"]

def test_interpolate_valid_segments_follows_fitted_line():
    seg_ln = pd.DataFrame({"t": [0, 1, 2, 3, 4], "ln_Q": [2.3, 2.2, 2.1, 2.0, 1.9]})
    valid_segments = [{"slope": -0.1, "intercept": 2.3, "baseflow_segment_ln": seg_ln}]

    out = interpolate_valid_segments(valid_segments, num_points=10)
    interp = out[0]["segment_interpolated"]

    assert len(interp) == 10
    assert np.allclose(interp["ln_Q"], 2.3 - 0.1 * interp["t"])

def _build_interpolated_results(q0_alpha_pairs):
    baseflow_results = []
    for sid, (q0, alpha) in enumerate(q0_alpha_pairs):
        dates = pd.date_range("2000-01-01", periods=25, freq="D")
        q = q0 * np.exp(-alpha * np.arange(25))
        seg = pd.DataFrame({"Date": dates, "Q": q, "decreasing_segment_id": sid})
        baseflow_results.append(
            {"decreasing_segment_id": sid, "inflection_point": None, "baseflow_segment": seg}
        )

    log_results = transform_segments_to_log_space(baseflow_results)
    valid, _ = filter_segments_by_linearity(log_results, r2_min=0.95)
    valid = keep_longest_segment_per_decreasing_branch(valid)
    return interpolate_valid_segments(valid, num_points=50)


def test_align_overlapping_segments_merges_overlapping_ranges_only():
    interp = _build_interpolated_results([(20.0, 0.03), (12.0, 0.03), (1.0, 0.03)])

    df_aligned, aligned_blocks, df_non_aligned = align_overlapping_segments(
        interp, q_tolerance=0.01
    )

    assert len(aligned_blocks) == 2

    q_ranges = sorted(
        (block["Q"].min(), block["Q"].max()) for block in aligned_blocks
    )
    assert q_ranges[0][1] == pytest.approx(1.0, rel=1e-6)
    assert q_ranges[1][1] == pytest.approx(20.0, rel=1e-6)


def test_align_overlapping_segments_returns_empty_with_fewer_than_two_segments():
    df_aligned, aligned_blocks, df_non_aligned = align_overlapping_segments(
        [], q_tolerance=0.01
    )

    assert df_aligned.empty
    assert aligned_blocks == []
    assert list(df_non_aligned.columns) == ["Date", "Q", "ln_Q", "t", "t_aligned", "cluster_id"]

def test_apply_global_shift_raises_without_valid_blocks():
    with pytest.raises(ValueError):
        apply_global_shift(
            [pd.DataFrame(), None], interpolated_blocks=[], df_non_aligned=pd.DataFrame()
        )

def test_get_lh_k_from_mrc_method_mrc():
    result = get_lh_k_from_mrc({"fit_results": {"alpha": 0.05}}, method="mrc")
    assert result["alpha"] == pytest.approx(0.05)
    assert result["k_day"] == pytest.approx(np.exp(-0.05))
    assert result["k_hour"] == pytest.approx(np.exp(-0.05 / 24))


def test_get_lh_k_from_mrc_is_case_insensitive():
    result = get_lh_k_from_mrc({"fit_results": {"alpha": 0.05}}, method="MRC")
    assert result["method"] == "mrc"


def test_get_lh_k_from_mrc_raises_without_fit_results():
    with pytest.raises(ValueError):
        get_lh_k_from_mrc({"fit_results": None}, method="mrc")


def test_get_lh_k_from_mrc_mean_and_median_ignore_positive_slopes():
    valid_segments = [{"slope": -0.05}, {"slope": -0.07}, {"slope": 0.02}]

    result_mean = get_lh_k_from_mrc({"valid_segments": valid_segments}, method="mean")
    result_median = get_lh_k_from_mrc({"valid_segments": valid_segments}, method="median")

    assert result_mean["alpha"] == pytest.approx(0.06)
    assert result_median["alpha"] == pytest.approx(0.06)


def test_get_lh_k_from_mrc_raises_without_negative_slopes():
    with pytest.raises(ValueError):
        get_lh_k_from_mrc({"valid_segments": [{"slope": 0.02}]}, method="mean")


def test_get_lh_k_from_mrc_raises_for_invalid_method():
    with pytest.raises(ValueError):
        get_lh_k_from_mrc({}, method="bogus")

def test_fit_maillet_model_recovers_known_alpha_and_q0():
    t = np.arange(30)
    alpha_true, q0_true = 0.08, 15.0
    df_mrc = pd.DataFrame({"t_global": t, "ln_Q": np.log(q0_true) - alpha_true * t})

    fit_results, model = fit_maillet_model(df_mrc)

    assert fit_results["alpha"] == pytest.approx(alpha_true)
    assert fit_results["Q0"] == pytest.approx(q0_true)
    assert fit_results["R2"] == pytest.approx(1.0)


def test_fit_maillet_model_returns_none_for_insufficient_data():
    assert fit_maillet_model(pd.DataFrame()) == (None, None)
    assert fit_maillet_model(pd.DataFrame({"t_global": [0], "ln_Q": [1.0]})) == (None, None)

def test_compute_mrc_raises_for_missing_columns():
    with pytest.raises(ValueError):
        compute_mrc(pd.DataFrame({"Q": [1, 2, 3]}))

    with pytest.raises(ValueError):
        compute_mrc(pd.DataFrame({"Date": pd.date_range("2000-01-01", periods=3)}))


def test_compute_mrc_raises_when_no_valid_recession_segments():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=30, freq="D")
    q_noise = np.clip(5 + rng.normal(0, 3, 30), 0.1, None)
    df_noise = pd.DataFrame({"Date": dates, "Q": q_noise})

    with pytest.raises(ValueError):
        compute_mrc(df_noise, min_recession_length=10)