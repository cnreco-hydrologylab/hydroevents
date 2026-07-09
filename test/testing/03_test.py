# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import pytest

from hydroevents import separate_baseflow
from hydroevents.baseflow import lyne_hollick_filter, compute_bfi


def make_hourly_streamflow():
    rng = np.random.default_rng(42)

    dates = pd.date_range("2000-01-01", periods=10 * 24, freq="h")
    t = np.arange(len(dates))

    q = (
        5.0
        + 0.8 * np.sin(2 * np.pi * t / 24)
        + 0.3 * np.sin(2 * np.pi * t / (24 * 4))
        + rng.normal(0, 0.05, len(t))
    )

    q = np.clip(q, 0.1, None)

    return pd.DataFrame({
        "Date": dates,
        "Q": q
    })

def make_fake_mrc_results():
    return {
        "k_estimates": {
            "mrc": {
                "method": "mrc",
                "alpha": 0.05,
                "k_day": np.exp(-0.05),
                "k_hour": np.exp(-0.05 / 24),
            },
            "mean": {
                "method": "mean",
                "alpha": 0.04,
                "k_day": np.exp(-0.04),
                "k_hour": np.exp(-0.04 / 24),
            },
            "median": {
                "method": "median",
                "alpha": 0.06,
                "k_day": np.exp(-0.06),
                "k_hour": np.exp(-0.06 / 24),
            },
        }
    }


def reference_lh_pass(Qs, a, forward=True):

    Qs = np.asarray(Qs, dtype=float)
    n = len(Qs)
    f = np.zeros(n)
    order = range(1, n) if forward else range(n - 2, -1, -1)
    for t in order:
        prev = t - 1 if forward else t + 1
        f[t] = a * f[prev] + (1 + a) / 2 * (Qs[t] - Qs[prev])
        if Qs[t] - f[t] > Qs[t]:
            f[t] = 0
    return Qs - f


def test_baseflow_is_physically_consistent():
    df = make_hourly_streamflow()

    results = separate_baseflow(
        df,
        q_col="Q",
        date_col="Date",
        mrc_results=make_fake_mrc_results(),
        k_method="mrc",
        direction="f",
        nan_to_zero=True,
    )

    out = results["df"]

    assert (out["Baseflow"] <= out["Q_input"] + 1e-12).all()
    assert (out["Baseflow"] >= 0).all()
    assert (out["Stormflow"] >= 0).all()
    assert 0 <= results["bfi"] <= 1

def test_k_hour_is_taken_from_mrc_results():
    df = make_hourly_streamflow()
    mrc_results = make_fake_mrc_results()

    results = separate_baseflow(
        df,
        q_col="Q",
        date_col="Date",
        mrc_results=mrc_results,
        k_method="mrc",
        direction="f",
        nan_to_zero=True,
    )

    expected_k = mrc_results["k_estimates"]["mrc"]["k_hour"]

    assert np.isclose(results["k_hour"], expected_k)
    assert results["k_method"] == "mrc"

def test_filter_directions_produce_valid_results():
    df = make_hourly_streamflow()
    mrc_results = make_fake_mrc_results()

    outputs = {}

    for direction in ["f", "r", "frf"]:
        results = separate_baseflow(
            df,
            q_col="Q",
            date_col="Date",
            mrc_results=mrc_results,
            k_method="mrc",
            direction=direction,
            nan_to_zero=True,
        )

        out = results["df"]
        outputs[direction] = out["Baseflow"].values

        assert (out["Baseflow"] <= out["Q_input"] + 1e-12).all()
        assert (out["Baseflow"] >= 0).all()
        assert 0 <= results["bfi"] <= 1

    assert not np.allclose(outputs["f"], outputs["r"])
    assert not np.allclose(outputs["f"], outputs["frf"])

def test_nan_to_zero_changes_input_series():
    df = make_hourly_streamflow()
    df.loc[20:25, "Q"] = np.nan

    res_zero = separate_baseflow(
        df,
        q_col="Q",
        date_col="Date",
        mrc_results=make_fake_mrc_results(),
        k_method="mrc",
        nan_to_zero=True,
    )

    res_nan = separate_baseflow(
        df,
        q_col="Q",
        date_col="Date",
        mrc_results=make_fake_mrc_results(),
        k_method="mrc",
        nan_to_zero=False,
    )

    assert res_zero["df"].loc["2000-01-01 20:00":"2000-01-02 01:00", "Q_input"].eq(0).all()
    assert res_nan["df"].loc["2000-01-01 20:00":"2000-01-02 01:00", "Q_input"].isna().all()

def test_k_methods_mrc_mean_median_are_used_correctly():
    df = make_hourly_streamflow()
    mrc_results = make_fake_mrc_results()

    for method in ["mrc", "mean", "median"]:
        results = separate_baseflow(
            df,
            q_col="Q",
            date_col="Date",
            mrc_results=mrc_results,
            k_method=method,
            nan_to_zero=True,
        )

        expected_k = mrc_results["k_estimates"][method]["k_hour"]

        assert results["k_method"] == method
        assert np.isclose(results["k_hour"], expected_k)
        assert 0 <= results["bfi"] <= 1

def test_lyne_hollick_forward_matches_reference_recurrence():
    rng = np.random.default_rng(1)
    Qs = np.clip(5 + rng.normal(0, 2, 30), 0.1, None)
    a = 0.925

    result = lyne_hollick_filter(Qs, a, direction="f")
    expected = reference_lh_pass(Qs, a, forward=True)

    assert np.allclose(result, expected)


def test_lyne_hollick_reverse_matches_reference_recurrence():
    rng = np.random.default_rng(1)
    Qs = np.clip(5 + rng.normal(0, 2, 30), 0.1, None)
    a = 0.925

    result = lyne_hollick_filter(Qs, a, direction="r")
    expected = reference_lh_pass(Qs, a, forward=False)

    assert np.allclose(result, expected)


def test_lyne_hollick_multi_pass_equals_sequential_single_passes():
    Qs = np.array([10.0, 8.0, 12.0, 6.0, 9.0, 7.0])
    a = 0.9

    step1 = lyne_hollick_filter(Qs, a, "f")
    step2 = lyne_hollick_filter(step1, a, "r")
    step3 = lyne_hollick_filter(step2, a, "f")

    combined = lyne_hollick_filter(Qs, a, "frf")

    assert np.allclose(step3, combined)


def test_lyne_hollick_baseflow_never_exceeds_streamflow():
    rng = np.random.default_rng(7)
    Qs = np.clip(5 + rng.normal(0, 3, 100), 0.05, None)
    Qs[40:45] = [20, 2, 1.5, 1.2, 1.0]

    baseflow = lyne_hollick_filter(Qs, a=0.98, direction="f")

    assert (baseflow <= Qs + 1e-9).all()
    assert np.any(np.isclose(baseflow, Qs))


def test_compute_bfi_known_ratio():
    bfi = compute_bfi(baseflow=[2, 2, 2], streamflow=[4, 4, 4])
    assert bfi == pytest.approx(0.5)

def test_compute_bfi_nan_for_non_positive_total_streamflow():
    assert np.isnan(compute_bfi(baseflow=[1, 1], streamflow=[0, 0]))
    assert np.isnan(compute_bfi(baseflow=[1, 1], streamflow=[-1, -1]))


def test_compute_bfi_nan_propagation_depends_on_input_type():
    bfi_from_list = compute_bfi(baseflow=[1, 1, "x"], streamflow=[4, 4, 4])
    bfi_from_series = compute_bfi(
        baseflow=pd.Series([1, 1, "x"]), streamflow=pd.Series([4, 4, 4])
    )

    assert np.isnan(bfi_from_list)
    assert bfi_from_series == pytest.approx(2 / 12)

def test_separate_baseflow_manual_k_hour_without_mrc_results():
    df = make_hourly_streamflow()

    results = separate_baseflow(
        df, q_col="Q", date_col="Date", k_hour=0.9, mrc_results=None
    )

    assert results["k_method"] == "manual"
    assert results["k_hour"] == pytest.approx(0.9)
    assert results["k_info"] == {"method": "manual", "k_hour": 0.9}

def test_separate_baseflow_missing_q_col_raises():
    df = make_hourly_streamflow()
    with pytest.raises(ValueError):
        separate_baseflow(df, q_col="DOES_NOT_EXIST", date_col="Date")

def test_separate_baseflow_missing_date_col_raises():
    df = make_hourly_streamflow()
    with pytest.raises(ValueError):
        separate_baseflow(df, q_col="Q", date_col="DOES_NOT_EXIST")

def test_separate_baseflow_missing_k_estimates_key_raises():
    df = make_hourly_streamflow()
    with pytest.raises(ValueError):
        separate_baseflow(df, q_col="Q", date_col="Date", mrc_results={})

def test_separate_baseflow_invalid_k_method_raises():
    df = make_hourly_streamflow()
    mrc_results = {"k_estimates": {"mrc": {"method": "mrc", "k_hour": 0.9}}}
    with pytest.raises(ValueError):
        separate_baseflow(
            df, q_col="Q", date_col="Date", mrc_results=mrc_results, k_method="mean"
        )

def test_separate_baseflow_none_k_estimate_raises():
    df = make_hourly_streamflow()
    mrc_results = {"k_estimates": {"mrc": None}}
    with pytest.raises(ValueError):
        separate_baseflow(
            df, q_col="Q", date_col="Date", mrc_results=mrc_results, k_method="mrc"
        )

def test_separate_baseflow_q_original_preserves_nan_even_with_nan_to_zero():
    df = make_hourly_streamflow()
    df.loc[5:8, "Q"] = np.nan

    results = separate_baseflow(
        df, q_col="Q", date_col="Date", k_hour=0.9, nan_to_zero=True
    )
    out = results["df"]

    assert out["Q_original"].isna().sum() == 4
    assert out["Q_input"].isna().sum() == 0

def test_separate_baseflow_without_date_col_keeps_default_index():
    df = make_hourly_streamflow()
    results = separate_baseflow(df, q_col="Q", date_col=None, k_hour=0.9)
    assert isinstance(results["df"].index, pd.RangeIndex)

def test_separate_baseflow_direction_is_echoed_in_output():
    df = make_hourly_streamflow()
    results = separate_baseflow(
        df, q_col="Q", date_col="Date", k_hour=0.9, direction="fr"
    )
    assert results["direction"] == "fr"
