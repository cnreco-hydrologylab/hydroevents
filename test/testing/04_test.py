# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:47:51 2026

@author: s.ortenzi
"""

import numpy as np
import pandas as pd

import pytest

from hydroevents import (
    discharge_to_mm,
    estimate_first_peak_frac_min,
    identify_events,
    identify_rain_events,
    associate_events,
    final_clean_events,
    simulate_linear_reservoir,
    rainfall_runoff_event,
)
from hydroevents.events import (
    fit_event_linear_reservoir,
    add_linear_reservoir_fit,
)

def make_two_peaks_series(
    n=90,
    peak1_c=30,
    peak2_c=52,
    half_width=8,
    height=10.0,
    valley_floor=0.5,
    baseline=0.02,
):
    """Serie oraria con due picchi triangolari nello stesso segmento attivo
    (il livello tra i due picchi resta sempre sopra 'baseline', cosi' non si
    creano due segmenti separati per il solo fatto che il flusso torni a
    zero)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    q = np.full(n, baseline)
    lo, hi = peak1_c - half_width, peak2_c + half_width
    for i in range(lo, hi + 1):
        t1 = max(0.0, height * (1 - abs(i - peak1_c) / half_width))
        t2 = max(0.0, height * (1 - abs(i - peak2_c) / half_width))
        q[i] = max(t1, t2, valley_floor)
    return pd.DataFrame({"Stormflow_mm": q}, index=idx)

def test_discharge_to_mm_known_value():
    result = discharge_to_mm(pd.Series([1.0]), area_km2=1.0, dt_seconds=3600.0)
    assert result.iloc[0] == pytest.approx(3.6)


def test_discharge_to_mm_invalid_area_raises():
    with pytest.raises(ValueError):
        discharge_to_mm(pd.Series([1.0]), area_km2=0.0, dt_seconds=3600.0)


def test_estimate_first_peak_frac_min_uses_fallback_with_single_value():
    frac = estimate_first_peak_frac_min([5.0], fallback=0.30)
    assert frac == pytest.approx(0.30)


def test_estimate_first_peak_frac_min_is_clipped_to_bounds():
    frac = estimate_first_peak_frac_min([8.0, 8.0, 8.0], min_frac=0.10, max_frac=0.80)
    assert frac == pytest.approx(0.80)


def test_simulate_linear_reservoir_zero_rc_gives_zero_output():
    P = np.zeros(30)
    P[0] = 10.0
    q = simulate_linear_reservoir(P, rc=0.0, kd=5.0)
    assert np.allclose(q, 0.0)


def test_simulate_linear_reservoir_mass_balance():
    rc, kd = 0.5, 3.0
    P = np.zeros(60)
    P[0] = 10.0
    q = simulate_linear_reservoir(P, rc, kd)
    assert np.sum(q) == pytest.approx(rc * np.sum(P), abs=1e-6)

def test_identify_events_no_events_on_flat_noise():
    df = pd.DataFrame(
        {"Stormflow_mm": np.full(50, 0.02)},
        index=pd.date_range("2020-01-01", periods=50, freq="h"),
    )
    events = identify_events(df, tol=None, min_prominence=0.001, min_distance=3, width=None)
    assert events.empty

def test_identify_events_split_runoff_when_valley_is_deep_and_far():
    df = make_two_peaks_series(valley_floor=0.5)  # valle profonda, picchi a 22h di distanza
    events = identify_events(
        df, tol=None, min_prominence=0.001, min_distance=3, width=None, min_separation_h=12
    )
    assert len(events) == 2
    assert events["Split_Runoff"].all()

def test_identify_events_no_split_when_valley_is_shallow():
    df = make_two_peaks_series(valley_floor=3.0)  # valle poco profonda
    events = identify_events(
        df, tol=None, min_prominence=0.001, min_distance=3, width=None, min_separation_h=12
    )
    assert len(events) == 1
    assert events.loc[0, "Split_Runoff"] == False  # noqa: E712

def test_identify_events_no_split_when_peaks_are_close_in_time():

    df = make_two_peaks_series(peak1_c=30, peak2_c=36, half_width=4, valley_floor=0.5)
    events = identify_events(
        df, tol=None, min_prominence=0.001, min_distance=3, width=None, min_separation_h=12
    )
    assert len(events) == 1
    assert events.loc[0, "Split_Runoff"] == False  # noqa: E712

def test_identify_events_missing_column_raises():
    df = pd.DataFrame({"Q": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="h"))
    with pytest.raises(ValueError):
        identify_events(df, tol=None, min_prominence=0.001, min_distance=3)

def test_identify_rain_events_far_apart_pulses_stay_separate():
    idx = pd.date_range("2020-01-01", periods=60, freq="h")
    p = np.zeros(60)
    p[10:13] = [2, 4, 2]
    p[30:33] = [1, 6, 1]
    df = pd.DataFrame({"P": p}, index=idx)

    events = identify_rain_events(df, tol_P=0.1, dry_interval_hours=6, min_cumulative_P=5)

    assert len(events) == 2
    assert events["Total_P"].tolist() == pytest.approx([8.0, 8.0])


def test_identify_rain_events_close_pulses_are_merged():
    idx = pd.date_range("2020-01-01", periods=60, freq="h")
    p = np.zeros(60)
    p[10:13] = [2, 4, 2]
    p[16:19] = [1, 6, 1]  
    df = pd.DataFrame({"P": p}, index=idx)

    events = identify_rain_events(df, tol_P=0.1, dry_interval_hours=6, min_cumulative_P=5)

    assert len(events) == 1
    assert events.loc[0, "Total_P"] == pytest.approx(16.0)


def test_identify_rain_events_small_pulse_is_discarded():
    idx = pd.date_range("2020-01-01", periods=60, freq="h")
    p = np.zeros(60)
    p[10:12] = [1, 1]
    df = pd.DataFrame({"P": p}, index=idx)

    events = identify_rain_events(df, tol_P=0.1, dry_interval_hours=6, min_cumulative_P=5)

    assert events.empty

def _make_precip_runoff_pair():
    idx = pd.date_range("2020-01-01", periods=50, freq="h")

    p = np.zeros(50)
    p[5:8] = [2, 4, 2]
    p[12] = 1.0         
    df_precip = pd.DataFrame({"P": p}, index=idx)

    q = np.zeros(50)
    q[5:21] = 0.3
    q[10] = 2.0  
    df_runoff = pd.DataFrame({"Stormflow_mm": q}, index=idx)

    runoff_events = pd.DataFrame(
        {
            "Start": [idx[5]],
            "End": [idx[20]],
            "Peak": [idx[10]],
            "First_Peak": [idx[10]],
            "Max_Peak": [idx[10]],
            "Stormflow_mm": [df_runoff["Stormflow_mm"].sum()],
            "Split_Runoff": [False],
            "Noise_tol": [np.nan],
            "Low_flow": [np.nan],
        }
    )

    rain_events = pd.DataFrame(
        {
            "Start": [idx[5]],
            "End": [idx[7]],
            "Peak": [idx[6]],
            "Total_P": [8.0],
        }
    )

    return idx, df_precip, df_runoff, runoff_events, rain_events

def test_associate_events_basic_match_succeeds():
    idx, df_precip, df_runoff, runoff_events, rain_events = _make_precip_runoff_pair()

    associated, discarded = associate_events(runoff_events, rain_events, df_precip, df_runoff)

    assert len(associated) == 1
    assert associated.loc[0, "PeakLag_h"] == pytest.approx(4.0)
    assert associated.loc[0, "Residual_Rain_mm"] == pytest.approx(1.0)
    assert discarded.empty

def test_associate_events_discarded_when_lag_exceeds_max_lag_h():
    idx, df_precip, df_runoff, runoff_events, rain_events = _make_precip_runoff_pair()

    runoff_events = runoff_events.copy()
    far_peak = idx[6] + pd.Timedelta(hours=200)
    runoff_events["First_Peak"] = [far_peak]
    runoff_events["Peak"] = [far_peak]
    runoff_events["Max_Peak"] = [far_peak]

    associated, discarded = associate_events(
        runoff_events, rain_events, df_precip, df_runoff, max_lag_h=120
    )

    assert associated.empty
    assert discarded.loc[0, "Discard_Reason"] == "no_rain_within_lag"


def test_associate_events_discarded_for_low_runoff_efficiency():
    idx, df_precip, df_runoff, runoff_events, rain_events = _make_precip_runoff_pair()

    df_runoff_high = df_runoff.copy()
    df_runoff_high["Stormflow_mm"] = 0.0
    df_runoff_high.loc[idx[5:21], "Stormflow_mm"] = 1.0  # runoff ~= pioggia -> efficienza troppo alta

    runoff_events = runoff_events.copy()
    runoff_events["Stormflow_mm"] = [df_runoff_high["Stormflow_mm"].sum()]

    associated, discarded = associate_events(
        runoff_events, rain_events, df_precip, df_runoff_high, runoff_ratio_limit=0.90
    )

    assert associated.empty
    assert "efficiency" in discarded["Discard_Reason"].tolist()


def test_associate_events_empty_inputs_return_empty_frames():
    idx, df_precip, df_runoff, _, rain_events = _make_precip_runoff_pair()

    associated, discarded = associate_events(
        pd.DataFrame(), rain_events, df_precip, df_runoff
    )

    assert associated.empty
    assert discarded.empty

def test_final_clean_events_filters_invalid_rc_and_outliers():
    associated = pd.DataFrame(
        {
            "Runoff_Volume_mm": [5, 12, 5, 5, 5, 5],
            "Rain_Volume_mm":   [10, 10, 10, 10, 10, 10],
            "Total_Event_h":    [24, 30, 500, 28, 26, 27],
            "PeakLag_h":        [10, 8, 12, 300, 9, 11],
        }
    )

    clean, outliers = final_clean_events(associated)

    assert sorted(clean.index.tolist()) == [0, 4, 5]
    assert sorted(outliers["Discard_Reason"].tolist()) == [
        "final_filter",
        "final_filter",
        "invalid_rc",
    ]


def test_final_clean_events_empty_input_returns_empty_frames():
    clean, outliers = final_clean_events(pd.DataFrame())
    assert clean.empty
    assert outliers.empty

def test_fit_event_linear_reservoir_recovers_known_parameters():
    n = 60
    P = np.zeros(n)
    P[2:6] = [4.0, 8.0, 5.0, 2.0]

    rc_true, kd_true = 0.45, 8.0
    Qobs = simulate_linear_reservoir(P, rc_true, kd_true)

    rc_fit, kd_fit, rmse = fit_event_linear_reservoir(P, Qobs)

    assert rc_fit == pytest.approx(rc_true, abs=0.02)
    assert kd_fit == pytest.approx(kd_true, abs=0.5)
    assert rmse == pytest.approx(0.0, abs=1e-6)


def test_fit_event_linear_reservoir_returns_nan_for_degenerate_input():
    P = np.array([0.0, 0.0, 0.0])
    Qobs = np.array([0.0, 0.0, 0.0])
    rc_fit, kd_fit, rmse = fit_event_linear_reservoir(P, Qobs)
    assert np.isnan(rc_fit) and np.isnan(kd_fit) and np.isnan(rmse)


def test_add_linear_reservoir_fit_recovers_parameters_for_single_event():
    idx = pd.date_range("2020-01-01", periods=60, freq="h")
    P = np.zeros(60)
    P[5:9] = [3, 6, 4, 2]

    rc_true, kd_true = 0.4, 6.0
    q = simulate_linear_reservoir(P, rc_true, kd_true)

    df_rain = pd.DataFrame({"P": P}, index=idx)
    df_runoff = pd.DataFrame({"Stormflow_mm": q}, index=idx)

    events_df = pd.DataFrame(
        {
            "Rain_Start": [idx[5]],
            "Runoff_Start": [idx[5]],
            "Runoff_End": [idx[59]],
        }
    )

    out = add_linear_reservoir_fit(events_df, df_rain, df_runoff)

    assert out.loc[0, "RC_lr"] == pytest.approx(rc_true, abs=0.02)
    assert out.loc[0, "kd_lr_h"] == pytest.approx(kd_true, abs=0.5)


def test_rainfall_runoff_event_end_to_end_smoke():
    """End-to-end smoke test for the top-level orchestrator: a single
    synthetic rainfall pulse driving a known linear-reservoir response
    should be detected, associated, and cleaned into exactly one event
    with recovered volumes/RC consistent with the known inputs."""
    n = 250
    idx = pd.date_range("2020-01-01", periods=n, freq="h")

    P = np.zeros(n)
    P[50:54] = [5.0, 8.0, 5.0, 2.0]  # 20 mm total

    rc_true, kd_true = 0.4, 6.0
    q = simulate_linear_reservoir(P, rc_true, kd_true)

    df_rain = pd.DataFrame({"P": P}, index=idx)
    df_runoff = pd.DataFrame({"Stormflow_mm": q}, index=idx)

    result = rainfall_runoff_event(
        df_runoff=df_runoff,
        df_rain=df_rain,
        basin_name="smoke_basin",
        min_prominence=0.001,
        max_duration_h=200,
        max_lag_h=48,
    )

    for key in (
        "basin_name",
        "df_runoff",
        "df_rain",
        "runoff_events",
        "rain_events",
        "associated_events",
        "discarded_events",
        "clean_events",
        "final_outliers",
    ):
        assert key in result

    clean_events = result["clean_events"]

    assert len(clean_events) == 1
    assert clean_events.index[0] == "smoke_basin_E001"
    assert result["final_outliers"].empty

    event = clean_events.iloc[0]
    assert event["Rain_Volume_mm"] == pytest.approx(20.0, rel=0.05)
    assert event["Runoff_Volume_mm"] == pytest.approx(rc_true * 20.0, rel=0.1)
    assert event["RC_lr"] == pytest.approx(rc_true, abs=0.05)
