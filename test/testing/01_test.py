# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:13:07 2026

@author: s.ortenzi
"""

import numpy as np
import pandas as pd

import pytest

from hydroevents import preprocess_streamflow
from hydroevents.preprocessing import fill_streamflow_gaps, to_daily


def make_hourly_q():
    rng = np.random.default_rng(42)

    dates = pd.date_range(
        "2000-01-01 00:00",
        periods=10 * 24,
        freq="h"
    )

    t = np.arange(len(dates))

    q = (
        5.0
        + 0.6 * np.sin(2 * np.pi * t / 24)
        + 0.2 * np.sin(2 * np.pi * t / (24 * 5))
        + rng.normal(0, 0.08, len(t))
    )

    q = np.clip(q, 0.1, None)

    return pd.DataFrame({
        "Date": dates,
        "Q": q
    })


def test_negative_values_are_counted_and_cleaned():
    df = make_hourly_q()
    df.loc[10:15, "Q"] = -1.0

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=True,
    )

    assert summary["num_negative_to_nan"] == 6
    assert df_processed.loc[10:15, "Q"].notna().all()
    assert not (df_processed.loc[10:15, "Q"] < 0).any()


def test_negative_values_remain_nan_when_run_exceeds_max_gap_interp():
    df = make_hourly_q()
    df.loc[10:140, "Q"] = -1.0

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=True,
    )

    assert summary["num_negative_to_nan"] == 131
    assert df_processed.loc[10:140, "Q"].isna().all()


def test_short_zero_flow_is_counted_and_cleaned():
    df = make_hourly_q()
    df.loc[20:30, "Q"] = 0.0
    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=True,
    )

    assert summary["num_nan_from_zero"] == 11
    assert df_processed.loc[20:30, "Q"].notna().all()
    assert not (df_processed.loc[20:30, "Q"] == 0).any()


def test_long_zero_flow_run_is_not_treated_as_missing():
    df = make_hourly_q()
    df.loc[20:55, "Q"] = 0.0

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=True,
    )

    assert summary["num_nan_from_zero"] == 0
    assert (df_processed.loc[20:55, "Q"] == 0).all()


def test_q_original_is_preserved():
    df = make_hourly_q()
    df.loc[10:15, "Q"] = -1.0

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=False,
    )

    assert "Q_original" in df_processed.columns
    pd.testing.assert_series_equal(
        df_processed["Q_original"].reset_index(drop=True),
        df["Q"].reset_index(drop=True),
        check_names=False,
    )


def test_short_gap_is_filled_when_gap_filling_is_enabled():
    df = make_hourly_q()
    df.loc[40:45, "Q"] = np.nan

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=True,
    )

    assert df_processed.loc[40:45, "Q"].notna().all()


def test_long_gap_remains_nan():
    df = make_hourly_q()
    df.loc[50:180, "Q"] = np.nan

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=True,
    )

    assert df_processed.loc[50:180, "Q"].isna().all()


def test_daily_aggregation_is_mean_of_hourly_values():
    df = make_hourly_q()

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=False,
    )

    expected_first_day_mean = df.loc[0:23, "Q"].mean()

    assert np.isclose(
        df_daily.iloc[0]["Q"],
        expected_first_day_mean,
        equal_nan=False,
    )


def test_summary_is_returned_as_dictionary():
    df = make_hourly_q()

    df_processed, df_daily, summary = preprocess_streamflow(
        df,
        date_col="Date",
        q_col="Q",
        apply_gap_filling=True,
    )

    assert isinstance(summary, dict)
    assert len(summary) > 0


def test_medium_gap_is_filled_by_interpolation_not_left_missing():
    df = make_hourly_q()
    df.loc[30:100, "Q"] = np.nan

    df_processed, summary = fill_streamflow_gaps(df, q_col="Q")

    assert df_processed.loc[30:100, "Q"].notna().all()
    assert summary["num_linear_filled"] == 71
    assert summary["num_moving_filled"] == 0


def test_fill_streamflow_gaps_missing_column_raises():
    df = make_hourly_q()
    with pytest.raises(ValueError):
        fill_streamflow_gaps(df, q_col="DOES_NOT_EXIST")

def test_to_daily_missing_columns_raise():
    df = make_hourly_q()
    with pytest.raises(ValueError):
        to_daily(df, date_col="DOES_NOT_EXIST")
    with pytest.raises(ValueError):
        to_daily(df, q_col="DOES_NOT_EXIST")


def test_to_daily_aggregates_multiple_days_correctly():
    df = make_hourly_q()
    daily = to_daily(df)

    assert len(daily) == 10
    for day in range(10):
        expected = df.loc[day * 24:(day + 1) * 24 - 1, "Q"].mean()
        assert np.isclose(daily.iloc[day]["Q"], expected)


def test_to_daily_drops_rows_with_unparseable_dates():
    df = make_hourly_q()
    df["Date"] = df["Date"].astype(object)
    df.loc[5, "Date"] = "not-a-date"

    daily = to_daily(df)

    assert len(daily) == 10


def test_to_daily_day_entirely_nan_gives_nan_mean():
    df = make_hourly_q()
    df.loc[24:47, "Q"] = np.nan 

    daily = to_daily(df)

    assert np.isnan(daily.iloc[1]["Q"])
    assert not np.isnan(daily.iloc[0]["Q"])
    assert not np.isnan(daily.iloc[2]["Q"])

def test_preprocess_streamflow_missing_columns_raise():
    df = make_hourly_q()
    with pytest.raises(ValueError):
        preprocess_streamflow(df, date_col="DOES_NOT_EXIST")
    with pytest.raises(ValueError):
        preprocess_streamflow(df, q_col="DOES_NOT_EXIST")