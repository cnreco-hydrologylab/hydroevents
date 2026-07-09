# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:39:19 2026

@author: sofia
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fill_streamflow_gaps(
    df: pd.DataFrame,
    q_col: str = "Q",
    zero_threshold: int = 24,
    max_gap: int = 48,
    max_gap_interp: int = 120,
    window_size: int = 5,
    interp_method: str = 'pchip'
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean and gap-fill a streamflow time series.

    The procedure:
    1. converts the target column to numeric
    2. converts negative values to NaN
    3. converts short zero-flow sequences to NaN
    4. classifies missing values by contiguous gap length
    5. fills short gaps with a local centered mean
    6. fills medium gaps with  interpolation
    7. leaves long gaps as NaN

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the streamflow column.
    q_col : str, default="Q"
        Name of the streamflow column.
    zero_threshold : int, default=24
        Maximum length of zero-flow runs to be treated as missing values.
        For hourly data, this is expressed in hours / time steps.
    max_gap : int, default=48
        Maximum gap length filled using centered moving average.
    max_gap_interp : int, default=120
        Maximum gap length filled using interpolation.
        Gaps longer than this remain missing.
    window_size : int, default=5
        Number of time steps before and after each short gap used to compute the local mean.
    interp_method : {"linear", "pchip", "spline"}, default="pchip"
        Interpolation method used to fill intermediate gaps.
        
    Returns
    -------
    df_out : pandas.DataFrame
        Processed DataFrame with:
        - the cleaned/filled streamflow column
        - an additional copy of the original numeric series in `{q_col}_original`
    summary : dict
        Summary metrics of the applied cleaning and filling steps.

    Raises
    ------
    ValueError
        If `q_col` is not present in the input DataFrame.
    """
    if q_col not in df.columns:
        raise ValueError(f"Column '{q_col}' is not in the DataFrame.")

    df_out = df.copy()
    total_values = len(df_out)

    original_col = f"{q_col}_original"
    df_out[q_col] = pd.to_numeric(df_out[q_col], errors="coerce")
    df_out[original_col] = df_out[q_col].copy()

    # Negative values -> NaN
    n_neg = int((df_out[q_col] < 0).sum())
    df_out.loc[df_out[q_col] < 0, q_col] = np.nan

    # Short zero-flow runs -> NaN
    zero_mask = df_out[q_col].eq(0)
    grp_zero = (zero_mask != zero_mask.shift()).cumsum()
    zero_run_len = zero_mask.groupby(grp_zero).transform("sum")

    nan_from_zero_mask = zero_mask & (zero_run_len <= zero_threshold)
    num_nan_from_zero = int(nan_from_zero_mask.sum())
    df_out.loc[nan_from_zero_mask, q_col] = np.nan

    # Missing-data classification by contiguous gap length
    nan_mask = df_out[q_col].isna()
    grp_nan = (nan_mask != nan_mask.shift()).cumsum()
    nan_run_len = nan_mask.groupby(grp_nan).transform("sum")

    gap_small = nan_mask & (nan_run_len <= max_gap)
    gap_medium = nan_mask & (nan_run_len > max_gap) & (nan_run_len <= max_gap_interp)
    gap_large = nan_mask & (nan_run_len > max_gap_interp)

    # Fill short gaps with local centered mean
    filled_by_moving = pd.Series(False, index=df_out.index)
    
    for _, gap_idx in df_out.loc[gap_small].groupby(grp_nan).groups.items():
        gap_idx = list(gap_idx)
    
        start_pos = df_out.index.get_loc(gap_idx[0])
        end_pos = df_out.index.get_loc(gap_idx[-1])
    
        left_start = max(0, start_pos - window_size)
        right_end = min(len(df_out), end_pos + window_size + 1)
    
        local_values = df_out.iloc[left_start:right_end][q_col].dropna()
    
        if not local_values.empty:
            df_out.loc[gap_idx, q_col] = local_values.mean()
            filled_by_moving.loc[gap_idx] = True
    
    num_moving_filled = int(filled_by_moving.sum())
    
    # Fill medium gaps with interpolation
    still_nan_before_interp = df_out[q_col].isna()
    
    q_interp = df_out[q_col].interpolate(
        method=interp_method,
        limit=max_gap_interp,
        limit_direction="both",
        limit_area="inside",
    )
    
    fillable_medium = gap_medium & still_nan_before_interp & q_interp.notna()
    
    df_out.loc[fillable_medium, q_col] = q_interp.loc[fillable_medium]
    
    num_linear_filled = int(fillable_medium.sum())
    
    # Long gaps remain missing
    df_out.loc[gap_large, q_col] = np.nan

    summary = {
        "total_values": total_values,
        "num_negative_to_nan": n_neg,
        "num_nan_from_zero": num_nan_from_zero,
        "num_moving_filled": num_moving_filled,
        "num_linear_filled": num_linear_filled,
        "num_large_gap_values": int(gap_large.sum()),
        "final_nan": int(df_out[q_col].isna().sum()),
    }

    return df_out, summary


def to_daily(
    df: pd.DataFrame,
    date_col: str = "Date",
    q_col: str = "Q",
) -> pd.DataFrame:
    """
    Aggregate a sub-daily streamflow time series to daily mean values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a datetime column and a streamflow column.
    date_col : str, default="Date"
        Name of the datetime column.
    q_col : str, default="Q"
        Name of the streamflow column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per day, containing:
        - the datetime column named as `date_col`
        - the daily mean streamflow column named as `q_col`

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing column '{date_col}'")
    if q_col not in df.columns:
        raise ValueError(f"Missing column '{q_col}'")

    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
    df_out[q_col] = pd.to_numeric(df_out[q_col], errors="coerce")

    df_out = df_out.dropna(subset=[date_col])

    df_daily = (
        df_out.resample("D", on=date_col)[q_col]
        .mean()
        .reset_index()
    )

    return df_daily


def preprocess_streamflow(
    df: pd.DataFrame,
    date_col: str = "Date",
    q_col: str = "Q",
    apply_gap_filling: bool = True,
    zero_threshold: int = 24,
    max_gap: int = 48,
    max_gap_interp: int = 120,
    window_size: int = 5,
    interp_method: str = "pchip",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Preprocess a streamflow time series and derive the daily aggregated series.

    This wrapper:
    - optionally applies gap filling
    - always computes a daily mean streamflow series

    Parameters
    ----------
    df : pandas.DataFrame
        Input streamflow DataFrame.
    date_col : str, default="Date"
        Name of the datetime column.
    q_col : str, default="Q"
        Name of the streamflow column.
    apply_gap_filling : bool, default=True
        Whether to apply gap filling before daily aggregation.
    zero_threshold, max_gap, max_gap_interp, window_size :
        Parameters passed to `fill_streamflow_gaps`.

    Returns
    -------
    df_processed : pandas.DataFrame
        Sub-daily processed streamflow series.
    df_daily : pandas.DataFrame
        Daily aggregated streamflow series.
    summary : dict
        Summary of preprocessing operations.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing column '{date_col}'")
    if q_col not in df.columns:
        raise ValueError(f"Missing column '{q_col}'")

    df_work = df.copy()
    df_work[date_col] = pd.to_datetime(df_work[date_col], errors="coerce")

    if apply_gap_filling:
        df_processed, summary = fill_streamflow_gaps(
            df_work,
            q_col=q_col,
            zero_threshold=zero_threshold,
            max_gap=max_gap,
            max_gap_interp=max_gap_interp,
            window_size=window_size,
            interp_method=interp_method,
        )
        summary["gap_filling"] = "applied"
    else:
        df_processed = df_work.copy()
        df_processed[q_col] = pd.to_numeric(df_processed[q_col], errors="coerce")
        df_processed[f"{q_col}_original"] = df_processed[q_col].copy()
        summary = {
            "gap_filling": "skipped",
            "total_values": len(df_processed),
            "final_nan": int(df_processed[q_col].isna().sum()),
        }

    df_daily = to_daily(
        df_processed,
        date_col=date_col,
        q_col=q_col,
    )

    return df_processed, df_daily, summary