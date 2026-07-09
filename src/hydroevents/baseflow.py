# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:55:32 2026

@author: sofia
"""

from __future__ import annotations
from typing import Any


import pandas as pd
import numpy as np

def lyne_hollick_filter(Qs, a, direction="f"):
    """
    Apply the Lyne & Hollick digital filter.

    Parameters
    ----------
    Qs : array-like
        Streamflow series.
    a : float
        Filter parameter (alpha).
    direction : str, default="f"
        Filter direction:
        - "f"   : forward
        - "r"   : reverse
        - "frf" : multiple passes

    Returns
    -------
    np.ndarray
        Estimated baseflow series.
    """
    Qs = np.array(Qs, dtype=float)
    f = np.zeros(len(Qs))

    if len(direction) == 1:

        if direction == "f":
            for t in range(1, len(Qs)):
                f[t] = a * f[t - 1] + (1 + a) / 2 * (Qs[t] - Qs[t - 1])

                if Qs[t] - f[t] > Qs[t]:
                    f[t] = 0

        elif direction == "r":
            for t in range(len(Qs) - 2, -1, -1):
                f[t] = a * f[t + 1] + (1 + a) / 2 * (Qs[t] - Qs[t + 1])

                if Qs[t] - f[t] > Qs[t]:
                    f[t] = 0

        return Qs - f

    current_direction = direction[0]
    remaining_directions = direction[1:]

    filtered_Qs = lyne_hollick_filter(Qs, a, current_direction)
    return lyne_hollick_filter(filtered_Qs, a, remaining_directions)

def compute_bfi(
    baseflow: pd.Series,
    streamflow: pd.Series,
) -> float:
    """
    Compute the Baseflow Index (BFI).

    Parameters
    ----------
    baseflow : pandas.Series
        Baseflow component.
    streamflow : pandas.Series
        Total streamflow series.

    Returns
    -------
    float
        Baseflow Index.
    """
    total_baseflow = pd.to_numeric(baseflow, errors="coerce").sum()
    total_streamflow = pd.to_numeric(streamflow, errors="coerce").sum()

    if total_streamflow <= 0:
        return np.nan

    return float(total_baseflow / total_streamflow)

def separate_baseflow(
    df: pd.DataFrame,
    q_col: str = "Q",
    date_col: str | None = None,
    k_hour: float = 0.95,
    mrc_results: dict[str, Any] | None = None,
    k_method: str = "mrc",
    direction: str = "f",
    nan_to_zero: bool = True,
) -> dict[str, Any]:
    """
    Run the full baseflow separation workflow using the Lyne–Hollick filter.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing streamflow data.
    q_col : str, default="Q"
        Name of the streamflow column.
    date_col : str or None, default=None
        Optional name of the datetime column. If provided, it is converted to datetime
        and set as index in the returned DataFrame.
    k_hour : float, default=0.95
        Manual hourly Lyne-Hollick filter parameter. Used only if `mrc_results`
        is not provided.
    mrc_results : dict or None, default=None
        Output dictionary returned by `compute_mrc`. If provided, the Lyne-Hollick
        parameter is taken from `mrc_results["k_estimates"]`.
    k_method : {"mrc", "mean", "median"}, default="mrc"
        Method used to select the hourly Lyne-Hollick parameter from `mrc_results`:
        - "mrc": k derived from the fitted Master Recession Curve
        - "mean": k derived from the mean alpha of valid recession segments
        - "median": k derived from the median alpha of valid recession segments
    nan_to_zero : bool, default=False
        If True, missing values in the input streamflow series are replaced with zero
        before filtering. If False, missing values are kept as NaN.
    direction : str, default="f"
        Direction of the Lyne-Hollick filtering pass:
        - "f": forward
        - "r": reverse
        - combinations such as "frf": multiple passes
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'df': DataFrame with original/input/baseflow/stormflow series
        - 'bfi': Baseflow Index
        - 'k_hour': hourly filter parameter used
        - 'k_method': method used to select the filter parameter
        - 'k_info': dictionary with method, alpha, k_day and k_hour when available
    
    Raises
    ------
    ValueError
        If `q_col` is missing from the input DataFrame, or if `mrc_results`
        does not contain the requested k estimate.
    """
    if q_col not in df.columns:
        raise ValueError(f"Column '{q_col}' not found in input DataFrame.")

    df_out = df.copy()

    if date_col is not None:
        if date_col not in df_out.columns:
            raise ValueError(f"Column '{date_col}' not found in input DataFrame.")
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
        df_out = df_out.set_index(date_col)

    df_out[q_col] = pd.to_numeric(df_out[q_col], errors="coerce")
    df_out["Q_original"] = df_out[q_col].copy()

    if nan_to_zero:
        df_out["Q_input"] = df_out[q_col].fillna(0.0)
    else:
        df_out["Q_input"] = df_out[q_col].copy()
        
    if mrc_results is not None:
        if "k_estimates" not in mrc_results:
            raise ValueError("mrc_results must contain 'k_estimates'.")
    
        if k_method not in mrc_results["k_estimates"]:
            raise ValueError(
                f"k_method must be one of {list(mrc_results['k_estimates'].keys())}."
            )
    
        k_info = mrc_results["k_estimates"][k_method]
    
        if k_info is None:
            raise ValueError(f"No k estimate available for method '{k_method}'.")
    
        k_hour = k_info["k_hour"]
    
    else:
        k_info = {
            "method": "manual",
            "k_hour": k_hour,
        }

    baseflow = lyne_hollick_filter(df_out["Q_input"], a=k_hour,direction=direction)
    stormflow = df_out["Q_input"] - baseflow
    stormflow = stormflow.clip(lower=0.0)
    
    baseflow = np.minimum(baseflow, df_out["Q_input"].values)
    baseflow = np.maximum(baseflow, 0.0)

    df_out["Baseflow"] = baseflow
    df_out["Stormflow"] = stormflow

    bfi_total = compute_bfi(df_out["Baseflow"], df_out["Q_input"])

    return {
        "df": df_out,
        "bfi": bfi_total,
        "k_hour": k_hour,
        "k_method": k_info["method"],
        "k_info": k_info,
        "direction":direction
    }
    
    