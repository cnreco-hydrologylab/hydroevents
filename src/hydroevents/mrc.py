# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:44:58 2026

@author: sofia
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def extract_recession_segments(
    df,
    min_length=10,
    recession_tolerance=0.0,
):
    
    recession_segments = []
    start = None
    segment_id = 0

    Q = df["Q"].values
    dQ = np.diff(Q)

    for i in range(1, len(Q)):
        if dQ[i - 1] <= recession_tolerance:
            if start is None:
                start = i - 1
        else:
            if start is not None:
                end = i
                if (end - start) >= min_length:
                    seg = df.iloc[start:end][["Date", "Q"]].reset_index(drop=True)
                    seg["decreasing_segment_id"] = segment_id
                    recession_segments.append(seg)
                    segment_id += 1
                start = None

    if start is not None and (len(Q) - start) >= min_length:
        seg = df.iloc[start:][["Date", "Q"]].reset_index(drop=True)
        seg["decreasing_segment_id"] = segment_id
        recession_segments.append(seg)
        segment_id += 1

    return recession_segments

def identify_inflection_points(recession_segments, min_baseflow_length=5, skip_after_inflection=2):
    baseflow_results = []
    
    for segment_df in recession_segments:
        Q = segment_df["Q"].values

        if len(Q) < 3:
            continue

        dQ = np.diff(Q)
        d2Q = np.diff(dQ)

        if len(d2Q) == 0:
            continue

        inflection_idx = np.argmin(d2Q) + 2

        if inflection_idx >= len(segment_df):
            continue

        inflection_point = segment_df.iloc[[inflection_idx]]
        
        if skip_after_inflection >= min_baseflow_length:
            raise ValueError(
                "skip_after_inflection must be smaller than min_baseflow_length."
            )
        
        if skip_after_inflection > min_baseflow_length / 2:
            raise ValueError(
                "skip_after_inflection should not exceed half of min_baseflow_length."
            )
                
        baseflow_start_idx = inflection_idx + skip_after_inflection
        baseflow_segment = segment_df.iloc[baseflow_start_idx:].copy()

        if len(baseflow_segment) < min_baseflow_length:
            continue

        baseflow_results.append({
            "decreasing_segment_id": segment_df["decreasing_segment_id"].iloc[0],
            "inflection_point": inflection_point,
            "baseflow_segment": baseflow_segment
        })

    return baseflow_results

def transform_segments_to_log_space(baseflow_results):

    log_results = []
    
    for res in baseflow_results:
        segment_plot = res["baseflow_segment"].copy()
        segment_ln = segment_plot.copy()
        segment_ln = segment_ln[segment_ln["Q"] > 0].copy()
        
        if segment_ln.empty or len(segment_ln) < 2:
            continue

        segment_ln["ln_Q"] = np.log(segment_ln["Q"])
        segment_ln["t"] = (segment_ln["Date"] - segment_ln["Date"].iloc[0]).dt.days

        segment_ln.replace([np.inf, -np.inf], np.nan, inplace=True)
        segment_ln.dropna(subset=["ln_Q"], inplace=True)

        if segment_ln.empty or len(segment_ln) < 2:
            continue

        log_results.append({
            "decreasing_segment_id": res["decreasing_segment_id"],
            "inflection_point": res["inflection_point"],
            "baseflow_segment": segment_plot,
            "baseflow_segment_ln": segment_ln[["Date", "Q", "ln_Q", "t"]]
        })
    return log_results

def filter_segments_by_linearity(log_results, r2_min=0.95):
    valid_segments = []
    discarded_segments = []

    for res in log_results:
        segment_ln = res.get("baseflow_segment_ln")

        if segment_ln is None or segment_ln.empty:
            continue

        X = segment_ln["t"].values.reshape(-1, 1)
        Y = segment_ln["ln_Q"].values

        model = LinearRegression()
        model.fit(X, Y)

        r2 = model.score(X, Y)
        slope = model.coef_[0]
        intercept = model.intercept_

        res["R2"] = r2
        res["slope"] = slope
        res["intercept"] = intercept

        if r2 >= r2_min and slope < 0:
            valid_segments.append(res)
        else:
            discarded_segments.append(res)

    return valid_segments, discarded_segments

def keep_longest_segment_per_decreasing_branch(valid_segments):
    best = {}

    for res in valid_segments:
        sid = res.get("decreasing_segment_id")
        seg = res.get("baseflow_segment_ln")

        if sid is None or seg is None or seg.empty:
            continue

        n = len(seg)

        if sid not in best:
            best[sid] = res
        else:
            old_n = len(best[sid]["baseflow_segment_ln"])
            if n > old_n:
                best[sid] = res

    return list(best.values())

def interpolate_valid_segments(valid_segments, num_points=100):
    interpolated_results = []

    for res in valid_segments:
        slope = res["slope"]
        intercept = res["intercept"]
        segment_real = res.get("baseflow_segment_ln")

        if segment_real is None or segment_real.empty:
            continue

        t_start = segment_real["t"].min()
        t_end = segment_real["t"].max()

        t_interp = np.linspace(t_start, t_end, num_points)
        ln_q_interp = intercept + slope * t_interp

        df_interp = pd.DataFrame({
            "t": t_interp,
            "ln_Q": ln_q_interp
        })

        res["segment_interpolated"] = df_interp
        interpolated_results.append(res)

    return interpolated_results

def align_overlapping_segments(interpolated_results, q_tolerance=0.01):
    segment_info = []

    for result in interpolated_results:
        real_segment = result.get("baseflow_segment_ln")
        interp_segment = result.get("segment_interpolated")

        if real_segment is None or interp_segment is None or real_segment.empty:
            continue

        if "t" not in real_segment.columns:
            real_segment = real_segment.copy()
            real_segment["t"] = (
                real_segment["Date"] - real_segment["Date"].iloc[0]).dt.days

        q_max = real_segment["Q"].max()
        q_min = real_segment["Q"].min()
        
        slope = result.get('slope')
        intercept = result.get('intercept')

        segment_info.append({
            "q_max": q_max,
            "q_min": q_min,
            "slope": slope,
            "intercept": intercept,
            "interp": interp_segment.copy(),
            "real": real_segment.copy()
        })
        
    
    if len(segment_info) < 2:
        return pd.DataFrame(), [], pd.DataFrame(columns=["Date", "Q", "ln_Q", "t", "t_aligned", "cluster_id"])

    segment_info.sort(key=lambda x: x["q_max"], reverse=True)

    aligned_blocks = []
    non_aligned_segments = []
    cluster_id = 0
    t_offset = 0.0
    current_block = []

    previous_segment = segment_info[0]
    previous_real = previous_segment["real"].copy()
    previous_real["t_aligned"] = previous_real["t"] + t_offset
    previous_real["cluster_id"] = cluster_id
    current_block.append(previous_real)
    
    for i in range(1, len(segment_info)):
        current = segment_info[i]
        current_real = current["real"].copy()

        if current_real.empty:
            continue

        if current["q_max"] > previous_segment["q_min"] - q_tolerance:
            f_prev = interp1d(
                previous_segment["interp"]["ln_Q"],
                previous_segment["interp"]["t"],
                fill_value="extrapolate"
            )
            ln_q_target = np.log(current["q_max"])
            t_match_prev = float(f_prev(ln_q_target))
            t_offset += t_match_prev - current["real"]["t"].min()

            current_real["t_aligned"] = current_real["t"] + t_offset
            current_real["cluster_id"] = cluster_id
            current_block.append(current_real)
        
        else:
            if current_block:
                aligned_blocks.append(pd.concat(current_block, ignore_index=True))


            cluster_id += 1
            t_offset = 0.0

            current_real = current['real'].copy()
            current_real["t_aligned"] = current_real["t"] + t_offset
            current_real["cluster_id"] = cluster_id

            current_block = [current_real]

        previous_segment = current

    if current_block:
        #for b in current_block:
        aligned_blocks.append(pd.concat(current_block, ignore_index=True))


    df_aligned = (
        pd.concat(aligned_blocks, ignore_index=True)
        if aligned_blocks else pd.DataFrame()
    )
    df_aligned = df_aligned.sort_values(['cluster_id', 't_aligned']).reset_index(drop=True)

    if non_aligned_segments:
        valid_non_aligned = [
            df_ for df_ in non_aligned_segments
            if df_ is not None and "t_aligned" in df_.columns
        ]
        df_non_aligned = (
            pd.concat(valid_non_aligned, ignore_index=True)
            if valid_non_aligned else
            pd.DataFrame(columns=["Date", "Q", "ln_Q", "t", "t_aligned", "cluster_id"])
        )
    else:
        df_non_aligned = pd.DataFrame(
            columns=["Date", "Q", "ln_Q", "t", "t_aligned", "cluster_id"]
        )

    return df_aligned, aligned_blocks, df_non_aligned

def interpolate_aligned_blocks(aligned_blocks, num_points=100):
    interpolated_blocks = []

    for i, block in enumerate(aligned_blocks):
        if (
            block is None or block.empty or
            "t_aligned" not in block.columns or
            "Q" not in block.columns
        ):
            continue

        block_clean = block.copy()
        block_clean["Q"] = block_clean["Q"].replace([np.inf, -np.inf], np.nan)
        block_clean["Q"] = (
            block_clean["Q"]
            .replace({0: np.nan})
            .interpolate()
            .bfill()
            .ffill()
        )

        ln_q = np.log(block_clean["Q"].values)
        t = block_clean["t_aligned"].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(t, ln_q)

        t_interp = np.linspace(t.min(), t.max(), num_points)
        ln_q_interp = model.predict(t_interp.reshape(-1, 1))

        interpolated_blocks.append(pd.DataFrame({
            "t_aligned": t_interp,
            "ln_Q": ln_q_interp
        }))

    return interpolated_blocks

def apply_global_shift(aligned_blocks, interpolated_blocks=None, df_non_aligned=None, q_tolerance=0.01):
    """
    Globally align independent recession blocks using a cumulative MRC fit.

    Workflow:
    1. sort blocks from highest to lowest Qmax
    2. keep the first block as reference
    3. fit ln(Q) = a + b*t_global using all already-shifted real points
    4. shift the next block so its Qmax lies on the current cumulative fit
    5. add the shifted block to the cumulative MRC
    6. repeat until all blocks are aligned

    The final MRC is built only from real shifted points.
    """

    block_objects = []

    for i, block in enumerate(aligned_blocks):
        if block is None or block.empty:
            continue

        if "Q" not in block.columns or "t_aligned" not in block.columns:
            continue

        block_clean = block.copy()
        block_clean = block_clean.replace([np.inf, -np.inf], np.nan)
        block_clean = block_clean.dropna(subset=["Q", "t_aligned"])
        block_clean = block_clean[block_clean["Q"] > 0]

        if block_clean.empty:
            continue

        q_max = block_clean["Q"].max()

        block_objects.append({
            "index": i,
            "q_max": q_max,
            "real": block_clean,
        })
        
    # non-aligned segments
    if df_non_aligned is not None and not df_non_aligned.empty:
    
        for cid, g in df_non_aligned.groupby("cluster_id"):
    
            if g is None or g.empty:
                continue
    
            if "Q" not in g.columns or "t_aligned" not in g.columns:
                continue
    
            g_clean = g.copy()
            g_clean = g_clean.replace([np.inf, -np.inf], np.nan)
            g_clean = g_clean.dropna(subset=["Q", "t_aligned"])
            g_clean = g_clean[g_clean["Q"] > 0]
    
            if g_clean.empty:
                continue
    
            q_max = g_clean["Q"].max()
    
            block_objects.append({
                "index": f"non_aligned_{cid}",
                "q_max": q_max,
                "real": g_clean,
            })

    if not block_objects:
        raise ValueError("No valid aligned blocks found for global alignment.")

    block_objects.sort(key=lambda x: x["q_max"], reverse=True)

    shifted_blocks = [None] * len(aligned_blocks)

    shifted_interp_blocks = (
        [None] * len(interpolated_blocks)
        if interpolated_blocks is not None
        else []
    )

    # First block = reference, no global shift
    first = block_objects[0]
    first_real = first["real"].copy()
    first_real["t_global"] = first_real["t_aligned"]

    shifted_blocks[first["index"]] = first_real

    cumulative_points = [
        first_real[["t_global", "Q"]].copy()
    ]

    alignment_log = []

    alignment_log.append({
        "block_index": first["index"],
        "q_max": float(first["q_max"]),
        "t_shift": 0.0,
        "reference": "first_block",
        "fit_alpha_before_shift": np.nan,
        "fit_r2_before_shift": np.nan,
    })

    # Align each lower-flow block to the cumulative fit
    for obj in block_objects[1:]:

        current_real = obj["real"].copy()

        df_ref = pd.concat(cumulative_points, ignore_index=True)
        df_ref = df_ref.replace([np.inf, -np.inf], np.nan)
        df_ref = df_ref.dropna(subset=["t_global", "Q"])
        df_ref = df_ref[df_ref["Q"] > 0]

        if len(df_ref) < 2:
            raise ValueError("Not enough cumulative points to fit MRC.")

        x = df_ref["t_global"].values.reshape(-1, 1)
        y = np.log(df_ref["Q"].values)

        fit = LinearRegression()
        fit.fit(x, y)

        slope = float(fit.coef_[0])
        intercept = float(fit.intercept_)
        alpha = -slope
        r2 = float(fit.score(x, y))

        q_target = float(current_real["Q"].max())
        ln_q_target = np.log(q_target)

        # From ln(Q) = intercept + slope*t
        # t = (ln(Q) - intercept) / slope
        if np.isclose(slope, 0.0):
            raise ValueError("Cumulative MRC fit slope is too close to zero.")

        t_target = float((ln_q_target - intercept) / slope)

        t_shift_raw = t_target - float(current_real["t_aligned"].min())
        
        # allow only positive/right shifts
        t_shift = max(0.0, t_shift_raw)

        current_real["t_global"] = current_real["t_aligned"] + t_shift

        shifted_blocks[obj["index"]] = current_real

        cumulative_points.append(
            current_real[["t_global", "Q"]].copy()
        )

        alignment_log.append({
            "block_index": obj["index"],
            "q_max": q_target,
            "t_shift": float(t_shift),
            "reference": "cumulative_fit",
            "fit_alpha_before_shift": float(alpha),
            "fit_r2_before_shift": float(r2),
        })

    df_mrc = pd.concat(cumulative_points, ignore_index=True)
    df_mrc = df_mrc.replace([np.inf, -np.inf], np.nan)
    df_mrc = df_mrc.dropna(subset=["t_global", "Q"])
    df_mrc = df_mrc[df_mrc["Q"] > 0].copy()

    df_mrc["ln_Q"] = np.log(df_mrc["Q"])
    df_mrc = df_mrc.sort_values("t_global").reset_index(drop=True)

    df_alignment_log = pd.DataFrame(alignment_log)

    return shifted_blocks, shifted_interp_blocks, df_mrc, df_alignment_log


def get_lh_k_from_mrc(results: dict, method: str = "mrc") -> dict:
    """
    Estimate the Lyne-Hollick recession parameter k from MRC results.
    """
    method = method.lower()

    if method == "mrc":
        fit_results = results.get("fit_results")
        if fit_results is None:
            raise ValueError("No MRC fit results available.")

        alpha = fit_results["alpha"]

    elif method in {"mean", "median"}:
        valid_segments = results.get("valid_segments", [])

        alphas = []
        for seg in valid_segments:
            slope = seg.get("slope")
            if slope is not None and slope < 0:
                alphas.append(-slope)

        if len(alphas) == 0:
            raise ValueError("No valid segment slopes available.")

        if method == "mean":
            alpha = float(np.mean(alphas))
        else:
            alpha = float(np.median(alphas))

    else:
        raise ValueError("method must be one of: 'mrc', 'mean', 'median'.")

    return {
        "method": method,
        "alpha": float(alpha),
        "k_day": float(np.exp(-alpha)),
        "k_hour": float(np.exp(-alpha / 24)),
    }

def build_shifted_segments_dataframe(
    shifted_blocks,
    df_non_shifted=None,
):
    """
    Build a single DataFrame containing all globally shifted
    recession segments for plotting the MRC.

    Parameters
    ----------
    shifted_blocks : list
        List of globally shifted aligned blocks.
    df_non_shifted : pandas.DataFrame or None
        Non-aligned shifted recession segments.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all shifted segments with:
        - Q
        - ln_Q
        - t_global
        - cluster_id
        - segment_id
    """

    dfs = []
    segment_id = 0

    for block in shifted_blocks:
        if block is None or block.empty:
            continue

        df_tmp = block.copy()

        if "ln_Q" not in df_tmp.columns:
            df_tmp["ln_Q"] = np.log(df_tmp["Q"])

        df_tmp["segment_id"] = segment_id

        dfs.append(
            df_tmp[
                [
                    "Date",
                    "Q",
                    "ln_Q",
                    "t",
                    "t_aligned",
                    "t_global",
                    "cluster_id",
                    "segment_id",
                ]
            ]
        )

        segment_id += 1

    if df_non_shifted is not None and not df_non_shifted.empty:

        for _, g in df_non_shifted.groupby("cluster_id"):

            df_tmp = g.copy()

            if "ln_Q" not in df_tmp.columns:
                df_tmp["ln_Q"] = np.log(df_tmp["Q"])

            if "t_global" not in df_tmp.columns:
                df_tmp["t_global"] = df_tmp["t_aligned"]

            df_tmp["segment_id"] = segment_id

            dfs.append(
                df_tmp[
                    [
                        "Q",
                        "ln_Q",
                        "t_global",
                        "cluster_id",
                        "segment_id",
                    ]
                ]
            )

            segment_id += 1

    if len(dfs) == 0:
        return pd.DataFrame(
            columns=[
                "Q",
                "ln_Q",
                "t_global",
                "cluster_id",
                "segment_id",
            ]
        )

    return pd.concat(dfs, ignore_index=True)


def fit_maillet_model(df_mrc):
    if df_mrc.empty or len(df_mrc) < 2:
        return None, None

    x = df_mrc["t_global"].values.reshape(-1, 1)
    y = df_mrc["ln_Q"].values

    valid = np.isfinite(x).flatten() & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return None, None

    model = LinearRegression()
    model.fit(x, y)

    alpha = -model.coef_[0]
    ln_q0 = model.intercept_
    q0 = np.exp(ln_q0)
    r2 = model.score(x, y)

    fit_results = {
        "alpha": alpha,
        "k_day": np.exp(-alpha),
        "k_hour": np.exp(-alpha / 24),
        "ln_Q0": ln_q0,
        "Q0": q0,
        "R2": r2
    }

    return fit_results, model

def compute_mrc(
    df: pd.DataFrame,
    min_recession_length: int = 10,
    recession_tolerance: float = 0.0,
    min_baseflow_length: int = 5,
    skip_after_inflection: int = 2,
    r2_threshold: float = 0.95,
    num_interp_points: int = 100,
    q_tolerance: float = 0.01,
) -> dict[str, object]:
    """
    Run the full Master Recession Curve (MRC) workflow.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily streamflow DataFrame containing columns 'Date' and 'Q'.
    min_recession_length : int, default=10
        Minimum recession segment length.
    min_baseflow_length : int, default=5
        Minimum baseflow segment length after the inflection point.
    r2_threshold : float, default=0.95
        Minimum linearity threshold in ln(Q)-t space.
    num_interp_points : int, default=100
        Number of interpolation points used for valid segments and aligned blocks.
    q_tolerance : float, default=0.01
        Discharge tolerance used during alignment.

    Returns
    -------
    dict
        Dictionary containing the intermediate and final outputs of the MRC workflow.
    """
    
    if "Q" not in df.columns or "Date" not in df.columns:
        raise ValueError("Input DataFrame must contain 'Date' and 'Q'")
    
    recession_segments = extract_recession_segments(
        df,
        min_length=min_recession_length,
        recession_tolerance=recession_tolerance,
    )

    baseflow_results = identify_inflection_points(
        recession_segments,
        min_baseflow_length=min_baseflow_length,
        skip_after_inflection=skip_after_inflection,
    )

    log_results = transform_segments_to_log_space(baseflow_results)

    valid_segments, discarded_segments = filter_segments_by_linearity(
        log_results,
        r2_min=r2_threshold,
    )

    valid_segments = keep_longest_segment_per_decreasing_branch(valid_segments)

    if len(valid_segments) == 0:
        raise ValueError("No valid recession segments after filtering.")

    interpolated_results = interpolate_valid_segments(
        valid_segments,
        num_points=num_interp_points,
    )

    df_aligned, aligned_blocks, df_non_aligned = align_overlapping_segments(
        interpolated_results,
        q_tolerance=q_tolerance,
    )

    interpolated_blocks = interpolate_aligned_blocks(
        aligned_blocks,
        num_points=num_interp_points,
    )

    shifted_blocks, shifted_interp_blocks, df_mrc, df_alignment_log = apply_global_shift(
        aligned_blocks,
        interpolated_blocks,
        df_non_aligned=df_non_aligned,
        q_tolerance=q_tolerance,
    )
    
    df_non_shifted = pd.DataFrame()

    fit_results, model = fit_maillet_model(df_mrc)
    
    df_shifted_segments = build_shifted_segments_dataframe(
        shifted_blocks,
        df_non_shifted,
    )

    k_estimates = {}
    
    for method in ["mrc", "mean", "median"]:
        try:
            k_estimates[method] = get_lh_k_from_mrc(
                {
                    "fit_results": fit_results,
                    "valid_segments": valid_segments,
                },
                method=method,
            )
        except ValueError:
            k_estimates[method] = None
    
    df_k_estimates = pd.DataFrame(
        [v for v in k_estimates.values() if v is not None]
    )

    return {
        "recession_segments": recession_segments,
        "baseflow_results": baseflow_results,
        "log_results": log_results,
        "valid_segments": valid_segments,
        "discarded_segments": discarded_segments,
        "interpolated_results": interpolated_results,
        "df_aligned": df_aligned,
        "aligned_blocks": aligned_blocks,
        "df_non_aligned": df_non_aligned,
        "interpolated_blocks": interpolated_blocks,
        "shifted_blocks": shifted_blocks,
        "shifted_interp_blocks": shifted_interp_blocks,
        "df_non_shifted": df_non_shifted,
        "df_mrc": df_mrc,
        "fit_results": fit_results,
        "model": model,
        "k_estimates": k_estimates,
        "df_k_estimates": df_k_estimates,
        "df_shifted_segments": df_shifted_segments,
        "df_alignment_log": df_alignment_log,
    }

        
    
    