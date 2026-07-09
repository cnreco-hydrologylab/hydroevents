# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:57:51 2026

@author: sofia
"""

from __future__ import annotations
from typing import Any


import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution

def discharge_to_mm(
    Q: pd.Series | np.ndarray,
    area_km2: float,
    dt_seconds: float,
) -> pd.Series:
    """
    Convert discharge (m³/s) to runoff depth (mm per timestep).

    Parameters
    ----------
    Q : array-like
        Discharge time series in m³/s.
    area_km2 : float
        Catchment area in km².
    dt_seconds : float
        Time step in seconds (e.g. 3600 for hourly data).

    Returns
    -------
    pd.Series
        Runoff depth in mm per timestep.
    """
    if area_km2 <= 0:
        raise ValueError("area_km2 must be > 0")

    Q = pd.Series(Q)

    area_m2 = area_km2 * 1e6

    runoff_mm = Q * dt_seconds / area_m2 * 1000

    return runoff_mm

def estimate_first_peak_frac_min(
    block_peak_vals,
    min_frac: float = 0.10,
    max_frac: float = 0.80,
    fallback: float = 0.30,
) -> float:
    vals = np.asarray(block_peak_vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if len(vals) <= 1:
        return float(fallback)

    peak_max = np.max(vals)
    if not np.isfinite(peak_max) or peak_max <= 0:
        return float(fallback)

    ratios = vals / peak_max
    ratios = ratios[np.isfinite(ratios)]

    if len(ratios) == 0:
        return float(fallback)

    frac = float(np.median(ratios))

    frac = float(np.clip(frac, min_frac, max_frac))
    return frac

def identify_events(
    df: pd.DataFrame,
    tol: float | None,
    min_prominence: float,
    min_distance: int,
    width: int | None = None,
    min_separation_h: float | None = None,
) -> pd.DataFrame:
    """
    Identify runoff events from the hourly stormflow series.
    """
    df = df.copy()

    if "Stormflow_mm" not in df.columns:
        raise ValueError("df must contain column 'Stormflow_mm'")

    df["Stormflow_mm"] = pd.to_numeric(df["Stormflow_mm"], errors="coerce").clip(lower=0)
    df = df.dropna(subset=["Stormflow_mm"]).copy()

    if df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].sort_index().copy()

    if df.empty:
        return pd.DataFrame()

    q = df["Stormflow_mm"].values.astype(float)

    positive = df.loc[df["Stormflow_mm"] > 0, "Stormflow_mm"]

    if len(positive) == 0:
        noise_est = 0.0
        q20 = 0.0
    else:
        q30 = positive.quantile(0.30)
        low_vals = positive[positive <= q30]
        if len(low_vals) == 0:
            low_vals = positive.copy()

        med = float(np.median(low_vals))
        mad = float(np.median(np.abs(low_vals - med)))
        noise_est = med + mad
        q20 = float(positive.quantile(0.20))

    if tol is None or pd.isna(tol):
        noise_tol = max(noise_est, 1e-10)
    else:
        noise_tol = max(noise_est, float(tol), 1e-10)

    positive = df.loc[df["Stormflow_mm"] > 0, "Stormflow_mm"]

    if len(positive) == 0:
        low_non_noise = pd.Series(dtype=float)
    else:
        low_non_noise = positive[(positive > noise_tol) & (positive <= q20)]
        if len(low_non_noise) == 0:
            low_non_noise = positive.copy()

    low_flow_ref = float(np.median(low_non_noise)) if len(low_non_noise) else float(q20)

    df["Excess"] = np.clip(df["Stormflow_mm"] - noise_tol, 0.0, None)

    peaks, _ = find_peaks(
        df["Excess"].values.astype(float),
        prominence=min_prominence,
        distance=min_distance,
        width=width,
    )

    peak_threshold = 0.01
    valid_mask = q[peaks] >= peak_threshold
    peaks = peaks[valid_mask]

    if len(peaks) == 0:
        return pd.DataFrame()

    active_mask = q > noise_tol
    active_idx = np.where(active_mask)[0]

    if len(active_idx) == 0:
        return pd.DataFrame()

    breaks = np.where(np.diff(active_idx) > 1)[0]
    seg_starts = [active_idx[0]] + [active_idx[b + 1] for b in breaks]
    seg_ends = [active_idx[b] for b in breaks] + [active_idx[-1]]
    segments = list(zip(seg_starts, seg_ends))

    valid_segments: list[dict[str, Any]] = []

    if min_separation_h is None:
        min_separation_h = 12

    alpha_split = 0.10
    min_persistence_steps = 3
    peak_keep_frac = 0.15

    for seg_start_i, seg_end_i in segments:
        seg_df = df.iloc[seg_start_i:seg_end_i + 1].copy()
        if seg_df.empty:
            continue

        seg_max = float(seg_df["Stormflow_mm"].max())
        if seg_max < peak_threshold:
            continue

        seg_peak_idx = [pk for pk in peaks if seg_start_i <= pk <= seg_end_i]

        if len(seg_peak_idx) == 0:
            local_idx = int(np.argmax(seg_df["Stormflow_mm"].values))
            peak_t = seg_df.index[local_idx]
            peak_h = float(seg_df.iloc[local_idx]["Stormflow_mm"])

            if peak_h < peak_threshold:
                continue

            valid_segments.append(
                {
                    "Start": seg_df.index[0],
                    "End": seg_df.index[-1],
                    "Peak": peak_t,
                    "Peak_height": peak_h,
                    "First_Peak": peak_t,
                    "First_Peak_height": peak_h,
                    "Max_Peak": peak_t,
                    "Max_Peak_height": peak_h,
                    "segment": seg_df.copy(),
                    "Split_Runoff": False,
                }
            )
            continue

        peak_info = []
        for pk in sorted(seg_peak_idx):
            ts = df.index[pk]
            h = float(q[pk])
            peak_info.append((pk, ts, h))

        seg_peak_max = max(h for _, _, h in peak_info)

        peak_info_main = [
            (pk_i, pk_t, pk_h)
            for (pk_i, pk_t, pk_h) in peak_info
            if pk_h >= peak_keep_frac * seg_peak_max
        ]

        if len(peak_info_main) == 0:
            peak_info_main = peak_info.copy()

        blocks = [[0]]

        for k in range(len(peak_info_main) - 1):
            pk1_i, pk1_t, pk1_h = peak_info_main[k]
            pk2_i, pk2_t, pk2_h = peak_info_main[k + 1]

            gap_h = (pk2_t - pk1_t).total_seconds() / 3600.0

            if gap_h < min_separation_h:
                blocks[-1].append(k + 1)
                continue

            inter_slice = df.iloc[pk1_i:pk2_i + 1]
            if inter_slice.empty:
                blocks[-1].append(k + 1)
                continue

            min_val_idx = inter_slice["Stormflow_mm"].idxmin()
            min_val = float(inter_slice.loc[min_val_idx, "Stormflow_mm"])

            split_threshold = max(noise_tol, alpha_split * pk1_h)

            below_mask = inter_slice["Stormflow_mm"] <= split_threshold
            n_below = int(below_mask.sum())

            if (min_val <= split_threshold) and (n_below >= min_persistence_steps):
                blocks.append([k + 1])
            else:
                blocks[-1].append(k + 1)

        if len(blocks) == 1:
            block_bounds = [(seg_df.index[0], seg_df.index[-1])]
        else:
            split_times_between_blocks = []

            for b in range(len(blocks) - 1):
                last_peak_curr = blocks[b][-1]
                first_peak_next = blocks[b + 1][0]

                i1 = peak_info_main[last_peak_curr][0]
                i2 = peak_info_main[first_peak_next][0]

                inter_slice = df.iloc[i1:i2 + 1]
                min_val_idx = inter_slice["Stormflow_mm"].idxmin()
                split_times_between_blocks.append(min_val_idx)

            starts = [seg_df.index[0]]
            ends = []

            for split_t in split_times_between_blocks:
                ends.append(split_t)

                next_times = seg_df.index[seg_df.index > split_t]
                if len(next_times) > 0:
                    starts.append(next_times[0])

            ends.append(seg_df.index[-1])

            block_bounds = list(zip(starts, ends))

        for b_idx, block in enumerate(blocks):
            ev_start, ev_end = block_bounds[b_idx]
            subseg = df.loc[ev_start:ev_end].copy()

            if subseg.empty:
                continue

            sub_peak_info = [peak_info_main[j] for j in block]
            block_peak_vals = [x[2] for x in sub_peak_info]

            if len(block_peak_vals) == 0:
                continue

            first_peak_frac_main = estimate_first_peak_frac_min(
                block_peak_vals=block_peak_vals,
                min_frac=0.10,
                max_frac=0.80,
                fallback=0.30,
            )

            max_item = max(sub_peak_info, key=lambda x: x[2])
            max_peak_idx = max_item[1]
            max_peak_h = max_item[2]

            if max_peak_h < peak_threshold:
                continue

            if len(block_peak_vals) == 1:
                first_peak_idx = max_peak_idx
                first_peak_h = max_peak_h
            else:
                thr_first = first_peak_frac_main * max_peak_h
                first_peak_idx = None
                first_peak_h = None

                for _, pkt, pkh in sub_peak_info:
                    if pkh >= thr_first:
                        first_peak_idx = pkt
                        first_peak_h = pkh
                        break

                if first_peak_idx is None:
                    first_peak_idx = max_peak_idx
                    first_peak_h = max_peak_h

            valid_segments.append(
                {
                    "Start": subseg.index[0],
                    "End": subseg.index[-1],
                    "Peak": max_peak_idx,
                    "Peak_height": max_peak_h,
                    "First_Peak": first_peak_idx,
                    "First_Peak_height": first_peak_h,
                    "Max_Peak": max_peak_idx,
                    "Max_Peak_height": max_peak_h,
                    "segment": subseg.copy(),
                    "Split_Runoff": len(blocks) > 1,
                }
            )

    events_df = pd.DataFrame(valid_segments)

    if not events_df.empty:
        events_df["Event_Duration_h"] = (
            events_df["End"] - events_df["Start"]
        ).dt.total_seconds() / 3600.0

        events_df["Stormflow_mm"] = events_df.apply(
            lambda row: row["segment"]["Stormflow_mm"].sum(),
            axis=1,
        )

        events_df = events_df.sort_values("Start").reset_index(drop=True)

    if events_df.empty:
        events_df = pd.DataFrame(
            columns=[
                "Start", "End", "Peak", "Peak_height",
                "First_Peak", "First_Peak_height",
                "Max_Peak", "Max_Peak_height",
                "segment", "Split_Runoff",
                "Event_Duration_h", "Stormflow_mm",
                "Noise_tol", "Low_flow",
            ]
        )

    events_df["Noise_tol"] = noise_tol
    events_df["Low_flow"] = low_flow_ref

    return events_df


def identify_rain_events(
    df: pd.DataFrame,
    tol_P: float = 0.1,
    dry_interval_hours: float = 6,
    min_cumulative_P: float = 5,
) -> pd.DataFrame:
    """
    Identify rainfall events from the hourly precipitation series.
    """
    
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    
    df = df[~df.index.isna()].sort_index().copy()

    if "P" not in df.columns:
        raise ValueError("df must contain column 'P'")

    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"])

    if df.empty:
        return pd.DataFrame(columns=["Start", "End", "Total_P", "Peak"])

    is_raining = df["P"] > tol_P

    df["rain_group"] = (
        (is_raining.astype(int).diff().gt(0).cumsum()) * is_raining.astype(int)
    )

    if df["rain_group"].max() == 0:
        return pd.DataFrame(columns=["Start", "End", "Total_P", "Peak"])

    rain_periods = (
        df[is_raining]
        .groupby("rain_group", group_keys=False)
        .apply(
            lambda g: pd.Series(
                {
                    "Start": g.index.min(),
                    "End": g.index.max(),
                    "Total_P": g["P"].sum(),
                    "Peak": g["P"].idxmax(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    merged_events = []

    if not rain_periods.empty:
        current_event_start = rain_periods.iloc[0]["Start"]
        current_event_end = rain_periods.iloc[0]["End"]
        current_event_total_p = rain_periods.iloc[0]["Total_P"]
        current_event_peak = rain_periods.iloc[0]["Peak"]

        for i in range(1, len(rain_periods)):
            next_event = rain_periods.iloc[i]
            time_gap = (next_event["Start"] - current_event_end).total_seconds() / 3600

            if time_gap <= dry_interval_hours:
                current_event_end = next_event["End"]
                current_event_total_p += next_event["Total_P"]

                if df.loc[next_event["Peak"], "P"] > df.loc[current_event_peak, "P"]:
                    current_event_peak = next_event["Peak"]
            else:
                merged_events.append(
                    {
                        "Start": current_event_start,
                        "End": current_event_end,
                        "Total_P": current_event_total_p,
                        "Peak": current_event_peak,
                    }
                )
                current_event_start = next_event["Start"]
                current_event_end = next_event["End"]
                current_event_total_p = next_event["Total_P"]
                current_event_peak = next_event["Peak"]

        merged_events.append(
            {
                "Start": current_event_start,
                "End": current_event_end,
                "Total_P": current_event_total_p,
                "Peak": current_event_peak,
            }
        )

    merged_events_df = pd.DataFrame(merged_events)

    if not merged_events_df.empty:
        merged_events_df = merged_events_df[
            merged_events_df["Total_P"] >= min_cumulative_P
        ].reset_index(drop=True)

    return merged_events_df


def associate_events(
    runoff_events: pd.DataFrame,
    rain_events: pd.DataFrame,
    df_precip: pd.DataFrame,
    df_runoff: pd.DataFrame,
    merge_runoff_peaks: bool = False,
    max_peak_gap_h: float | None = None,
    max_duration_h: float | None = None,
    max_lag_h: float | None = None,
    rain_keep_frac: float = 0.20,
    runoff_ratio_limit: float = 0.90,
    backfill_hours: float = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Associate rainfall and runoff events without saving outputs.
    Returns:
        associated_df, discarded_events_df
    """
    if max_duration_h is None:
        max_duration_h = 360

    if max_lag_h is None:
        max_lag_h = 120

    if merge_runoff_peaks and max_peak_gap_h is None:
        max_peak_gap_h = 3

    if runoff_events.empty or rain_events.empty:
        return pd.DataFrame(), pd.DataFrame()

    runoff_events = runoff_events.copy()
    rain_events = rain_events.copy()
    df_precip_orig = df_precip.copy()
    df_precip_masked = df_precip.copy()
    df_precip_masked["AssignedTo"] = None

    for col in ["Start", "End", "Peak", "First_Peak", "Max_Peak"]:
        if col in runoff_events.columns:
            runoff_events[col] = pd.to_datetime(runoff_events[col], errors="coerce")

    for col in ["Start", "End", "Peak"]:
        if col in rain_events.columns:
            rain_events[col] = pd.to_datetime(rain_events[col], errors="coerce")

    runoff_sorted = runoff_events.sort_values("Start").reset_index(drop=True)
    rain_sorted = rain_events.sort_values("Start").reset_index(drop=True)

    if not merge_runoff_peaks:
        runoff_merged_df = runoff_sorted.copy()
        runoff_merged_df["Peaks"] = runoff_merged_df["First_Peak"].apply(lambda x: [x])
        runoff_merged_df["Merged_Runoff"] = False
    else:
        merged_runoff = []
        i = 0

        while i < len(runoff_sorted):
            row = runoff_sorted.loc[i]
            start, end = row["Start"], row["End"]
            peaks = [row["First_Peak"]]
            stormflow = row.get("Stormflow_mm", 0)

            split_flag = bool(row.get("Split_Runoff", False))

            j = i + 1
            while j < len(runoff_sorted):
                next_row = runoff_sorted.loc[j]
                peak_gap_h = (next_row["First_Peak"] - peaks[-1]).total_seconds() / 3600.0

                if peak_gap_h <= max_peak_gap_h:
                    end = max(end, next_row["End"])
                    peaks.append(next_row["First_Peak"])
                    stormflow += next_row.get("Stormflow_mm", 0)
                    split_flag = split_flag or bool(next_row.get("Split_Runoff", False))
                    j += 1
                else:
                    break

            merged_runoff.append(
                {
                    "Start": start,
                    "End": end,
                    "Peaks": peaks,
                    "First_Peak": row["First_Peak"],
                    "Stormflow_mm": stormflow,
                    "Merged_Runoff": len(peaks) > 1,
                    "Split_Runoff": split_flag,
                    "Max_Peak": row["Max_Peak"] if "Max_Peak" in row.index else row["First_Peak"],
                    "Noise_tol": row["Noise_tol"] if "Noise_tol" in row.index else np.nan,
                    "Low_flow": row["Low_flow"] if "Low_flow" in row.index else np.nan,
                }
            )
            i = j

        runoff_merged_df = pd.DataFrame(merged_runoff)

    assignments = []
    used_rain_idx = set()
    unmatched_events = []

    for ridx, r_row in runoff_merged_df.iterrows():
        rs = r_row["Start"]
        re = r_row["End"]
        runoff_peaks = r_row["Peaks"]
        runoff_volume = r_row["Stormflow_mm"]
        first_peak = r_row["First_Peak"]

        candidates = rain_sorted[
            (~rain_sorted.index.isin(used_rain_idx))
            & (rain_sorted["Start"] <= first_peak)
            & (rain_sorted["Peak"] <= first_peak)
            & (rain_sorted["Start"] <= re)
        ].copy()

        if candidates.empty:
            unmatched_events.append(
                {
                    "Runoff_Start": rs,
                    "Runoff_End": re,
                    "Runoff_Peaks": runoff_peaks,
                    "First_Runoff_Peak": first_peak,
                    "Runoff_Volume_mm": runoff_volume,
                    "Discard_Reason": "no_rain_candidate",
                }
            )
            continue

        candidates["Lag_h"] = (first_peak - candidates["Peak"]).dt.total_seconds() / 3600.0
        candidates = candidates[(candidates["Lag_h"] >= 0) & (candidates["Lag_h"] <= max_lag_h)].copy()

        if candidates.empty:
            unmatched_events.append(
                {
                    "Runoff_Start": rs,
                    "Runoff_End": re,
                    "Runoff_Peaks": runoff_peaks,
                    "First_Runoff_Peak": first_peak,
                    "Runoff_Volume_mm": runoff_volume,
                    "Discard_Reason": "no_rain_within_lag",
                }
            )
            continue

        max_total_p = candidates["Total_P"].max()
        candidates = candidates[candidates["Total_P"] >= rain_keep_frac * max_total_p].copy()

        if candidates.empty:
            unmatched_events.append(
                {
                    "Runoff_Start": rs,
                    "Runoff_End": re,
                    "Runoff_Peaks": runoff_peaks,
                    "First_Runoff_Peak": first_peak,
                    "Runoff_Volume_mm": runoff_volume,
                    "Discard_Reason": "no_relevant_rain_candidate",
                }
            )
            continue

        candidates = candidates.sort_values(by=["Lag_h", "Total_P"], ascending=[True, False])
        best = candidates.iloc[0]
        best_idx = best.name

        assignments.append(
            {
                "runoff_idx": ridx,
                "Runoff_Start": rs,
                "Runoff_End": re,
                "Runoff_Peaks": runoff_peaks,
                "Runoff_Volume_mm": runoff_volume,
                "Rain_idx": best_idx,
                "Rain_Start": best["Start"],
                "Rain_End": best["End"],
                "Rain_End_original": best["End"],
                "Rain_Peak": best["Peak"],
            }
        )

        used_rain_idx.add(best_idx)

    if len(assignments) == 0:
        return pd.DataFrame(), pd.DataFrame(unmatched_events)
    
    assignments = sorted(assignments, key=lambda x: x["Runoff_Start"])
    
    for a in assignments:
        ridx = a["runoff_idx"]
    
        rain_start = a["Rain_Start"]
        rain_end_main = a["Rain_End"]
    
        main_mask_assigned = (
            (df_precip_masked.index >= rain_start)
            & (df_precip_masked.index <= rain_end_main)
            & (df_precip_masked["P"] > 0.0)
        )
    
        df_precip_masked.loc[main_mask_assigned, "AssignedTo"] = f"runoff_{ridx}_main"
        df_precip_masked.loc[main_mask_assigned, "P"] = 0.0

    assignments = sorted(assignments, key=lambda x: x["Runoff_Start"])
    
    associated_events = []

    for a in assignments:
        ridx = a["runoff_idx"]
        run = runoff_merged_df.loc[ridx]

        first_peak = run["First_Peak"]
        runoff_start = run["Start"]
        runoff_end = run["End"]

        rain_start = a["Rain_Start"]
        rain_end_main_raw = a["Rain_End"]
        rain_peak = a["Rain_Peak"]

        rain_end_main = rain_end_main_raw
        
        other_main_starts = [
            x["Rain_Start"] for x in assignments
            if x["runoff_idx"] != ridx
        ]
                
        future_main_starts = [t for t in other_main_starts if t > rain_start]
        
        if future_main_starts:
            next_main_start = min(future_main_starts)
        else:
            next_main_start = pd.NaT
        
        if pd.notna(next_main_start) and next_main_start <= runoff_end:
            residual_stop = next_main_start - pd.Timedelta(hours=1)
        else:
            residual_stop = runoff_end

        residual_mask = (
            (df_precip_masked.index > rain_end_main) &
            (df_precip_masked.index <= residual_stop) &
            (df_precip_masked["P"] > 0.0)
        )

        residual_volume = df_precip_masked.loc[residual_mask, "P"].sum()

        if residual_mask.any():
            residual_start = df_precip_masked.index[residual_mask].min()
            residual_end = df_precip_masked.index[residual_mask].max()
        else:
            residual_start = pd.NaT
            residual_end = pd.NaT

        df_precip_masked.loc[residual_mask, "AssignedTo"] = f"runoff_{ridx}_residual"
        df_precip_masked.loc[residual_mask, "P"] = 0.0

        previous_rain_end = pd.NaT
        
        if associated_events:
            previous_rain_ends = [
                ev["Rain_End"]
                for ev in associated_events
                if pd.notna(ev.get("Rain_End"))
            ]
        
            if previous_rain_ends:
                previous_rain_end = max(previous_rain_ends)
        
        shifted_rain_start = rain_start
        aligned_runoff_start = runoff_start
        new_runoff_start = runoff_start
        backfill_start = pd.NaT
        backfill_end = pd.NaT
        
        if runoff_start < rain_start:
        
            low_flow = run.get("Low_flow", np.nan)
        
            if pd.notna(low_flow) and first_peak in df_runoff.index:
                s = df_runoff["Stormflow_mm"]
                loc = s.index.get_loc(first_peak)
        
                while loc > 0 and s.iloc[loc - 1] > low_flow:
                    loc -= 1
        
                new_runoff_start = s.index[loc]
            else:
                new_runoff_start = runoff_start
        
            candidate_search_start = new_runoff_start - pd.Timedelta(hours=backfill_hours)
        
            if pd.notna(previous_rain_end):
                search_start = max(
                    candidate_search_start,
                    previous_rain_end + pd.Timedelta(hours=1),
                    df_precip_masked.index.min(),
                )
            else:
                search_start = max(
                    candidate_search_start,
                    df_precip_masked.index.min(),
                )
        
            back_mask = (
                (df_precip_masked.index >= search_start)
                & (df_precip_masked.index < rain_start)
                & (df_precip_masked["P"] > 0.0)
            )
        
            if back_mask.any():
                back_times = df_precip_masked.index[back_mask]
        
                shifted_rain_start = back_times.min()
                backfill_start = back_times.min()
                backfill_end = back_times.max()
        
                search_runoff_mask = (
                    (df_runoff.index >= shifted_rain_start)
                    & (df_runoff.index <= runoff_start)
                )
        
                runoff_segment = df_runoff.loc[search_runoff_mask, "Stormflow_mm"]
        
                if not runoff_segment.empty:
                    aligned_runoff_start = runoff_segment.idxmin()
                else:
                    aligned_runoff_start = new_runoff_start
        
                df_precip_masked.loc[back_mask, "AssignedTo"] = f"runoff_{ridx}_backfill"
                df_precip_masked.loc[back_mask, "P"] = 0.0
        
            else:
                shifted_rain_start = rain_start
                aligned_runoff_start = new_runoff_start
        
        if aligned_runoff_start < shifted_rain_start:
            aligned_runoff_start = shifted_rain_start

        if pd.notna(residual_end):
            rain_end = min(max(rain_end_main, residual_end), runoff_end)
        else:
            rain_end = min(rain_end_main, runoff_end)

        aligned_rain_end = rain_end
        aligned_runoff_end = runoff_end

        if aligned_runoff_start < shifted_rain_start:
            aligned_runoff_start = shifted_rain_start

        if aligned_runoff_end < aligned_runoff_start or aligned_rain_end < shifted_rain_start:
            unmatched_events.append(
                {
                    "Runoff_Start": aligned_runoff_start,
                    "Runoff_End": aligned_runoff_end,
                    "Runoff_Peaks": run["Peaks"],
                    "First_Runoff_Peak": first_peak,
                    "Rain_Start": shifted_rain_start,
                    "Rain_End": aligned_rain_end,
                    "Rain_Peak": rain_peak,
                    "Discard_Reason": "invalid_temporal_bounds_after_alignment",
                }
            )
            continue

        rain_mask_final = (
            (df_precip_orig.index >= shifted_rain_start)
            & (df_precip_orig.index <= aligned_rain_end)
            & (df_precip_orig["P"] > 0.0)
        )
        aligned_rain_volume = df_precip_orig.loc[rain_mask_final, "P"].sum()

        run_mask_final = (
            (df_runoff.index >= aligned_runoff_start)
            & (df_runoff.index <= aligned_runoff_end)
        )
        aligned_runoff_volume = df_runoff.loc[run_mask_final, "Stormflow_mm"].sum()

        total_event_h = (aligned_runoff_end - shifted_rain_start).total_seconds() / 3600.0

        if shifted_rain_start > aligned_runoff_end:
            unmatched_events.append(
                {
                    "Runoff_Start": aligned_runoff_start,
                    "Runoff_End": aligned_runoff_end,
                    "Runoff_Peaks": run["Peaks"],
                    "First_Runoff_Peak": first_peak,
                    "Runoff_Volume_mm": aligned_runoff_volume,
                    "Rain_Start": shifted_rain_start,
                    "Rain_End": aligned_rain_end,
                    "Rain_Peak": rain_peak,
                    "Discard_Reason": "rain_starts_after_runoff_end",
                }
            )
            continue

        if rain_peak > first_peak:
            unmatched_events.append(
                {
                    "Runoff_Start": aligned_runoff_start,
                    "Runoff_End": aligned_runoff_end,
                    "Runoff_Peaks": run["Peaks"],
                    "First_Runoff_Peak": first_peak,
                    "Runoff_Volume_mm": aligned_runoff_volume,
                    "Rain_Start": shifted_rain_start,
                    "Rain_End": aligned_rain_end,
                    "Rain_Peak": rain_peak,
                    "Discard_Reason": "rain_peak_after_first_runoff_peak",
                }
            )
            continue

        if total_event_h < 0:
            unmatched_events.append(
                {
                    "Runoff_Start": aligned_runoff_start,
                    "Runoff_End": aligned_runoff_end,
                    "Runoff_Peaks": run["Peaks"],
                    "First_Runoff_Peak": first_peak,
                    "Runoff_Volume_mm": aligned_runoff_volume,
                    "Rain_Start": shifted_rain_start,
                    "Rain_End": aligned_rain_end,
                    "Rain_Peak": rain_peak,
                    "Discard_Reason": "negative_total_event_duration",
                }
            )
            continue

        if not (shifted_rain_start <= first_peak <= aligned_runoff_end):
            unmatched_events.append(
                {
                    "Runoff_Start": aligned_runoff_start,
                    "Runoff_End": aligned_runoff_end,
                    "Runoff_Peaks": run["Peaks"],
                    "First_Runoff_Peak": first_peak,
                    "Runoff_Volume_mm": aligned_runoff_volume,
                    "Rain_Start": shifted_rain_start,
                    "Rain_End": aligned_rain_end,
                    "Rain_Peak": rain_peak,
                    "Discard_Reason": "reference_runoff_peak_outside_rain_runoff_window",
                }
            )
            continue

        associated_events.append(
            {
                "Runoff_Start": aligned_runoff_start,
                "Runoff_End": aligned_runoff_end,
                "Runoff_Peaks": run["Peaks"],
                "Runoff_Volume_mm": aligned_runoff_volume,
                "Rain_Start": shifted_rain_start,
                "Rain_End": aligned_rain_end,
                "Rain_End_original": a["Rain_End_original"],
                "Rain_End_main": rain_end_main,
                "Rain_Peak": rain_peak,
                "Rain_Volume_mm": aligned_rain_volume,
                "Rain_EventID": a["Rain_idx"],
                "PeakLag_h": (first_peak - rain_peak).total_seconds() / 3600.0,
                "RainStart_to_RunoffPeak_h": (first_peak - shifted_rain_start).total_seconds() / 3600.0,
                "RainEnd_to_RunoffPeak_h": (first_peak - a["Rain_End_original"]).total_seconds() / 3600.0,
                "Total_Event_h": total_event_h,
                "First_Runoff_Peak": first_peak,
                "Merged_Runoff": run.get("Merged_Runoff", False),
                "Split_Runoff": run.get("Split_Runoff", False),
                "Residual_Rain_mm": residual_volume,
                "Residual_Rain_Start": residual_start,
                "Residual_Rain_End": residual_end,
                "Backfill_Rain_Start": backfill_start,
                "Backfill_Rain_End": backfill_end,
            }
        )

    associated_df = pd.DataFrame(associated_events)

    if associated_df.empty:
        return associated_df, pd.DataFrame(unmatched_events)

    mask_valid = associated_df["Runoff_Volume_mm"] < runoff_ratio_limit * associated_df["Rain_Volume_mm"]
    discarded_due_to_efficiency = associated_df.loc[~mask_valid].copy()
    associated_df = associated_df.loc[mask_valid].reset_index(drop=True)

    for i, row in associated_df.iterrows():
        mask_rain = (
            (df_precip.index >= row["Rain_Start"])
            & (df_precip.index <= row["Rain_End"])
        )
        rain_values = df_precip.loc[mask_rain, "P"]

        associated_df.loc[i, "Imax_mm/h"] = rain_values.max()
        associated_df.loc[i, "Imean_mm/h"] = (
            rain_values[rain_values > 0].mean() if (rain_values > 0).any() else 0
        )

    duration_hours = (
        associated_df["Runoff_End"] - associated_df["Runoff_Start"]
    ).dt.total_seconds() / 3600.0

    long_events_mask = duration_hours <= max_duration_h

    discarded_due_to_length = associated_df.loc[~long_events_mask].copy()
    associated_df = associated_df.loc[long_events_mask].reset_index(drop=True)

    discarded_due_to_efficiency["Discard_Reason"] = "efficiency"
    discarded_due_to_length["Discard_Reason"] = "length"

    discarded_events = pd.concat(
        [discarded_due_to_efficiency, discarded_due_to_length, pd.DataFrame(unmatched_events)],
        ignore_index=True,
        sort=False,
    )

    return associated_df, discarded_events


def final_clean_events(
    associated_events_df: pd.DataFrame,
    max_duration_h: float | None = None,
    max_lag_h: float | None = None,
    rc_limit: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Final cleaning of associated rainfall–runoff events.
    Returns:
        clean_events_df, final_outliers_df
    """
    if associated_events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    associated_events_df = associated_events_df.copy()

    associated_events_df["RC"] = (
        associated_events_df["Runoff_Volume_mm"] / associated_events_df["Rain_Volume_mm"]
    )
    
    invalid_rc_mask = ~associated_events_df["RC"].between(0, 1)
    
    invalid_rc_outliers = associated_events_df.loc[invalid_rc_mask].copy()
    invalid_rc_outliers["Discard_Reason"] = "invalid_rc"
    
    associated_events_df = associated_events_df.loc[~invalid_rc_mask].copy()
    
    if associated_events_df.empty:
        return pd.DataFrame(), invalid_rc_outliers

    associated_events_df["Duration_h"] = associated_events_df["Total_Event_h"]

    q1_d = associated_events_df["Duration_h"].quantile(0.25)
    q3_d = associated_events_df["Duration_h"].quantile(0.75)
    iqr_d = q3_d - q1_d
    upper_limit_d = q3_d + 1.5 * iqr_d

    q1_l = associated_events_df["PeakLag_h"].quantile(0.25)
    q3_l = associated_events_df["PeakLag_h"].quantile(0.75)
    iqr_l = q3_l - q1_l
    upper_limit_l = q3_l + 1.5 * iqr_l

    final_max_duration_h = 360 if max_duration_h is None else max_duration_h
    final_max_lag_h = 120 if max_lag_h is None else max_lag_h

    outlier_mask = (
        (associated_events_df["RC"] >= rc_limit)
        | (associated_events_df["Duration_h"] > upper_limit_d)
        | (associated_events_df["Duration_h"] > final_max_duration_h)
        | (associated_events_df["PeakLag_h"] > upper_limit_l)
        | (associated_events_df["PeakLag_h"] > final_max_lag_h)
    )

    final_outliers_df = associated_events_df.loc[outlier_mask].copy()
    final_outliers_df["Discard_Reason"] = "final_filter"
    
    clean_events_df = associated_events_df.loc[~outlier_mask].copy()
    
    final_outliers_df = pd.concat(
        [invalid_rc_outliers, final_outliers_df],
        ignore_index=True,
        sort=False,
    )
    
    return clean_events_df, final_outliers_df

def simulate_linear_reservoir(P, rc, kd):
    P = np.asarray(P, dtype=float)
    qsim = np.zeros_like(P, dtype=float)
    S = 0.0

    for t in range(len(P)):
        qsim[t] = S / kd
        S = S + rc * P[t] - qsim[t]
        S = max(S, 0.0)

    return qsim


def fit_event_linear_reservoir(P, Qobs):
    P = np.asarray(P, dtype=float)
    Qobs = np.asarray(Qobs, dtype=float)

    mask = np.isfinite(P) & np.isfinite(Qobs)
    P = P[mask]
    Qobs = Qobs[mask]

    if len(P) < 3 or np.nansum(P) <= 0 or np.nansum(Qobs) <= 0:
        return np.nan, np.nan, np.nan

    def objective(params):
        rc, kd = params
        qsim = simulate_linear_reservoir(P, rc, kd)
        return np.sqrt(np.mean((Qobs - qsim) ** 2))

    result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0), (0.5, 200.0)],
        polish=True,
        seed=42,
    )

    rc_fit, kd_fit = result.x
    rmse = result.fun

    return rc_fit, kd_fit, rmse

def add_linear_reservoir_fit(
    events_df: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_runoff: pd.DataFrame,
) -> pd.DataFrame:

    events_df = events_df.copy()

    events_df["RC_lr"] = np.nan
    events_df["kd_lr_h"] = np.nan
    events_df["RMSE_lr"] = np.nan

    for i, row in events_df.iterrows():

        start = min(row["Rain_Start"], row["Runoff_Start"])
        end = row["Runoff_End"]

        P = df_rain.loc[start:end, "P"].copy()
        Qobs = df_runoff.loc[start:end, "Stormflow_mm"].copy()

        common_index = P.index.intersection(Qobs.index)
        P = P.loc[common_index].values
        Qobs = Qobs.loc[common_index].values

        rc_fit, kd_fit, rmse = fit_event_linear_reservoir(P, Qobs)

        events_df.loc[i, "RC_lr"] = rc_fit
        events_df.loc[i, "kd_lr_h"] = kd_fit
        events_df.loc[i, "RMSE_lr"] = rmse

    return events_df

def rainfall_runoff_event(
    df_runoff: pd.DataFrame,
    df_rain: pd.DataFrame,
    basin_name: str | None = None,
    area_km2: float | None = None,
    dt_seconds: float | None = None,
    stormflow_col: str = "Stormflow",
    tol: float | None = None,
    min_prominence: float = 0.001,
    min_distance: int = 3,
    width: int | None = None,
    min_separation_h: float | None = None,
    tol_P: float = 0.1,
    dry_interval_hours: float = 6,
    min_cumulative_P: float = 5,
    merge_runoff_peaks: bool = False,
    max_peak_gap_h: float | None = None,
    max_duration_h: float | None = 360,
    max_lag_h: float | None = 120,
    rain_keep_frac: float = 0.20,
    runoff_ratio_limit: float = 0.90,
    backfill_hours: float = 6,
    final_rc_limit: float = 0.90,
) -> dict[str, pd.DataFrame | str | None]:
    """
    Complete workflow for rainfall–runoff event detection and cleaning.

    Returns a dictionary with:
        - basin_name
        - runoff_events
        - rain_events
        - associated_events
        - discarded_events
        - clean_events
        - final_outliers
    """
    
    df_runoff = df_runoff.copy()
    df_rain = df_rain.copy()

    if not isinstance(df_runoff.index, pd.DatetimeIndex):
        if "Date" in df_runoff.columns:
            df_runoff["Date"] = pd.to_datetime(df_runoff["Date"], errors="coerce")
            df_runoff = df_runoff.set_index("Date")
        else:
            df_runoff.index = pd.to_datetime(df_runoff.index, errors="coerce")

    if not isinstance(df_rain.index, pd.DatetimeIndex):
        if "Date" in df_rain.columns:
            df_rain["Date"] = pd.to_datetime(df_rain["Date"], errors="coerce")
            df_rain = df_rain.set_index("Date")
        else:
            df_rain.index = pd.to_datetime(df_rain.index, errors="coerce")

    df_runoff = df_runoff[~df_runoff.index.isna()].sort_index().copy()
    df_rain = df_rain[~df_rain.index.isna()].sort_index().copy()

    if "Stormflow_mm" not in df_runoff.columns:
        if area_km2 is None:
            raise ValueError(
                "Stormflow_mm not found in df_runoff. "
                "Provide area_km2 and dt_seconds to convert from m³/s."
            )

        if dt_seconds is None:
            dt_seconds = df_runoff.index.to_series().diff().median().total_seconds()

            if pd.isna(dt_seconds) or dt_seconds <= 0:
                raise ValueError("Could not infer a valid dt_seconds from df_runoff index.")

        if stormflow_col not in df_runoff.columns:
            raise ValueError(f"Column '{stormflow_col}' not found in df_runoff.")

        df_runoff["Stormflow_mm"] = discharge_to_mm(
            pd.to_numeric(df_runoff[stormflow_col], errors="coerce"),
            area_km2=area_km2,
            dt_seconds=dt_seconds,
        )

    df_flow = df_runoff[["Stormflow_mm"]].copy()
    
    df_flow["Stormflow_mm"] = (
        pd.to_numeric(df_flow["Stormflow_mm"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    
    df_precip = df_rain[["P"]].copy()
    
    df_precip["P"] = (
        pd.to_numeric(df_precip["P"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    df_flow.index = pd.to_datetime(df_flow.index, errors="coerce").round("h")
    df_precip.index = pd.to_datetime(df_precip.index, errors="coerce").round("h")

    df_flow = df_flow[~df_flow.index.isna()].sort_index()
    df_precip = df_precip[~df_precip.index.isna()].sort_index()

    full_index = pd.date_range(
        start=df_precip.index.min(),
        end=df_precip.index.max(),
        freq="1h",
    )
    df_precip = df_precip.reindex(full_index, fill_value=0)

    df = df_flow.merge(df_precip, left_index=True, right_index=True, how="inner")

    if "Stormflow_mm" not in df.columns or "P" not in df.columns:
        raise ValueError("Columns 'Stormflow_mm' or 'P' not found after merge.")

    df_runoff = df[["Stormflow_mm"]].copy()
    df_rain = df[["P"]].copy()
    
    runoff_events = identify_events(
        df=df_runoff,
        tol=tol,
        min_prominence=min_prominence,
        min_distance=min_distance,
        width=width,
        min_separation_h=min_separation_h,
    )

    rain_events = identify_rain_events(
        df=df_rain,
        tol_P=tol_P,
        dry_interval_hours=dry_interval_hours,
        min_cumulative_P=min_cumulative_P,
    )

    associated_events, discarded_events = associate_events(
        runoff_events=runoff_events,
        rain_events=rain_events,
        df_precip=df_rain,
        df_runoff=df_runoff,
        merge_runoff_peaks=merge_runoff_peaks,
        max_peak_gap_h=max_peak_gap_h,
        max_duration_h=max_duration_h,
        max_lag_h=max_lag_h,
        rain_keep_frac=rain_keep_frac,
        runoff_ratio_limit=runoff_ratio_limit,
        backfill_hours=backfill_hours,
    )

    clean_events, final_outliers = final_clean_events(
        associated_events_df=associated_events,
        max_duration_h=max_duration_h,
        max_lag_h=max_lag_h,
        rc_limit=final_rc_limit,
    )
    
    if not clean_events.empty:
        clean_events = add_linear_reservoir_fit(
            events_df=clean_events,
            df_rain=df_rain,
            df_runoff=df_runoff,
        )

    if basin_name is not None and not clean_events.empty:
        clean_events = clean_events.copy()
        clean_events["Event_ID"] = [f"{basin_name}_E{i+1:03d}" for i in range(len(clean_events))]
        clean_events = clean_events.set_index("Event_ID")

    return {
        "basin_name": basin_name,
        "df_runoff": df_runoff,
        "df_rain": df_rain,
        "runoff_events": runoff_events,
        "rain_events": rain_events,
        "associated_events": associated_events,
        "discarded_events": discarded_events,
        "clean_events": clean_events,
        "final_outliers": final_outliers,
    }