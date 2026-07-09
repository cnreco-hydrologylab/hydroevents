# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:32:44 2026

@author: sofia
"""

from hydroevents import (
    preprocess_streamflow,
    compute_mrc,
    separate_baseflow,
    rainfall_runoff_event,
)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


#q_file = r"C:\Users\hydroevents\example\data\Q_example.xlsx"
#p_file = r"C:\Users\hydroevents\example\data\P_example.xlsx"

q_file = Path(
    r"C:\hydroevents\example\data\Q_example.xlsx")
p_file = Path(
    r"C:\hydroevents\example\data\P_example.xlsx")

out_dir = q_file.parent / "output_demo_workflow"
out_dir.mkdir(parents=True, exist_ok=True)

date_col = "Date"
q_col = "Q"
p_col = "P"

Area_km2= 137

# Choose the window used for the figures; if None, first year is used
plot_start_tot = None   # e.g. "2003-01-01"
plot_end_tot = None     # e.g. "2003-03-31"

df_q = pd.read_excel(q_file)
df_p = pd.read_excel(p_file)

print("\n--- INPUT DATA ---")
print(f"Q columns: {list(df_q.columns)}")
print(f"P columns: {list(df_p.columns)}")


df_q["Date"] = pd.to_datetime(df_q["Date"])
df_p["Date"] = pd.to_datetime(df_p["Date"])

df_processed, df_daily, summary = preprocess_streamflow(
    df_q,
    date_col="Date",
    q_col="Q",
    apply_gap_filling=True,
)

print("\n--- PREPROCESSING ---")
print(summary)

mrc_results = compute_mrc(df_daily)

print("\n--- MRC ---")
print(f"alpha: {mrc_results['fit_results']['alpha']:.4f}")
print(f"k_day: {mrc_results['fit_results']['k_day']:.4f}")
print(f"k_hour: {mrc_results['fit_results']['k_hour']:.4f}")
print(f"R²: {mrc_results['fit_results']['R2']:.2f}")
print(f"Valid recession segments: {len(mrc_results['valid_segments'])}")


bf_results = separate_baseflow(
    df_processed,
    q_col="Q",
    date_col="Date",
    mrc_results=mrc_results,
    k_method="mrc",
    nan_to_zero=True,
    direction = 'f'
)

print("\n--- BASEFLOW ---")
print(f"BFI: {bf_results['bfi']:.3f}")

df_runoff = bf_results["df"].copy()

df_runoff.index = pd.to_datetime(df_runoff.index)

df_rain = df_p.copy()
df_rain = df_rain.set_index("Date")
df_rain.index = pd.to_datetime(df_rain.index)

df_rain["P"] = pd.to_numeric(df_rain["P"], errors="coerce")
df_rain = df_rain.sort_index()


events_results = rainfall_runoff_event(
    df_runoff=df_runoff,
    df_rain=df_rain,
    basin_name="test_basin",
    area_km2 = Area_km2,
    min_prominence=0.001,
    max_duration_h=360,
    max_lag_h=120,
)

runoff_events_df = events_results["runoff_events"]
rain_events_df = events_results["rain_events"]
associated_events_df = events_results["associated_events"]
discarded_events_df = events_results["discarded_events"]
clean_events_df = events_results["clean_events"]
final_outliers_df = events_results["final_outliers"]

print("\n--- EVENTS ---")
print(f"Runoff events: {len(runoff_events_df)}")
print(f"Rain events: {len(rain_events_df)}")
print(f"Associated events: {len(associated_events_df)}")
print(f"Clean events: {len(clean_events_df)}")
print(f"Final outliers: {len(final_outliers_df)}")

df = clean_events_df.copy()

print("\n--- OUTPUT DATAFRAMES ---")

dfs_info = {
    "runoff_events_df": runoff_events_df,
    "rain_events_df": rain_events_df,
    "associated_events_df": associated_events_df,
    "clean_events_df": clean_events_df,
}

for name, dfi in dfs_info.items():
    print(f"\n{name}")
    print(f"Rows: {len(dfi)}")
    print(f"Columns: {list(dfi.columns)}")

# --------------------------------------------------
# CONTROL PLOTS
# --------------------------------------------------

# Select plotting window
if plot_start_tot is None:
    plot_start = df_runoff.index.min()
else:
    plot_start = pd.to_datetime(plot_start_tot)

if plot_end_tot is None:
    plot_end = plot_start + pd.DateOffset(years=1)
else:
    plot_end = pd.to_datetime(plot_end_tot)

df_runoff_plot = df_runoff.loc[plot_start:plot_end].copy()
df_rain_plot = df_rain.loc[plot_start:plot_end].copy()

events_plot = clean_events_df[
    (pd.to_datetime(clean_events_df["Rain_Start"]) >= plot_start)
    & (pd.to_datetime(clean_events_df["Rain_Start"]) <= plot_end)
].copy()

# =========================
# MRC SEGMENTS
# =========================

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    df_daily[date_col],
    df_daily[q_col],
    color="0.75",
    linewidth=1,
    label="Daily streamflow",
    zorder=1,
)

for i, seg in enumerate(mrc_results["valid_segments"]):
    if "baseflow_segment_ln" in seg:
        rec = seg["baseflow_segment_ln"]
        ax.plot(
            rec[date_col],
            rec[q_col],
            color="tab:red",
            linewidth=1.8,
            label="Selected recession segment" if i == 0 else None,
            zorder=3,
        )
        ax.scatter(
            rec[date_col].iloc[0],
            rec[q_col].iloc[0],
            color="black",
            s=12,
            marker="o",
            label="Inflection point" if i == 0 else None,
            zorder=4,
        )

ax.set_xlabel("Date")
ax.set_ylabel("Q [m³/s]")
ax.set_title("Recession-segment selection")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(out_dir / "01_recession_segments.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# =========================
# MASTER RECESSION CURVE
# =========================

fit_results = mrc_results["fit_results"]
df_mrc = mrc_results["df_mrc"]

if fit_results is not None and not df_mrc.empty:
    alpha = fit_results["alpha"]
    q0 = fit_results["Q0"]
    r2 = fit_results["R2"]

    t_fit = np.linspace(df_mrc["t_global"].min(), df_mrc["t_global"].max(), 200)
    q_fit = q0 * np.exp(-alpha * t_fit)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.scatter(
        df_mrc["t_global"],
        df_mrc[q_col],
        s=10,
        color="gray",
        alpha=0.7,
        label="Aligned recession points",
    )

    ax.plot(
        t_fit,
        q_fit,
        color="black",
        linewidth=2,
        label="Maillet fit",
    )

    textstr = (
        r"$\alpha$ = " + f"{alpha:.4f} d$^{{-1}}$\n"
        + r"$R^2$ = " + f"{r2:.2f}"
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("Shifted time [days]")
    ax.set_ylabel("Q [m³/s]")
    ax.set_title("MRC Maillet fit")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "02_mrc.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# =========================
# BASEFLOW SEPARATION
# =========================

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    df_runoff_plot.index,
    df_runoff_plot["Q_input"],
    color="black",
    linewidth=1.1,
    label="Total discharge",
)

ax.plot(
    df_runoff_plot.index,
    df_runoff_plot["Baseflow"],
    color="tab:blue",
    linewidth=1.2,
    label="Baseflow",
)

ax.fill_between(
    df_runoff_plot.index,
    df_runoff_plot["Baseflow"],
    df_runoff_plot["Q_input"],
    color="lightgray",
    alpha=0.6,
    label="Direct runoff",
)

ax.set_xlabel("Date")
ax.set_ylabel("Q [m³/s]")
ax.set_title("Baseflow separation")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(out_dir / "03_baseflow.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# =========================
# EVENT IDENTIFICATION
# =========================

df_runoff_html = df_runoff.copy()
df_rain_html = df_rain.copy()
events_html = clean_events_df.copy()

fig = go.Figure()

fig.add_trace(
    go.Scattergl(
        x=df_runoff_html.index,
        y=df_runoff_html["Stormflow"],
        mode="lines",
        name="Stormflow",
        line=dict(color="black", width=1),
    )
)

fig.add_trace(
    go.Bar(
        x=df_rain_html.index,
        y=df_rain_html[p_col],
        name="Rainfall",
        yaxis="y2",
        marker=dict(color="cornflowerblue", line=dict(color="cornflowerblue", width=2)),
        opacity=0.85,
    )
)

main_peak_x = []
main_peak_y = []

for i, row in events_html.iterrows():
    if pd.isna(row.get("Runoff_Start")) or pd.isna(row.get("Runoff_End")):
        continue

    ev_start = pd.to_datetime(row["Runoff_Start"])
    ev_end = pd.to_datetime(row["Runoff_End"])

    peaks = row.get("Runoff_Peaks", [])
    if isinstance(peaks, (list, tuple)) and len(peaks) > 0:
        peak_time = max(
            peaks,
            key=lambda p: df_runoff_html.loc[p, "Stormflow"] if p in df_runoff_html.index else -np.inf,
        )
        if peak_time in df_runoff_html.index:
            main_peak_x.append(peak_time)
            main_peak_y.append(df_runoff_html.loc[peak_time, "Stormflow"])

    event_runoff = df_runoff_html.loc[ev_start:ev_end, "Stormflow"]

    fig.add_vrect(
        x0=ev_start,
        x1=ev_end,
        fillcolor="rgba(118,130,100,0.3)",
        opacity=0.18,
        line_width=0,
        layer="below",
    )

    fig.add_trace(
        go.Scattergl(
            x=event_runoff.index,
            y=event_runoff.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(123,13,17,0.3)",
            line=dict(color="rgba(255,140,0,0)"),
            name="Event runoff" if i == events_html.index[0] else None,
            showlegend=i == events_html.index[0],
        )
    )

fig.add_trace(
    go.Scattergl(
        x=main_peak_x,
        y=main_peak_y,
        mode="markers",
        name="Main runoff peak",
        marker=dict(symbol="diamond", size=5, color="firebrick", line=dict(color="black", width=0.8)),
    )
)

fig.update_layout(
    template="plotly_white",
    width=1100,
    height=500,
    xaxis=dict(title="Date", rangeslider=dict(visible=True, thickness=0.15), type="date"),
    yaxis=dict(title="Stormflow [mm/h]", rangemode="tozero"),
    yaxis2=dict(title="Rainfall [mm/h]", overlaying="y", side="right", autorange="reversed"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.write_html(out_dir / "04_event_identification.html", include_plotlyjs="cdn")
fig.show()

# =========================
# EVENT-SCALE METRICS
# =========================

rmse_threshold = 0.05

events_metrics = clean_events_df.copy()

valid = (
    events_metrics["Rain_Volume_mm"].notna()
    & events_metrics["Runoff_Volume_mm"].notna()
)

events_metrics = events_metrics.loc[valid].copy()

valid_sim = (
    events_metrics["RC_lr"].notna()
    & events_metrics["RMSE_lr"].notna()
    & (events_metrics["RMSE_lr"] <= rmse_threshold)
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 4),
)

#P vs R

ax = axes[0]

ax.scatter(
    events_metrics["Rain_Volume_mm"],
    events_metrics["Runoff_Volume_mm"],
    s=30,
    color="black",
    alpha=0.75,
)

ax.set_xlabel("Rainfall volume [mm]")
ax.set_ylabel("Runoff volume [mm]")
ax.set_title("(a) P vs R")
ax.grid(alpha=0.3)

xmax = events_metrics["Rain_Volume_mm"].max() * 1.05
ymax = events_metrics["Runoff_Volume_mm"].max() * 1.05

ax.set_xlim(0, xmax if np.isfinite(xmax) and xmax > 0 else 1)
ax.set_ylim(0, ymax if np.isfinite(ymax) and ymax > 0 else 1)

#P vs RC obs / RC sim

ax = axes[1]

ax.scatter(
    events_metrics["Rain_Volume_mm"],
    events_metrics["RC"],
    s=30,
    marker="o",
    alpha=0.75,
    label=r"$RC_{obs}$",
)

ax.scatter(
    events_metrics.loc[valid_sim, "Rain_Volume_mm"],
    events_metrics.loc[valid_sim, "RC_lr"],
    s=35,
    marker="X",
    alpha=0.80,
    label=rf"$RC_{{sim}}$ | RMSE $\leq$ {rmse_threshold}",
)

ax.set_xlabel("Rainfall volume [mm]")
ax.set_ylabel("Runoff coefficient [-]")
ax.set_title("(b) P vs RC")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(frameon=False)

xmax = events_metrics["Rain_Volume_mm"].max() * 1.05
ax.set_xlim(0, xmax if np.isfinite(xmax) and xmax > 0 else 1)

plt.tight_layout()

plt.savefig(
    out_dir / "figure_04_event_scale_metrics.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


print(f"\nFigures saved to: {out_dir}")