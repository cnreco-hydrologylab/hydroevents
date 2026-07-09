# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:09:33 2026

@author: s.ortenzi
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from hydroevents import (
    preprocess_streamflow,
    compute_mrc,
    separate_baseflow,
    rainfall_runoff_event,
)

# =========================
# PATH
# =========================

input_file = Path(r".\Q_example.xlsx")
p_file = Path(r".\P_example.xlsx")
out_dir = Path(r".\output")
out_dir.mkdir(parents=True, exist_ok=True)

date_col = "Date"
q_col = "Q"
p_col = "P"

plot_start_tot = "2003-01-01"
plot_end_tot= "2003-03-31"

# =========================
# READ SYNTHETIC DATA
# =========================
df_q = pd.read_excel(input_file)
df_q[date_col] = pd.to_datetime(df_q[date_col])
df_q = df_q.sort_values(date_col)

df_p = pd.read_excel(p_file)
df_p[date_col] = pd.to_datetime(df_p[date_col])
df_rain = df_p.copy()
df_rain = df_rain.set_index(date_col)
df_rain.index = pd.to_datetime(df_rain.index)
df_rain[p_col] = pd.to_numeric(df_rain[p_col], errors="coerce")
df_rain = df_rain.sort_index()

# =========================
# LIGHT SMOOTHING FOR
# SYNTHETIC VISUALIZATION
# =========================

df_q[q_col] = (
    df_q[q_col]
    .rolling(
        window=5,
        center=True,
        min_periods=1,
    )
    .mean()
)

# =========================
# HYDROEVENTS PREPROCESSING
# =========================

artificial_issues = [
    ("short zero-flow", "2003-05-15 18:00", "2003-05-16 12:00"),
    ("short gap", "2003-05-05 12:00", "2003-05-05 15:00"),
    ("intermediate gap", "2003-03-04 08:00", "2003-03-06 08:00"),
    ("negative values", "2002-09-11 03:00", "2002-09-11 14:00"),
]

df_gap = df_q.copy()

for label, start, end in artificial_issues:

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    mask = (
        (df_gap[date_col] >= start)
        & (df_gap[date_col] <= end)
    )

    if label == "negative values":
        df_gap.loc[mask, q_col] = -np.random.uniform(
            0.01,
            0.10,
            mask.sum(),
        )

    elif label == "short zero-flow":
        df_gap.loc[mask, q_col] = 0

    else:
        df_gap.loc[mask, q_col] = np.nan 

# =========================
# 1. PREPROCESSING ON GAPPED SERIES
# =========================

df_processed, df_daily, summary = preprocess_streamflow(
    df_gap,
    date_col=date_col,
    q_col=q_col,
    apply_gap_filling=True,
)

print("\n--- PREPROCESSING ---")
print(summary)

# =========================
# 2. MRC ON DAILY PROCESSED SERIES
# =========================

mrc_results = compute_mrc(df_daily)

# =========================
# 3. BASEFLOW ON FULL PROCESSED SERIES
# =========================

bf_results = separate_baseflow(
    df_processed,
    q_col=q_col,
    date_col=date_col,
    mrc_results=mrc_results,
    k_method="mrc",
    nan_to_zero=True,
    direction="f",
)

df_runoff = bf_results["df"].copy()

# important: same logic as demo_workflow
if "Date" in df_runoff.columns:
    df_runoff["Date"] = pd.to_datetime(df_runoff["Date"])
    df_runoff = df_runoff.set_index("Date")

df_runoff.index = pd.to_datetime(df_runoff.index)
df_runoff = df_runoff.sort_index()

print("\n--- BASEFLOW ---")
print(f"BFI: {bf_results['bfi']:.3f}")
print(df_runoff[["Q_input", "Baseflow", "Stormflow"]].describe())

# =========================
# 4. EVENTS ON FULL RUNOFF SERIES
# =========================

events_results = rainfall_runoff_event(
    df_runoff=df_runoff,
    df_rain=df_rain,
    basin_name="synthetic_basin",
    area_km2=120,
    min_prominence=0.001,
    max_duration_h=360,
    max_lag_h=120,
)

clean_events_df = events_results["clean_events"]

print("\n--- EVENTS ---")
print(f"Runoff events: {len(events_results['runoff_events'])}")
print(f"Rain events: {len(events_results['rain_events'])}")
print(f"Associated events: {len(events_results['associated_events'])}")
print(f"Clean events: {len(clean_events_df)}")
print(f"Final outliers: {len(events_results['final_outliers'])}")

# =========================
# PREPARE DATA FOR PLOTTING
# =========================

plot_start_dt = pd.to_datetime(plot_start_tot)
plot_end_dt = pd.to_datetime(plot_end_tot)

# preprocessing plots
df_gap_plot = (
    df_gap
    .set_index(date_col)
    .sort_index()
)

if date_col in df_processed.columns:
    df_processed_plot = (
        df_processed
        .set_index(date_col)
        .sort_index()
    )
else:
    df_processed_plot = df_processed.sort_index()

# runoff plots
df_runoff_plot = (
    df_runoff
    .loc[plot_start_dt:plot_end_dt]
    .copy()
)

# rainfall plots
df_rain_plot = (
    df_rain
    .loc[plot_start_dt:plot_end_dt]
    .copy()
)

# events plots
events_plot = clean_events_df[
    (pd.to_datetime(clean_events_df["Rain_Start"]) >= plot_start_dt)
    & (pd.to_datetime(clean_events_df["Rain_Start"]) <= plot_end_dt)
].copy()

report_cols = [
    "Rain_Start",
    "Rain_End",
    "Runoff_Start",
    "Runoff_End",
    "Rain_Peak",
    "First_Runoff_Peak",
    "Runoff_Peaks",
    "Rain_Volume_mm",
    "Runoff_Volume_mm",
    "RC",
]

available_cols = [c for c in report_cols if c in events_plot.columns]

print("\n" + "=" * 120)
print("EVENT REPORT")
print("=" * 120)

with pd.option_context(
    "display.max_columns", None,
    "display.max_colwidth", 200,
    "display.width", 200,
):
    print(events_plot[available_cols])

print("=" * 120)

for i, row in events_plot.iterrows():

    print("\n" + "-" * 80)
    print(f"EVENT {i}")

    print(f"Rain start      : {row['Rain_Start']}")
    print(f"Rain end        : {row['Rain_End']}")
    print(f"Runoff start    : {row['Runoff_Start']}")
    print(f"Runoff end      : {row['Runoff_End']}")

    print(f"Rain peak       : {row['Rain_Peak']}")
    print(f"First peak      : {row['First_Runoff_Peak']}")

    if "Runoff_Peaks" in row:
        print(f"All peaks       : {row['Runoff_Peaks']}")

    print(f"Rain volume [mm]: {row['Rain_Volume_mm']:.2f}")
    print(f"Runoff vol [mm] : {row['Runoff_Volume_mm']:.2f}")

    if "RC" in row:
        print(f"RC              : {row['RC']:.3f}")


# =========================
# EVENT REPORT
# =========================

report_file = out_dir / "event_report.xlsx"

events_report = events_plot.copy()

# converto eventuali liste di picchi in stringhe leggibili
if "Runoff_Peaks" in events_report.columns:
    events_report["Runoff_Peaks"] = events_report["Runoff_Peaks"].apply(
        lambda x: "; ".join(pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"))
        if isinstance(x, (list, tuple)) and len(x) > 0
        else ""
    )

events_report.to_excel(report_file)

print(f"Event report saved: {report_file}")

# =========================
# FIGURE 1 — PREPROCESSING
# =========================

fig, axes = plt.subplots(
    2,
    len(artificial_issues),
    figsize=(14, 5),
    sharey=True,
)

for i, (label, start, end) in enumerate(artificial_issues):

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    plot_start = start - pd.Timedelta(days=2)
    plot_end = end + pd.Timedelta(days=2)

    gap_sub = df_gap_plot.loc[plot_start:plot_end]
    processed_sub = df_processed_plot.loc[plot_start:plot_end]

    # =========================
    # TOP ROW — CORRUPTED
    # =========================

    ax_top = axes[0, i]

    ax_top.plot(
        gap_sub.index,
        gap_sub[q_col],
        color="black",
        linewidth=1.2,
        label="Corrupted series",
    )

    ax_top.axvspan(start, end, alpha=0.18)

    ax_top.set_title(label)

    ax_top.grid(alpha=0.3)

    # =========================
    # BOTTOM ROW — PREPROCESSED
    # =========================

    ax_bottom = axes[1, i]

    ax_bottom.plot(
        processed_sub.index,
        processed_sub[q_col],
        color="red",
        linewidth=1.2,
        label="Preprocessed series",
    )

    ax_bottom.axvspan(start, end, alpha=0.18)

    ax_bottom.grid(alpha=0.3)

# labels
axes[0, 0].set_ylabel("Corrupted\nQ [m³/s]")
axes[1, 0].set_ylabel("Preprocessed\nQ [m³/s]")

for ax in axes[-1, :]:
    ax.set_xlabel("Date")

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig(
    out_dir / "figure_01_preprocessing.png",
    dpi=300,
    bbox_inches="tight",
)

#plt.show()
plt.close()

# =========================
# FIGURE 2 — MASTER RECESSION CURVE
# =========================

fit_results = mrc_results["fit_results"]
df_mrc = mrc_results["df_mrc"]

if fit_results is not None and not df_mrc.empty:

    alpha = fit_results["alpha"]
    q0 = fit_results["Q0"]
    r2 = fit_results["R2"]

    t_fit = np.linspace(
        df_mrc["t_global"].min(),
        df_mrc["t_global"].max(),
        200,
    )

    q_fit = q0 * np.exp(-alpha * t_fit)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.scatter(
        df_mrc["t_global"],
        df_mrc["Q"],
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
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
        ),
    )

    ax.set_xlabel("Shifted time [days]")
    ax.set_ylabel("Q [m³/s]")
    ax.set_title("Master Recession Curve reconstruction")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        out_dir / "figure_02_mrc.png",
        dpi=300,
        bbox_inches="tight",
    )

    #plt.show()
    plt.close()

# =========================
# FIGURE 3 — BASEFLOW SEPARATION
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

plt.savefig(
    out_dir / "figure_03_baseflow.png",
    dpi=300,
    bbox_inches="tight",
)

#plt.show()
plt.close()
"""
# =========================
# FIGURE 4 — EVENT IDENTIFICATION
# =========================

fig, ax_q = plt.subplots(figsize=(10, 4))

ax_q.plot(
    df_runoff_plot.index,
    df_runoff_plot["Stormflow"],
    color="black",
    linewidth=1.1,
    label="Stormflow",
)

ax_q.set_ylabel("Stormflow [mm]")
ax_q.set_xlabel("Date")

ax_p = ax_q.twinx()

ax_p.bar(
    df_rain_plot.index,
    df_rain_plot[p_col],
    width=0.03,
    alpha=0.35,
    label="Rainfall",
)

ax_p.set_ylabel("Rainfall [mm/h]")
ax_p.invert_yaxis()

for i, row in events_plot.iterrows():

    ax_q.axvspan(
        row["Runoff_Start"],
        row["Runoff_End"],
        alpha=0.15,
        label="Runoff event" if i == events_plot.index[0] else None,
    )

    ax_q.scatter(
        row["First_Runoff_Peak"],
        df_runoff_plot.loc[row["First_Runoff_Peak"], "Stormflow"],
        s=15,
        color="salmon",
        zorder=5,
        label="Runoff peak" if i == events_plot.index[0] else None,
    )

    ax_p.scatter(
        row["Rain_Peak"],
        df_rain_plot.loc[row["Rain_Peak"], p_col],
        s=20,
        color="tab:blue",
        marker="v",
        zorder=5,
        label="Rainfall peak" if i == events_plot.index[0] else None,
    )

ax_q.set_title("Rainfall–runoff event identification")
ax_q.grid(alpha=0.3)

lines_q, labels_q = ax_q.get_legend_handles_labels()
lines_p, labels_p = ax_p.get_legend_handles_labels()

ax_q.legend(
    lines_q + lines_p,
    labels_q + labels_p,
    frameon=False,
    loc="upper right",
)

plt.tight_layout()

plt.savefig(
    out_dir / "figure_04_events.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()"""
# =========================
# FIGURE 4 — EVENTS AND EVENT-SCALE METRICS
# =========================

rmse_threshold = 0.05

events_metrics = clean_events_df.copy()

events_metrics = events_metrics[
    events_metrics["RMSE_lr"] <= rmse_threshold
].copy()

fig = plt.figure(figsize=(10, 7))

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[2, 1],
)

# =========================
# TOP PANEL
# =========================

ax_q = fig.add_subplot(gs[0, :])

ax_q.plot(
    df_runoff_plot.index,
    df_runoff_plot["Stormflow"],
    color="black",
    linewidth=1.1,
    label="Stormflow",
)

ax_q.set_ylabel("Stormflow [mm]")

ax_p = ax_q.twinx()

ax_p.bar(
    df_rain_plot.index,
    df_rain_plot[p_col],
    width=0.03,
    alpha=0.35,
    label="Rainfall",
)

ax_p.set_ylabel("Rainfall [mm/h]")
ax_p.invert_yaxis()

for i, row in events_plot.iterrows():

    ax_q.axvspan(
        row["Runoff_Start"],
        row["Runoff_End"],
        alpha=0.15,
        label="Runoff event" if i == events_plot.index[0] else None,
    )

    ax_q.scatter(
        row["First_Runoff_Peak"],
        df_runoff_plot.loc[row["First_Runoff_Peak"], "Stormflow"],
        s=35,
        color="tab:red",
        zorder=5,
        label="Runoff peak" if i == events_plot.index[0] else None,
    )

    ax_p.scatter(
        row["Rain_Peak"],
        df_rain_plot.loc[row["Rain_Peak"], p_col],
        s=35,
        color="tab:blue",
        marker="v",
        zorder=5,
        label="Rainfall peak" if i == events_plot.index[0] else None,
    )

ax_q.set_title("(a) Rainfall–runoff event identification")
ax_q.grid(alpha=0.3)

lines_q, labels_q = ax_q.get_legend_handles_labels()
lines_p, labels_p = ax_p.get_legend_handles_labels()

ax_q.legend(
    lines_q + lines_p,
    labels_q + labels_p,
    frameon=False,
    loc="upper right",
)

# =========================
# BOTTOM LEFT
# =========================

ax1 = fig.add_subplot(gs[1, 0])

ax1.scatter(
    events_metrics["Rain_Volume_mm"],
    events_metrics["Runoff_Volume_mm"],
    s=25,
    color="black",
    alpha=0.7,
)

ax1.set_xlabel("Rainfall volume [mm]")
ax1.set_ylabel("Runoff volume [mm]")
ax1.set_title("(b) P vs R")
ax1.grid(alpha=0.3)

# automatic limits
xmax1 = events_metrics["Rain_Volume_mm"].max() * 1.05
ymax1 = events_metrics["Runoff_Volume_mm"].max() * 1.05

ax1.set_xlim(0, xmax1)
ax1.set_ylim(0, ymax1)

# =========================
# BOTTOM RIGHT
# =========================

ax2 = fig.add_subplot(gs[1, 1])

ax2.scatter(
    events_metrics["Rain_Volume_mm"],
    events_metrics["RC_lr"],
    s=25,
    color="black",
    alpha=0.7,
)

ax2.set_xlabel("Rainfall volume [mm]")
ax2.set_ylabel("Simulated RC")
ax2.set_title("(c) P vs RCsim")
ax2.grid(alpha=0.3)

# automatic limits
xmax2 = events_metrics["Rain_Volume_mm"].max() * 1.05
ymax2 = events_metrics["RC_lr"].max() * 1.05

ax2.set_xlim(0, xmax2)
ax2.set_ylim(0, ymax2)

# =========================
# FINALIZE
# =========================

fig.suptitle(
    "Event identification",
    y=0.98,
)

plt.tight_layout()

plt.savefig(
    out_dir / "figure_04_events_metrics.png",
    dpi=300,
    bbox_inches="tight",
)

#plt.show()
plt.close()

# =========================
# FIGURE HTML — EVENT IDENTIFICATION
# =========================

df_runoff_plot = df_runoff.copy()
df_rain_plot = df_rain.copy()
events_plot = clean_events_df.copy()

fig = go.Figure()

# Runoff line
fig.add_trace(
    go.Scattergl(
        x=df_runoff_plot.index,
        y=df_runoff_plot["Stormflow"],
        mode="lines",
        name="Stormflow",
        line=dict(color="black", width=1),
    )
)

# Rainfall inverted axis
fig.add_trace(
    go.Bar(
        x=df_rain_plot.index,
        y=df_rain_plot[p_col],
        name="Rainfall",
        yaxis="y2",
        marker=dict(
            color="cornflowerblue",
            line=dict(
                color="cornflowerblue",
                width=2
            )
        ),
        opacity=0.85,
    )
)

# =========================
# Events
# =========================

main_peak_x = []
main_peak_y = []

for i, row in events_plot.iterrows():

    ev_start = pd.to_datetime(row["Runoff_Start"])
    ev_end = pd.to_datetime(row["Runoff_End"])

    # ----------------------
    # Main peak
    # ----------------------

    if len(row["Runoff_Peaks"]) > 0:
    
        peak_time = max(
            row["Runoff_Peaks"],
            key=lambda p: df_runoff_plot.loc[p, "Stormflow"]
            if p in df_runoff_plot.index else -np.inf
        )
    
        main_peak_x.append(peak_time)
        main_peak_y.append(df_runoff_plot.loc[peak_time, "Stormflow"])

    # ----------------------
    # Event runoff
    # ----------------------

    event_runoff = df_runoff_plot.loc[
        ev_start:ev_end,
        "Stormflow"
    ]

    # evento totale
    fig.add_vrect(
        x0=ev_start,
        x1=ev_end,
        fillcolor="rgba(118,130,100,0.3)",
        opacity=0.18,
        line_width=0,
        layer="below",
    )

    # runoff volume
    fig.add_trace(
        go.Scattergl(
            x=event_runoff.index,
            y=event_runoff.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(123,13,17,0.3)",
            line=dict(
                color="rgba(255,140,0,0)"
            ),
            name="Event runoff"
            if i == events_plot.index[0]
            else None,
            showlegend=i == events_plot.index[0],
        )
    )

# =========================
# Main peaks (single trace)
# =========================

fig.add_trace(
    go.Scattergl(
        x=main_peak_x,
        y=main_peak_y,
        mode="markers",
        name="Main runoff peak",
        marker=dict(
            symbol="diamond",
            size=5,
            color="firebrick",
            line=dict(
                color="black",
                width=0.8
            ),
        ),
    )
)

fig.update_layout(
    #title="Rainfall–runoff event identification",
    template="plotly_white",
    width=1100,
    height=500,

    xaxis=dict(
        title="Date",
        rangeslider=dict(visible=True),
        type="date",
    ),

    yaxis=dict(
        title="Stormflow [mm/h]",
        rangemode="tozero",
    ),

    yaxis2=dict(
        title="Rainfall [mm/h]",
        overlaying="y",
        side="right",
        autorange="reversed",
    ),
    
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    )
)

fig.write_html(
    out_dir / "figure_04_event_identification.html",
    include_plotlyjs="cdn",
)

fig.show()