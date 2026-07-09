# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:18:04 2026

@author: sofia
"""

from .preprocessing import (
    fill_streamflow_gaps,
    to_daily,
    preprocess_streamflow,
)

from .mrc import (
    compute_mrc,
    get_lh_k_from_mrc,
    build_shifted_segments_dataframe,
)

from .baseflow import (
    lyne_hollick_filter,
    compute_bfi,
    separate_baseflow,
)

from .events import (
    discharge_to_mm,
    estimate_first_peak_frac_min,
    identify_events,
    identify_rain_events,
    associate_events,
    final_clean_events,
    simulate_linear_reservoir,
    rainfall_runoff_event,
)

__all__ = [
    # preprocessing
    "fill_streamflow_gaps",
    "to_daily",
    "preprocess_streamflow",

    # mrc
    "compute_mrc",
    "get_lh_k_from_mrc",
    "build_shifted_segments_dataframe",

    # baseflow
    "lyne_hollick_filter",
    "compute_bfi",
    "separate_baseflow",

    # events
    "discharge_to_mm",
    "estimate_first_peak_frac_min",
    "identify_events",
    "identify_rain_events",
    "associate_events",
    "final_clean_events",
    "simulate_linear_reservoir",
    "rainfall_runoff_event",
]