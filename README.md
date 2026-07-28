[![DOI](https://zenodo.org/badge/1294907041.svg)](https://doi.org/10.5281/zenodo.21650373)

# HydroEvents

HydroEvents is an open-source Python package for rainfall–runoff event extraction and event-based hydrological analysis. The workflow combines streamflow preprocessing, Master Recession Curve (MRC) analysis, baseflow separation, rainfall–runoff event identification, and event-scale runoff modelling within a single reproducible framework.

`hydroevents` provides a modular workflow for:

- streamflow preprocessing
- Master Recession Curve (MRC) estimation
- baseflow separation
- rainfall–runoff event identification and association

The package is designed for event-based hydrological analysis using hourly precipitation and streamflow data.

---

## Main features

- Streamflow preprocessing and gap filling
- Detection and correction of short discharge artefacts
- Master Recession Curve (MRC) estimation
- Recession parameter estimation from MRC
- Baseflow separation using the Lyne–Hollick filter
- Rainfall event identification
- Runoff event identification
- Automatic rainfall–runoff event association under physical constraints
- Event filtering and quality control
- Event-scale linear reservoir calibration

---

## Workflow

HydroEvents follows the workflow:

Streamflow preprocessing
→ Master Recession Curve (MRC)
→ Baseflow separation
→ Rainfall event identification
→ Runoff event identification
→ Rainfall–runoff association
→ Event filtering
→ Event-scale linear reservoir calibration

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/hydroevents.git
cd hydroevents
pip install -e .
```

### For Developers

If you plan to contribute, run the notebooks, or want a fully reproducible environment, set up the dedicated conda environment instead:

```bash
git clone https://github.com/yourusername/hydroevents.git
cd hydroevents
conda env create -f environment.yml
conda activate hydroevents
pip install -e .
```

Alternatively, use the provided `setup_conda_env.sh` script, which creates (or updates) the environment and installs the package for you:

```bash
./setup_conda_env.sh
```

If you don't have a global conda/Miniconda installation, pass a target directory and the script will bootstrap a private Miniconda distribution there before building the environment:

```bash
./setup_conda_env.sh -p ./conda
```

The script also generates an `activate_env.sh` helper in the repo root. Activate the environment with:

```bash
source activate_env.sh
```

---

## Requirements

Core dependencies:

- Python >= 3.10
- numpy
- pandas
- scipy
- scikit-learn

Optional dependencies used in notebooks and examples:

- matplotlib
- openpyxl

A full reproducible environment is available in:

```text
environment.yml
```

---

## Minimal example

```python
from hydroevents import (
    preprocess_streamflow,
    compute_mrc,
    separate_baseflow,
    rainfall_runoff_event,
)

# preprocess streamflow
df_processed, df_daily, summary = preprocess_streamflow(
    df_q,
    date_col="Date",
    q_col="Q",
)

# compute Master Recession Curve
mrc_results = compute_mrc(df_daily)

# baseflow separation
baseflow_results = separate_baseflow(
    df_processed,
    q_col="Q",
    date_col="Date",
    mrc_results=mrc_results,
    k_method="mrc",
)

# rainfall–runoff event analysis

df_runoff = baseflow_results["df"]

event_results = rainfall_runoff_event(
    df_runoff=baseflow_results["df"],
    df_rain=df_p,
    area_km2=150,
)
```

---

## Example workflow

Additional examples and demonstration datasets are available in:

```text
examples/
```

Additional step-by-step tutorials, parameter descriptions, and workflow demonstrations are available in:

```text
notebooks/
├── preprocessing.ipynb
├── mrc.ipynb
├── baseflow.ipynb
└── events.ipynb
```

---

## Package structure

```text
hydroevents/
├── README.md
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── environment.yml
├── setup_conda_env.sh
├── .gitignore
├── src/
│   └── hydroevents/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── mrc.py
│       ├── baseflow.py
│       └── events.py
├── examples/
│   ├── demo_workflow.py
│   └── data/
└── notebooks/
```

---

## Citation

If you use HydroEvents in scientific work, please cite the associated software publication and/or repository via the Zenodo DOI in the badge at the beginning of this README, or from the DOI url:

```text
10.5281/zenodo.21650374
```

GitHub automatically provides citation formats through the `CITATION.cff` file.

---

## License

This project is licensed under the BSD-3-Clause License as reported in the LICENCE file.

---

## Author(s)

- Sofia Ortenzi (sofia.ortenzi@cnr.it)
- Lucio Di Matteo (lucio.dimatteo@unipg.it)
- Martina Natali (martinanatali@cnr.it)
- Christian Massari (christian.massari@cnr.it)

