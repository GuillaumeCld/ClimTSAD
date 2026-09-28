# ClimTSAD

Climate time-series anomaly detection experiments and a reproducible case study.

The reproducible examples use the common period 1970--2023. The 1970--1999
observations provide a 30-year training climatology, and 2000--2023 is held out
for scoring and comparative event analysis.

The HTML export workflow is available through
`python -m anomaly_detection.plots`.

## Project layout

```text
src/anomaly_detection/  Reusable detectors, training code, and models
case_study/             Peak analysis, detector comparison, deep presets, and plots
data/               Input NumPy time series
outputs/            Generated scores, summaries, and figures
docs/               Experimental protocol source and PDF
scripts/            Data-extraction utilities
pyproject.toml      Package and installed command definitions
environment.yml     Reproducible CPU-first conda environment
```

Create and activate the complete environment from the project root with:

```bash
conda env create -f environment.yml
conda activate climate-anomaly-case-study
```

The environment installs this repository in editable mode. To use an existing
environment instead, run `python -m pip install -e . --no-deps`.

The IMP detector additionally requires the external `libimp` extension. Build
it separately under `../MP_CPP/build/src/` (or otherwise expose `libimp` on
`PYTHONPATH`). The environment includes its C++/CMake build toolchain but not
the external source code.

## Reproducible batch analysis

Classical detectors and event declustering use a fixed 10-day subsequence for
every series. Deep reconstruction models use the selected 96-step input preset.
Run one series with:

```bash
case-study --series temperature --seed 0
```

Without installation, the equivalent command is
`python case_study/example.py --series temperature --seed 0`.

Run all three series and create the cross-series summary with:

```bash
case-study --seed 0
```

Outputs are stored under `outputs/figures_evt/`.
Each series also writes `deep_losses.csv` with the final train, validation, and
test reconstruction MSE for every deep-learning model.
The overlapping reconstruction time series are saved to
`deep_reconstructions.csv`; their median, mean, and maximum absolute errors in
physical units are saved to `deep_reconstruction_summary.csv`.

The batch workflow is split by responsibility:

- `case_study/example.py` orchestrates the experiment.
- `case_study/analysis.py` prepares data and computes agreement summaries.
- `case_study/pot.py` performs percentile peak extraction, declustering,
  seasonal diagnostics, event ranking, and unlabelled detector agreement.
- `case_study/deep.py` owns deep-model hyperparameters, training, and reconstruction.
- `case_study/plotting.py` creates and saves all case-study figures.
