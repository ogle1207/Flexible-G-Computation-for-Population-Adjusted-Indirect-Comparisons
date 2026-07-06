# Supplementary Code

This archive contains the Python code used to generate the simulation outputs
reported in the manuscript and Supplementary Materials.

## Files

- `Supplementary_Code_1_main_simulation.py`: primary simulations, nonzero-effect
  simulations, estimator implementations, diagnostics, summary tables, and
  figures.
- `Supplementary_Code_2_additional_sensitivity.py`: low-event-rate,
  reverse-overlap, covariance-reconstruction, and additional bootstrap
  sensitivity analyses.
- `run_demo.py`: reduced workflow check. The demonstration creates expected
  output templates and records the intended run order; it does not attempt to
  rerun the full simulation.
- `requirements.txt`: Python package requirements.
- `package_versions.txt`: software versions recorded when this code archive
  was prepared.
- `outputs_expected/`: expected output-file templates.

## Run Order

1. Optional workflow check: run `python run_demo.py`.
2. Full primary and nonzero-effect simulation: run
   `python Supplementary_Code_1_main_simulation.py`.
3. Additional sensitivity analyses: run
   `python Supplementary_Code_2_additional_sensitivity.py`.

The scripts write outputs to `Data/`, `Results/`, and
`Final_Results_Analysis_ITC/`. The full simulation is computationally intensive;
rerunning the scripts will resume incomplete scenario-method combinations where
intermediate result files already exist. Because bootstrap resampling and
parallel execution include additional stochastic steps, reruns are expected to
reproduce qualitative results up to Monte Carlo variation rather than
bitwise-identical CSV files.

In the manuscript, `G-comp SL, SA` denotes the split-averaged Super Learner
G-computation sensitivity implementation. In the code, this implementation is
defined by `gcomp_sl_split_averaged_wrapper` and writes to
`Results/GCOMP_SL_SA`. It is distinct from `TMLE (SL+CF)`, where `CF` denotes
cross-fitted Super Learner TMLE.

## Environment

The code was prepared for Python 3 with numpy, pandas, scipy, scikit-learn,
statsmodels, xgboost, pygam, matplotlib, seaborn, joblib, and tqdm. The main
script writes `Final_Results_Analysis_ITC/package_versions.txt` when rerun.

## Anonymization

The archive is provided for double-anonymized review. It contains no author
names, institutional paths, e-mail addresses, or repository links. Local output
paths are relative to the working directory.
