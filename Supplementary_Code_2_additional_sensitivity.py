#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import time
import pickle
import warnings

# ================= start =================
# Store joblib temporary files in the project directory.
# This must be set before importing joblib and scikit-learn.
CUSTOM_TEMP_DIR = os.path.join(os.getcwd(), "joblib_temp")
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = CUSTOM_TEMP_DIR
# End joblib temporary-file configuration.

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Joblib reads the environment variable set above.
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import Supplementary_Code_1_main_simulation as sim



warnings.filterwarnings("ignore")

# =============================================================================
# Addon settings
# =============================================================================

# Sensitivity analyses can be run with fewer replicates for exploratory checks.
# Use 1000 replicates for event-rate and reverse-overlap analyses when feasible.
N_ADDON = 1000

# Covariance sensitivity only runs G-comp (SL) with independent covariance.
# Since it is only one method, 1000 is preferable.
N_COV_REPS = 1000

# Bootstrap=1000 sensitivity: keep consistent with your existing code.
N_BOOT1K_ADDON = 500

RESAMPLES_ADDON = 100
RESAMPLES_CF_ADDON = 100
N_STAR_ADDON = 2000

OUT_DIR = "Final_Results_Analysis_ITC"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Base learners: same library as the primary simulation script
# =============================================================================

def make_base_learners():
    return [
        ("lr_l2", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000
        )),
        ("lr_l1", LogisticRegression(
            penalty="l1", C=1.0, solver="liblinear", max_iter=1000
        )),
        ("rf_shallow", RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            max_features="sqrt",
            min_samples_leaf=10,
            random_state=444,
            n_jobs=1,
        )),
        ("xgb_shallow", sim.SafeXGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
        )),
        ("rf_deep", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=444,
            n_jobs=1,
        )),
        ("xgb_deep", sim.SafeXGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
        )),
        ("gam", sim.LogisticGAMClassifier()),
    ]


try:
    N_CPU = os.cpu_count() or 4
except Exception:
    N_CPU = 4

N_JOBS = max(1, N_CPU - 2)


# =============================================================================
# Reviewer-driven addon scenarios
# =============================================================================

LOW_EVENT_RATE = 0.15

LOW_EVENT_SCENS = [
    {
        "scenario_group": "low_event_rate_0.15",
        "fid": "EVR15_N_AC200meanX_AC0.15_nonlinear",
        "N_AC": 200,
        "meanX_AC": 0.15,
        "type": "nonlinear",
        "target_event_rate": LOW_EVENT_RATE,
        "truth": 0.0,
    },
    {
        "scenario_group": "low_event_rate_0.15",
        "fid": "EVR15_N_AC400meanX_AC0.3_nonlinear",
        "N_AC": 400,
        "meanX_AC": 0.30,
        "type": "nonlinear",
        "target_event_rate": LOW_EVENT_RATE,
        "truth": 0.0,
    },
    {
        "scenario_group": "low_event_rate_0.15",
        "fid": "EVR15_N_AC600meanX_AC0.45_linear",
        "N_AC": 600,
        "meanX_AC": 0.45,
        "type": "linear",
        "target_event_rate": LOW_EVENT_RATE,
        "truth": 0.0,
    },
]

REVERSE_OVERLAP_SCENS = [
    {
        "scenario_group": "reverse_overlap",
        "fid": "REVOV_N_AC200meanX_AC1.05_nonlinear",
        "N_AC": 200,
        "meanX_AC": 1.05,
        "type": "nonlinear",
        "target_event_rate": 0.35,
        "truth": 0.0,
    },
    {
        "scenario_group": "reverse_overlap",
        "fid": "REVOV_N_AC400meanX_AC0.9_nonlinear",
        "N_AC": 400,
        "meanX_AC": 0.90,
        "type": "nonlinear",
        "target_event_rate": 0.35,
        "truth": 0.0,
    },
    {
        "scenario_group": "reverse_overlap",
        "fid": "REVOV_N_AC200meanX_AC1.05_linear",
        "N_AC": 200,
        "meanX_AC": 1.05,
        "type": "linear",
        "target_event_rate": 0.35,
        "truth": 0.0,
    },
]

ADDON_SCENS = LOW_EVENT_SCENS + REVERSE_OVERLAP_SCENS

BOOT1000_EXTRA_SCENS = [
    {
        "fid": "N_AC400meanX_AC0.3_nonlinear",
        "N_AC": 400,
        "meanX_AC": 0.30,
        "type": "nonlinear",
        "truth": 0.0,
    }
]

COVARIANCE_EXTRA_SCENS = [
    {
        "fid": "N_AC200meanX_AC0.15_linear",
        "N_AC": 200,
        "meanX_AC": 0.15,
        "type": "linear",
        "truth": 0.0,
    },
    {
        "fid": "N_AC200meanX_AC0.15_nonlinear",
        "N_AC": 200,
        "meanX_AC": 0.15,
        "type": "nonlinear",
        "truth": 0.0,
    },
    # These two should already exist from your original code, but including
    # them here allows the final comparison table to include all covariance
    # sensitivity scenarios together.
    {
        "fid": "N_AC400meanX_AC0.3_linear",
        "N_AC": 400,
        "meanX_AC": 0.30,
        "type": "linear",
        "truth": 0.0,
    },
    {
        "fid": "N_AC400meanX_AC0.3_nonlinear",
        "N_AC": 400,
        "meanX_AC": 0.30,
        "type": "nonlinear",
        "truth": 0.0,
    },
]


# =============================================================================
# Utilities
# =============================================================================

def load_main_b0():
    with open("binary_settings.pkl", "rb") as f:
        settings = pickle.load(f)
    return settings["b_0_dict"]


def find_b0_for_event_rate(target_type, target_rate):
    """
    Calibrate b0 so that the BC target population has approximately the
    desired marginal event rate.
    """

    def obj(b0):
        # Fixed seed inside objective makes optimization less noisy.
        np.random.seed(sim.scenario_seed(f"ADDON_B0_{target_type}_{target_rate}"))
        df = sim.gen_data_internal(
            120000,
            sim.b_trt,
            sim.b_X,
            sim.b_EM,
            b0,
            sim.meanX_BC,
            sim.sdX,
            sim.corX,
            sim.allocation,
            target_type,
        )
        return (df["y"].mean() - target_rate) ** 2

    return minimize_scalar(obj, bounds=(-10, 10), method="bounded").x


def load_or_compute_low_event_b0():
    path = "addon_b0_low_event_rate.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    out = {}
    for st in sim.SCEN_TYPES:
        print(f"[b0] Calibrating low-event b0 for type={st}, event_rate={LOW_EVENT_RATE}")
        out[st] = find_b0_for_event_rate(st, LOW_EVENT_RATE)
        print(f"     b0={out[st]:.6f}")

    with open(path, "wb") as f:
        pickle.dump(out, f)

    return out


def dataset_len(ipd_path):
    if not os.path.exists(ipd_path):
        return 0
    try:
        with open(ipd_path, "rb") as f:
            return len(pickle.load(f))
    except Exception:
        return 0


def ensure_dataset(fid, n_ac, meanX_ac, scen_type, b0_st, n_sim):
    """
    Generate addon data if missing or too short.
    For these addon scenarios, true A vs B ITC is still 0 because
    AC and BC are generated using the same b_trt.
    """
    os.makedirs("Data", exist_ok=True)

    ipd_path = f"Data/IPD_AC_{fid}.pkl"
    ald_path = f"Data/ALD_BC_{fid}.pkl"

    if (
        os.path.exists(ipd_path)
        and os.path.exists(ald_path)
        and dataset_len(ipd_path) >= n_sim
    ):
        print(f"[data exists] {fid}")
        return

    print(f"[generate data] {fid} | n_sim={n_sim}")

    np.random.seed(sim.scenario_seed("ADDON_DATA_" + fid))

    ipd_ac = [
        sim.gen_data_internal(
            n_ac,
            sim.b_trt,
            sim.b_X,
            sim.b_EM,
            b0_st,
            meanX_ac,
            sim.sdX,
            sim.corX,
            sim.allocation,
            scen_type,
        )
        for _ in range(n_sim)
    ]

    ipd_bc = [
        sim.gen_data_internal(
            sim.N_BC,
            sim.b_trt,
            sim.b_X,
            sim.b_EM,
            b0_st,
            sim.meanX_BC,
            sim.sdX,
            sim.corX,
            sim.allocation,
            scen_type,
        )
        for _ in range(n_sim)
    ]

    ald_bc = sim.build_ald(ipd_bc)

    with open(ipd_path, "wb") as f:
        pickle.dump(ipd_ac, f)

    with open(ald_path, "wb") as f:
        pickle.dump(ald_bc, f)


def load_method_arrays(folder, fid):
    bp = f"Results/{folder.upper()}"

    paths = {
        "means": f"{bp}/means_{fid}.pkl",
        "vars": f"{bp}/variances_{fid}.pkl",
        "lcis": f"{bp}/lcis_{fid}.pkl",
        "ucis": f"{bp}/ucis_{fid}.pkl",
        "times": f"{bp}/times_{fid}.pkl",
    }

    if not all(os.path.exists(p) for p in paths.values()):
        return [], [], [], [], []

    try:
        arrays = {}
        for k, p in paths.items():
            with open(p, "rb") as f:
                arrays[k] = list(pickle.load(f))

        n = min(len(v) for v in arrays.values())
        return (
            arrays["means"][:n],
            arrays["vars"][:n],
            arrays["lcis"][:n],
            arrays["ucis"][:n],
            arrays["times"][:n],
        )
    except Exception:
        return [], [], [], [], []


def save_method_arrays(folder, fid, means, vars_, lcis, ucis, times):
    bp = f"Results/{folder.upper()}"
    os.makedirs(bp, exist_ok=True)

    with open(f"{bp}/means_{fid}.pkl", "wb") as f:
        pickle.dump(means, f)
    with open(f"{bp}/variances_{fid}.pkl", "wb") as f:
        pickle.dump(vars_, f)
    with open(f"{bp}/lcis_{fid}.pkl", "wb") as f:
        pickle.dump(lcis, f)
    with open(f"{bp}/ucis_{fid}.pkl", "wb") as f:
        pickle.dump(ucis, f)
    with open(f"{bp}/times_{fid}.pkl", "wb") as f:
        pickle.dump(times, f)


# =============================================================================
# 1. Run low-event and reverse-overlap scenarios
# =============================================================================

def run_addon_scenarios(base_learners):
    print("\n" + "=" * 70)
    print("ADDON 1 - Low-event-rate and reverse-overlap scenarios")
    print("=" * 70)

    b0_main = load_main_b0()
    b0_low = load_or_compute_low_event_b0()

    for sc in LOW_EVENT_SCENS:
        ensure_dataset(
            fid=sc["fid"],
            n_ac=sc["N_AC"],
            meanX_ac=sc["meanX_AC"],
            scen_type=sc["type"],
            b0_st=b0_low[sc["type"]],
            n_sim=N_ADDON,
        )

    for sc in REVERSE_OVERLAP_SCENS:
        ensure_dataset(
            fid=sc["fid"],
            n_ac=sc["N_AC"],
            meanX_ac=sc["meanX_AC"],
            scen_type=sc["type"],
            b0_st=b0_main[sc["type"]],
            n_sim=N_ADDON,
        )

    for sc in ADDON_SCENS:
        fid = sc["fid"]
        ipd, ald = sim.load_scenario_data(fid)

        if ipd is None:
            print(f"[missing data] {fid}")
            continue

        reps = min(N_ADDON, len(ipd))

        print(f"\n[run addon scenario] {fid} | reps={reps}")

        sim.run_scenario(
            fid=fid,
            ipd=ipd,
            ald_list=ald,
            reps=reps,
            base_learners=base_learners,
            n_jobs=N_JOBS,
            resamples=RESAMPLES_ADDON,
            resamples_cf=RESAMPLES_CF_ADDON,
            n_star=N_STAR_ADDON,
            run_robustness=False,
            save_diag=True,
        )


def collect_addon_scenario_metrics():
    rows = []

    for sc in ADDON_SCENS:
        fid = sc["fid"]
        truth = sc["truth"]

        for method_name, folder in sim.methods_map.items():
            means, vars_, lcis, ucis, times = load_method_arrays(folder, fid)

            if len(means) == 0:
                continue

            mt = sim.process_metrics(
                means=np.array(means),
                variances=np.array(vars_),
                lcis=np.array(lcis),
                ucis=np.array(ucis),
                times=np.array(times),
                truth=truth,
            )

            if not mt:
                continue

            mt.update({
                "fid": fid,
                "Method": method_name,
                "scenario_group": sc["scenario_group"],
                "N_AC": sc["N_AC"],
                "meanX_AC": sc["meanX_AC"],
                "type": sc["type"],
                "target_event_rate": sc["target_event_rate"],
                "Truth": truth,
                "N_reps_target": N_ADDON,
            })

            rows.append(mt)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["scenario_group", "type", "N_AC", "meanX_AC", "Method"])
        out = f"{OUT_DIR}/addon_eventrate_reverseoverlap_metrics.csv"
        df.to_csv(out, index=False)
        print(f"[saved] {out} ({len(df)} rows)")

    return df


# =============================================================================
# 2. Add missing bootstrap=1000 moderate nonlinear scenario
# =============================================================================

def run_boot1000_extra(base_learners):
    print("\n" + "=" * 70)
    print("ADDON 2 - Missing bootstrap=1000 moderate nonlinear scenario")
    print("=" * 70)

    os.makedirs("Results/BOOT1000", exist_ok=True)

    for sc in BOOT1000_EXTRA_SCENS:
        fid = sc["fid"]
        path = f"Results/BOOT1000/boot1000_{fid}.pkl"

        out = []
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    out = list(pickle.load(f))
            except Exception:
                out = []

        ipd, ald = sim.load_scenario_data(fid)

        if ipd is None:
            print(f"[missing data] {fid}")
            continue

        reps = min(N_BOOT1K_ADDON, len(ipd))
        n_done = len(out)

        if n_done >= reps:
            print(f"[skip] boot1000 {fid} already complete: {n_done}/{reps}")
            continue

        print(f"[run boot1000] {fid}: {n_done}/{reps}")

        batch_size = max(20, N_JOBS * 2)

        for bs in tqdm(range(n_done, reps, batch_size), desc=f"boot1000 {fid}"):
            be = min(bs + batch_size, reps)

            chunk = None

            for attempt, nj in enumerate([N_JOBS, max(1, N_JOBS // 2), max(1, N_JOBS // 4), 1]):
                try:
                    with Parallel(
                        n_jobs=nj,
                        backend="loky",
                        verbose=0,
                        max_nbytes=None,
                        batch_size=1,
                        pre_dispatch="2*n_jobs",
                        timeout=14400,
                    ) as parallel:
                        chunk = parallel(
                            delayed(sim.run_one_replicate)(
                                j,
                                ipd,
                                ald,
                                base_learners,
                                1000,
                                200,
                                N_STAR_ADDON,
                                False,
                            )
                            for j in range(bs, be)
                        )
                    break
                except Exception as e:
                    print(
                        f"[boot1000 batch {bs}-{be} attempt={attempt} "
                        f"n_jobs={nj}] {type(e).__name__}: {e}"
                    )
                    gc.collect()
                    time.sleep(5)

                    if nj == 1:
                        chunk = [None] * (be - bs)

            if chunk is None:
                chunk = [None] * (be - bs)

            out.extend(chunk)

            with open(path, "wb") as f:
                pickle.dump(out, f)

            gc.collect()

    df_boot = sim.collect_boot1000()

    if not df_boot.empty:
        out_csv = f"{OUT_DIR}/boot1000_sensitivity_updated.csv"
        df_boot.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv} ({len(df_boot)} rows)")


# =============================================================================
# 3. Add independent-covariance sensitivity for challenging primary scenarios
# =============================================================================

def run_single_method_resume(fid, ipd, ald, reps, folder, wrapper_fn, batch_size=25):
    """
    Run one method only and save under Results/{folder}.
    This avoids touching completed main-grid results.
    """

    means, vars_, lcis, ucis, times = load_method_arrays(folder, fid)
    n_done = len(means)

    if n_done >= reps:
        print(f"[skip] {folder} {fid} complete: {n_done}/{reps}")
        return

    print(f"[run single method] {folder} {fid}: {n_done}/{reps}")

    def worker(j):
        t0 = time.time()
        try:
            est, var, lci, uci = wrapper_fn(ipd[j], ald[j])
        except BaseException:
            est, var, lci, uci = np.nan, np.nan, np.nan, np.nan
        return est, var, lci, uci, time.time() - t0

    for bs in tqdm(range(n_done, reps, batch_size), desc=f"{folder} {fid}"):
        be = min(bs + batch_size, reps)

        chunk = None

        for attempt, nj in enumerate([N_JOBS, max(1, N_JOBS // 2), max(1, N_JOBS // 4), 1]):
            try:
                with Parallel(
                    n_jobs=nj,
                    backend="loky",
                    verbose=0,
                    max_nbytes=None,
                    batch_size=1,
                    pre_dispatch="2*n_jobs",
                    timeout=7200,
                ) as parallel:
                    chunk = parallel(delayed(worker)(j) for j in range(bs, be))
                break
            except Exception as e:
                print(
                    f"[single-method batch {bs}-{be} attempt={attempt} "
                    f"n_jobs={nj}] {type(e).__name__}: {e}"
                )
                gc.collect()
                time.sleep(5)

                if nj == 1:
                    chunk = [(np.nan, np.nan, np.nan, np.nan, np.nan)] * (be - bs)

        if chunk is None:
            chunk = [(np.nan, np.nan, np.nan, np.nan, np.nan)] * (be - bs)

        for est, var, lci, uci, tm in chunk:
            means.append(est)
            vars_.append(var)
            lcis.append(lci)
            ucis.append(uci)
            times.append(tm)

        save_method_arrays(folder, fid, means, vars_, lcis, ucis, times)

        gc.collect()


def run_covariance_extra(base_learners):
    print("\n" + "=" * 70)
    print("ADDON 3 - Independent covariance sensitivity for challenging primary scenarios")
    print("=" * 70)

    for sc in COVARIANCE_EXTRA_SCENS:
        fid = sc["fid"]
        ipd, ald = sim.load_scenario_data(fid)

        if ipd is None:
            print(f"[missing data] {fid}")
            continue

        reps = min(N_COV_REPS, len(ipd))

        def indep_wrapper(data_ac, data_bc):
            return sim.gcomp_sl_wrapper(
                data_ac,
                data_bc,
                RESAMPLES_ADDON,
                N_STAR_ADDON,
                base_learners,
                3,
                "independent",
            )

        run_single_method_resume(
            fid=fid,
            ipd=ipd,
            ald=ald,
            reps=reps,
            folder="GCOMP_SL_INDEP",
            wrapper_fn=indep_wrapper,
            batch_size=25,
        )


def collect_covariance_metrics():
    rows = []

    for sc in COVARIANCE_EXTRA_SCENS:
        fid = sc["fid"]
        truth = sc["truth"]

        for cor_label, folder in [
            ("AC-proxy correlation", "GCOMP_SL"),
            ("Independent covariance", "GCOMP_SL_INDEP"),
        ]:
            means, vars_, lcis, ucis, times = load_method_arrays(folder, fid)

            if len(means) == 0:
                continue

            mt = sim.process_metrics(
                means=np.array(means),
                variances=np.array(vars_),
                lcis=np.array(lcis),
                ucis=np.array(ucis),
                times=np.array(times),
                truth=truth,
            )

            if not mt:
                continue

            mt.update({
                "fid": fid,
                "Method": "G-comp (SL)",
                "correlation_assumption": cor_label,
                "N_AC": sc["N_AC"],
                "meanX_AC": sc["meanX_AC"],
                "type": sc["type"],
                "Truth": truth,
            })

            rows.append(mt)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["type", "N_AC", "meanX_AC", "correlation_assumption"])
        out_csv = f"{OUT_DIR}/covariance_sensitivity_expanded.csv"
        df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv} ({len(df)} rows)")

    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sim.write_package_versions()
    print("\n" + "#" * 70)
    print("# ADDON REVISION SENSITIVITY ANALYSES")
    print(f"# N_ADDON       = {N_ADDON}")
    print(f"# N_COV_REPS    = {N_COV_REPS}")
    print(f"# N_BOOT1K      = {N_BOOT1K_ADDON}")
    print(f"# N_JOBS        = {N_JOBS}")
    print("#" * 70 + "\n")

    base_learners = make_base_learners()

    # 1. Low event-rate and reverse-overlap scenarios
    run_addon_scenarios(base_learners)
    collect_addon_scenario_metrics()

    # 2. Add missing moderate nonlinear bootstrap=1000 scenario
    run_boot1000_extra(base_learners)

    # 3. Add independent covariance sensitivity for challenging primary scenarios
    run_covariance_extra(base_learners)
    collect_covariance_metrics()

    print("\n" + "#" * 70)
    print("# ADDON ANALYSES COMPLETE")
    print("# New outputs:")
    print(f"#   {OUT_DIR}/addon_eventrate_reverseoverlap_metrics.csv")
    print(f"#   {OUT_DIR}/boot1000_sensitivity_updated.csv")
    print(f"#   {OUT_DIR}/covariance_sensitivity_expanded.csv")
    print("#" * 70 + "\n")
