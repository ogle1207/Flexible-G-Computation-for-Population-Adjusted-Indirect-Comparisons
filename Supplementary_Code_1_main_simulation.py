#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
ITC simulation runner
==============================================================================
Single-machine use: run this script directly.
  - Generates data, runs simulations, checks completeness, and performs final analyses.
  - Rerunning resumes incomplete scenario-method combinations.
==============================================================================
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTENSOR_FLAGS']    = 'cxx='
os.environ['XGBOOST_VERBOSITY'] = '0'

from itertools import product
import matplotlib
matplotlib.use('Agg')
import sys, pickle, warnings, time, contextlib, gc, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit, expit
from scipy.stats   import norm, ks_2samp
from scipy.optimize import minimize, minimize_scalar
import statsmodels.api  as sm
import statsmodels.formula.api as smf
from sklearn.utils          import resample
from sklearn.base           import clone, BaseEstimator, ClassifierMixin
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from pygam   import LogisticGAM
from joblib  import Parallel, delayed
from tqdm    import tqdm

warnings.filterwarnings("ignore")

def write_package_versions(out_dir="Final_Results_Analysis_ITC"):
    """Write Python and package versions used for reproducibility."""
    import importlib.metadata as metadata

    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "pygam",
        "matplotlib",
        "seaborn",
        "joblib",
        "tqdm",
    ]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "package_versions.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"python=={sys.version.split()[0]}\n")
        for package in packages:
            try:
                version = metadata.version(package)
            except metadata.PackageNotFoundError:
                version = "not installed"
            handle.write(f"{package}=={version}\n")
    return path


# =============================================================================
# Single-machine settings
# =============================================================================
MACHINE_ID    = 1
N_MACHINES    = 1
ANALYSIS_ONLY = False    # Run simulations first, then run analysis.

assert 1 <= MACHINE_ID <= N_MACHINES, "MACHINE_ID must be between 1 and N_MACHINES"

# =============================================================================
# 0. Utility functions
# =============================================================================
@contextlib.contextmanager
def suppress_all_output():
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception: pass
    devnull_fd      = os.open(os.devnull, os.O_RDWR)
    saved_stdout_fd = os.dup(1); saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1); os.dup2(devnull_fd, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); yield
    finally:
        try: sys.stdout.flush(); sys.stderr.flush()
        except Exception: pass
        os.dup2(saved_stdout_fd, 1); os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd);   os.close(saved_stderr_fd); os.close(devnull_fd)

def _safe_X(X):
    if isinstance(X, pd.DataFrame): X = X.values
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def _safe_y(y):
    if isinstance(y, (pd.Series, pd.DataFrame)): y = y.values
    return np.ascontiguousarray(np.asarray(y).ravel().astype(np.int32))

def _wald_ci(est, var):
    if np.isnan(est) or np.isnan(var) or var < 0:
        return np.nan, np.nan
    se = np.sqrt(var)
    return est + norm.ppf(0.025)*se, est + norm.ppf(0.975)*se

def normalize_record_keys(record):
    return {k.replace(".","_").replace(" ","_"): v for k, v in record.items()}


# =============================================================================
# 1. Global settings
# =============================================================================
if not os.path.exists("Data"): os.makedirs("Data")

np.random.seed(555)

N_sim        = 1000
RESAMPLES    = 100
RESAMPLES_CF = 100
N_STAR       = 2000
BATCH_SIZE   = 25

N_AC_LIST    = [200, 400, 600]
N_BC         = 600
allocation   = 2/3
b_trt        = np.log(0.17)
b_trt_B      = np.log(0.17 * 0.75)
b_X          = -np.log(0.5)
b_EM         = -np.log(0.67)
event_rate   = 0.35
meanX_AC_LIST = [0.45, 0.3, 0.15]
meanX_BC      = 0.6
sdX, corX     = 0.4, 0.2
SCEN_TYPES    = ["linear", "nonlinear"]

pc = pd.DataFrame(list(product(N_AC_LIST, meanX_AC_LIST, SCEN_TYPES)),
                  columns=["N_AC","meanX_AC","type"])

pc_nonzero = pd.DataFrame(list(product(N_AC_LIST, meanX_AC_LIST, SCEN_TYPES)),
                          columns=["N_AC","meanX_AC","type"])

KEY_SCEN_1000B = [
    dict(N_AC=200, meanX_AC=0.15, type="nonlinear"),
    dict(N_AC=600, meanX_AC=0.45, type="linear"),
]
N_SIM_BOOT1K = 500

def scenario_seed(tag: str) -> int:
    return int(hashlib.md5(tag.encode()).hexdigest(), 16) % (2**31 - 1)

def main_fid(row):
    return f"N_AC{int(row['N_AC'])}meanX_AC{row['meanX_AC']}_{row['type']}"

def nz_fid(row):
    return f"NONZERO_N_AC{int(row['N_AC'])}meanX_AC{row['meanX_AC']}_{row['type']}"

def boot_fid(kc):
    return f"N_AC{kc['N_AC']}meanX_AC{kc['meanX_AC']}_{kc['type']}"

# ---------------------------------------------------------------------------
# Single-machine task assignment
# ---------------------------------------------------------------------------
ALL_MAIN_FIDS = {main_fid(r) for _, r in pc.iterrows()}
ALL_NZ_FIDS   = {nz_fid(r)   for _, r in pc_nonzero.iterrows()}

if ANALYSIS_ONLY:
    SKIP_FIDS = set()
    print("\n[ANALYSIS_ONLY] Final analysis only; no scenarios are skipped")
else:
    all_tasks = []
    for _, r in pc.iterrows():
        all_tasks.append((int(r['N_AC']), float(r['meanX_AC']),
                          r['type'], 'main', main_fid(r)))
    for _, r in pc_nonzero.iterrows():
        all_tasks.append((int(r['N_AC']), float(r['meanX_AC']),
                          r['type'], 'nz', nz_fid(r)))
    all_tasks.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))

    my_fids = set()
    for i, t in enumerate(all_tasks):
        if i % N_MACHINES == (MACHINE_ID - 1):
            my_fids.add(t[4])
    SKIP_FIDS = {t[4] for t in all_tasks} - my_fids

    print("\n" + "="*70)
    print(f"  Single machine - total {len(my_fids)} ")
    print("="*70)
    for t in all_tasks:
        mark = "*" if t[4] in my_fids else " "
        print(f"   {mark}  N_AC={t[0]:>3}  meanX={t[1]:.2f}  type={t[2]:<9}"
              f"  kind={t[3]:<4}  fid={t[4]}")
    print("="*70 + "\n")

MAIN_FIDS_ALL  = ALL_MAIN_FIDS - SKIP_FIDS
NZ_FIDS_ALL    = ALL_NZ_FIDS - SKIP_FIDS
BOOT_SCENS_ALL = [kc for kc in KEY_SCEN_1000B if boot_fid(kc) not in SKIP_FIDS]

print(f"[assigned tasks] primary {len(MAIN_FIDS_ALL)},"
      f" nonzero {len(NZ_FIDS_ALL)},"
      f" Boot1k {len(BOOT_SCENS_ALL)}")


# =============================================================================
# 2. Data generation
# =============================================================================
def gen_data_internal(N, b_trt, b_X, b_EM, b_0, meanX, sdX, corX,
                     allocation, type="linear"):
    rho = np.full((4,4), corX); np.fill_diagonal(rho, 1.0)
    N_active = int(round(N * allocation)); N_control = N - N_active
    sd_vec = np.full(4, sdX)
    cov_mat = np.diag(sd_vec) @ rho @ np.diag(sd_vec)
    mean_vec = np.full(4, meanX)
    X_active  = np.random.multivariate_normal(mean_vec, cov_mat, N_active)
    X_control = np.random.multivariate_normal(mean_vec, cov_mat, N_control)
    X = pd.DataFrame(np.vstack([X_active, X_control]), columns=["X1","X2","X3","X4"])
    trt = np.array([1]*N_active + [0]*N_control)
    X1,X2,X3,X4 = X["X1"],X["X2"],X["X3"],X["X4"]
    if type == "linear":
        LP = (b_0 + b_X*X1 + b_X*X2 + b_X*X3 + b_X*X4
              + b_trt*trt + b_EM*X1*trt + b_EM*X2*trt)
    else:
        rs = 0.6*X1 + 0.4*X3
        baseline = np.where(rs<-0.3, 0.5*rs,
                    np.where(rs<0.5, 0.9*rs**2, 1.4*rs))
        baseline = baseline + 0.8*X4
        bm_high = (X2>0.6).astype(float)
        hi = (rs>0.4).astype(float)
        mi = ((rs<=0.4)&(rs>=-0.1)).astype(float)
        lo = (rs<-0.1).astype(float)
        em_region = 2.5*hi*bm_high + 1.3*mi*bm_high - 1.0*lo*(1-bm_high)
        em_step = np.where(X1>0.8, 1.5, 0.0)
        em_total = em_region + 0.6*em_step
        LP = b_0 + baseline + b_trt*trt + b_EM*em_total*trt
    yprob = expit(LP); y = np.random.binomial(1, yprob, N)
    return pd.DataFrame({"X1":X1,"X2":X2,"X3":X3,"X4":X4,"trt":trt,"y":y})


def find_optimal_b0(target_type):
    def obj(b0):
        df = gen_data_internal(100000, b_trt, b_X, b_EM, b0,
                               meanX_BC, sdX, corX, allocation, target_type)
        return (df["y"].mean()-event_rate)**2
    return minimize_scalar(obj, bounds=(-8,8), method="bounded").x


def build_ald(IPD_BC_list):
    ALD = []
    for df in IPD_BC_list:
        agg_cov = df[["X1","X2","X3","X4"]].agg(["mean","std"]).unstack()
        agg_cov.index = [f"{c}_{s}" for c,s in agg_cov.index]
        ag = df.groupby("trt")["y"].agg(y_sum="sum", y_bar="mean", N_count="count")
        try:
            sB = ag.loc[1].rename({"y_sum":"y.B_sum","y_bar":"y.B_bar","N_count":"N.B"})
        except KeyError: sB = pd.Series({"y.B_sum":0,"y.B_bar":0.0,"N.B":0})
        try:
            sC = ag.loc[0].rename({"y_sum":"y.C_sum","y_bar":"y.C_bar","N_count":"N.C"})
        except KeyError: sC = pd.Series({"y.C_sum":0,"y.C_bar":0.0,"N.C":0})
        ALD.append(pd.concat([agg_cov, sB, sC]).to_frame().T)
    return pd.concat(ALD, ignore_index=True)


# =============================================================================
# 3. Estimator utilities
# =============================================================================
def estimate_maic_weights(X_EM, target_means):
    from scipy.special import logsumexp
    Xm = X_EM.astype(np.float64).values
    tv = np.array([target_means[c] for c in X_EM.columns], dtype=np.float64)
    def obj(a): return logsumexp(Xm @ a) - np.dot(tv, a)
    def grd(a):
        w = np.exp(Xm@a - logsumexp(Xm@a)); return Xm.T@w - tv
    try:
        r = minimize(obj, np.zeros(Xm.shape[1]), jac=grd, method="BFGS")
        if not r.success: return np.ones(Xm.shape[0])
        z = Xm @ r.x; return np.exp(z - z.max())
    except Exception: return np.ones(Xm.shape[0])


def calculate_bc_effect(data_BC):
    def get(d, ks):
        for k in ks:
            if k in d: return d[k]
        return np.nan
    try:
        yB = float(get(data_BC,["y_B_sum","yB_sum","y.B_sum"]))
        NB = float(get(data_BC,["N_B","NB","N.B"]))
        yC = float(get(data_BC,["y_C_sum","yC_sum","y.C_sum"]))
        NC = float(get(data_BC,["N_C","NC","N.C"]))
        if any(x<=0 for x in [yB,NB-yB,yC,NC-yC]): return np.nan, np.nan
        h = np.log((yB*(NC-yC))/(yC*(NB-yB)))
        v = 1/yC + 1/(NC-yC) + 1/yB + 1/(NB-yB)
        return h, v
    except Exception: return np.nan, np.nan


def generate_pseudo_population(data_AC, data_BC, n_star, cor_method="ac_proxy"):
    cov_cols = ["X1","X2","X3","X4"]
    means = np.array([data_BC[f"{v}_mean"] for v in cov_cols], dtype=float)
    stds  = np.array([data_BC[f"{v}_std"]  for v in cov_cols], dtype=float)
    if cor_method == "independent":
        cov = np.diag(stds**2)
    else:
        try:
            rho = data_AC[cov_cols].corr().values
            cov = rho*np.outer(stds,stds) + np.eye(4)*1e-6
        except Exception:
            cov = np.diag(stds**2)
    samp = np.random.multivariate_normal(means, cov, size=n_star)
    return pd.DataFrame(samp, columns=cov_cols)


# =============================================================================
# 4. Safe XGB / GAM / SuperLearner
# =============================================================================
class SafeXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **xgb_params):
        xgb_params.setdefault("verbosity", 0)
        xgb_params.setdefault("n_jobs", 1)
        xgb_params.setdefault("nthread", 1)
        xgb_params.setdefault("tree_method", "hist")
        xgb_params.setdefault("objective", "binary:logistic")
        xgb_params.setdefault("eval_metric", "logloss")
        self.xgb_params = xgb_params
        self._model = None; self._fallback = False
        self.classes_ = np.array([0,1])
    def fit(self, X, y):
        X = _safe_X(X); y = _safe_y(y)
        try:
            with suppress_all_output():
                m = XGBClassifier(**self.xgb_params); m.fit(X,y)
            self._model = m; self._fallback = False
        except BaseException:
            try:
                lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                with suppress_all_output(): lr.fit(X,y)
                self._model = lr; self._fallback = True
            except Exception:
                self._model = float(np.mean(y)) if len(y) else 0.5
                self._fallback = True
        return self
    def predict_proba(self, X):
        X = _safe_X(X)
        try:
            if isinstance(self._model, float):
                p = np.clip(self._model, 1e-6, 1-1e-6)
                return np.column_stack([np.full(len(X),1-p), np.full(len(X),p)])
            with suppress_all_output(): return self._model.predict_proba(X)
        except BaseException:
            return np.ones((len(X),2))*0.5
    def predict(self, X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)


class LogisticGAMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self._model=None; self.classes_=np.array([0,1]); self._fallback=False
    def fit(self, X, y):
        X = _safe_X(X); y = _safe_y(y)
        try:
            with suppress_all_output():
                gam = LogisticGAM()
                gam.gridsearch(X, y, lam=np.logspace(-3,2,3), progress=False)
            self._model = gam; self._fallback = False
        except Exception:
            lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            with suppress_all_output(): lr.fit(X,y)
            self._model = lr; self._fallback = True
        return self
    def predict_proba(self, X):
        X = _safe_X(X)
        if not self._fallback and isinstance(self._model, LogisticGAM):
            try:
                with suppress_all_output():
                    mu = np.clip(np.nan_to_num(self._model.predict_mu(X), nan=0.5),
                                 1e-6,1-1e-6)
                return np.column_stack([1-mu, mu])
            except Exception:
                return np.ones((len(X),2))*0.5
        with suppress_all_output(): return self._model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)


class SuperLearnerBinary(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners, K=3, random_state=444):
        self.base_learners = base_learners; self.K = K
        self.random_state = random_state
        self.fitted_learners_ = None; self.weights_ = None
        self.cv_risks_ = None; self.learner_names_ = None
        self.classes_ = np.array([0,1])
    def _solve_weights(self, P, y):
        n,L = P.shape
        def obj(w): return np.mean((y-P@w)**2)
        cons = {'type':'eq','fun': lambda w: np.sum(w)-1.0}
        bnds = [(0.,None)]*L
        r = minimize(obj, np.ones(L)/L, method="SLSQP",
                     bounds=bnds, constraints=[cons])
        if (not r.success) or np.any(np.isnan(r.x)): return np.ones(L)/L
        return r.x
    def fit(self, X, y):
        X = _safe_X(X); y = _safe_y(y)
        n,p = X.shape; L = len(self.base_learners)
        P_oof = np.zeros((n,L))
        self.fitted_learners_ = {nm:[] for nm,_ in self.base_learners}
        self.learner_names_ = [nm for nm,_ in self.base_learners]
        kf = KFold(n_splits=self.K, shuffle=True, random_state=self.random_state)
        for tr,val in kf.split(X):
            Xt = np.ascontiguousarray(X[tr]); yt = np.ascontiguousarray(y[tr])
            Xv = np.ascontiguousarray(X[val])
            for ell,(nm,proto) in enumerate(self.base_learners):
                try:
                    m = clone(proto)
                    with suppress_all_output(): m.fit(Xt, yt)
                    self.fitted_learners_[nm].append(m)
                    with suppress_all_output():
                        pv = np.clip(m.predict_proba(Xv)[:,1], 1e-6,1-1e-6)
                    P_oof[val,ell] = pv
                except BaseException:
                    P_oof[val,ell] = 0.5
        self.weights_ = self._solve_weights(P_oof, y)
        self.cv_risks_ = np.array([
            -np.mean(y*np.log(np.clip(P_oof[:,ell],1e-6,1-1e-6))
                     + (1-y)*np.log(np.clip(1-P_oof[:,ell],1e-6,1-1e-6)))
            for ell in range(L)
        ])
        return self
    def predict_proba(self, X):
        X = _safe_X(X); n = X.shape[0]; L = len(self.base_learners)
        P = np.zeros((n,L))
        for ell,(nm,_) in enumerate(self.base_learners):
            ms = self.fitted_learners_.get(nm,[])
            if not ms: P[:,ell] = 0.5; continue
            ps = np.zeros(n); cnt = 0
            for m in ms:
                try:
                    with suppress_all_output():
                        ps += np.clip(m.predict_proba(X)[:,1], 1e-6,1-1e-6)
                    cnt += 1
                except BaseException: continue
            P[:,ell] = ps/max(cnt,1) if cnt>0 else 0.5
        p1 = np.clip(P @ self.weights_, 1e-6, 1-1e-6)
        return np.column_stack([1-p1, p1])
    def predict(self, X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def get_diagnostics(self):
        if self.weights_ is None or self.learner_names_ is None: return None
        return {"weights":dict(zip(self.learner_names_, self.weights_.tolist())),
                "cv_risks":dict(zip(self.learner_names_, self.cv_risks_.tolist())),
                "dominant":self.learner_names_[int(np.argmax(self.weights_))]}


# =============================================================================
# 5. Estimator wrappers
# =============================================================================
def maic_wrapper(data_AC, data_BC, n_resamples):
    target = {"X1": data_BC["X1_mean"], "X2": data_BC["X2_mean"]}
    try:
        wm = estimate_maic_weights(data_AC[["X1","X2"]], target)
        ess_main = (wm.sum()**2)/np.sum(wm**2)
        ess_pct  = ess_main/len(data_AC)*100
    except Exception:
        ess_main, ess_pct = np.nan, np.nan
    hat = []
    for _ in range(n_resamples):
        try:
            db = resample(data_AC, replace=True)
            w  = estimate_maic_weights(db[["X1","X2"]], target)
            with suppress_all_output():
                fit = sm.GLM(db["y"], sm.add_constant(db["trt"]),
                             family=sm.families.Binomial(),
                             freq_weights=w).fit()
            hat.append(fit.params["trt"])
        except Exception: continue
    if not hat: return np.nan,np.nan,np.nan,np.nan, ess_main, ess_pct
    bc, var_bc = calculate_bc_effect(data_BC)
    est = np.mean(hat) - bc
    var = np.var(hat, ddof=1) + var_bc
    lci, uci = _wald_ci(est, var)
    return est, var, lci, uci, ess_main, ess_pct


def stc_wrapper(data_AC, data_BC):
    try:
        f = (f"y ~ X3 + X4 + trt * I(X1 - {data_BC['X1_mean']}) + "
             f"trt * I(X2 - {data_BC['X2_mean']})")
        with suppress_all_output(): m = smf.logit(f, data=data_AC).fit(disp=0)
        bc, var_bc = calculate_bc_effect(data_BC)
        est = m.params["trt"] - bc
        var = m.cov_params().loc["trt","trt"] + var_bc
        lci, uci = _wald_ci(est, var)
        return est, var, lci, uci
    except Exception: return np.nan,np.nan,np.nan,np.nan


def gcomp_ml_wrapper(data_AC, data_BC, n_resamples, n_star):
    try: x_star = generate_pseudo_population(data_AC, data_BC, n_star)
    except Exception: return np.nan,np.nan,np.nan,np.nan
    hat = []
    for _ in range(n_resamples):
        try:
            db = resample(data_AC, replace=True)
            with suppress_all_output():
                m = smf.logit("y ~ X3 + X4 + trt*X1 + trt*X2", data=db).fit(disp=0)
            dA = x_star.copy(); dA["trt"]=1
            dC = x_star.copy(); dC["trt"]=0
            mA = np.clip(m.predict(dA).mean(), 1e-6,1-1e-6)
            mC = np.clip(m.predict(dC).mean(), 1e-6,1-1e-6)
            hat.append(logit(mA)-logit(mC))
        except Exception: continue
    if not hat: return np.nan,np.nan,np.nan,np.nan
    bc, var_bc = calculate_bc_effect(data_BC)
    est = np.mean(hat) - bc; var = np.var(hat, ddof=1) + var_bc
    return est, var, *_wald_ci(est, var)


def gcomp_sl_wrapper(data_AC, data_BC, n_resamples, n_star, base_learners,
                     K_sl=3, cor_method="ac_proxy"):
    try: x_star = generate_pseudo_population(data_AC, data_BC, n_star, cor_method)
    except Exception: return np.nan,np.nan,np.nan,np.nan
    cols = ["trt","X1","X2","X3","X4"]; hat = []
    for b_idx in range(n_resamples):
        try:
            db = resample(data_AC, replace=True)
            sl = SuperLearnerBinary(base_learners=base_learners, K=K_sl,
                                    random_state=444 + 17*b_idx)
            with suppress_all_output(): sl.fit(db[cols].values, db["y"].values)
            dA = x_star.copy(); dA["trt"]=1
            dC = x_star.copy(); dC["trt"]=0
            with suppress_all_output():
                mA = np.clip(sl.predict_proba(dA[cols].values)[:,1].mean(), 1e-6,1-1e-6)
                mC = np.clip(sl.predict_proba(dC[cols].values)[:,1].mean(), 1e-6,1-1e-6)
            v = logit(mA)-logit(mC)
            if not np.isnan(v): hat.append(v)
        except Exception: continue
    if not hat: return np.nan,np.nan,np.nan,np.nan
    bc, var_bc = calculate_bc_effect(data_BC)
    hat = np.array(hat)
    rng = np.random.default_rng()
    bcd = rng.normal(bc, np.sqrt(max(var_bc,0.0)), size=len(hat))
    boot_itc = hat - bcd
    est = np.mean(hat) - bc; var = np.var(hat, ddof=1) + var_bc
    return est, var, np.percentile(boot_itc,2.5), np.percentile(boot_itc,97.5)


def gcomp_sl_crossfit_wrapper(data_AC, data_BC, n_resamples_cf, n_star,
                              base_learners, K_cf=5, K_sl=3):
    try: x_star = generate_pseudo_population(data_AC, data_BC, n_star)
    except Exception: return np.nan,np.nan,np.nan,np.nan
    cols = ["trt","X1","X2","X3","X4"]
    dA_full = x_star.copy(); dA_full["trt"]=1
    dC_full = x_star.copy(); dC_full["trt"]=0
    XA = _safe_X(dA_full[cols].values); XC = _safe_X(dC_full[cols].values)
    hat = []
    for b_idx in range(n_resamples_cf):
        try:
            db = resample(data_AC, replace=True)
            X = _safe_X(db[cols].values); y = _safe_y(db["y"].values)
            kf = KFold(n_splits=K_cf, shuffle=True, random_state=444+b_idx)
            mAf = np.zeros(K_cf); mCf = np.zeros(K_cf)
            for k,(tr,_) in enumerate(kf.split(X)):
                if len(np.unique(y[tr]))<2:
                    mAf[k]=mCf[k]=float(np.mean(y)); continue
                sl = SuperLearnerBinary(base_learners=base_learners, K=K_sl,
                                        random_state=444+7*b_idx+31*k)
                with suppress_all_output():
                    sl.fit(X[tr], y[tr])
                    mAf[k] = np.clip(sl.predict_proba(XA)[:,1].mean(), 1e-6,1-1e-6)
                    mCf[k] = np.clip(sl.predict_proba(XC)[:,1].mean(), 1e-6,1-1e-6)
            mA = np.clip(mAf.mean(),1e-6,1-1e-6); mC = np.clip(mCf.mean(),1e-6,1-1e-6)
            v = logit(mA)-logit(mC)
            if not np.isnan(v): hat.append(v)
        except Exception: continue
    if not hat: return np.nan,np.nan,np.nan,np.nan
    bc, var_bc = calculate_bc_effect(data_BC)
    hat = np.array(hat)
    rng = np.random.default_rng()
    bcd = rng.normal(bc, np.sqrt(max(var_bc,0.0)), size=len(hat))
    boot_itc = hat - bcd
    est = np.mean(hat)-bc; var = np.var(hat, ddof=1)+var_bc
    return est, var, np.percentile(boot_itc,2.5), np.percentile(boot_itc,97.5)


def tmle_wrapper(data_AC, data_BC, n_star, base_learners, K_sl=3):
    try:
        x_star = generate_pseudo_population(data_AC, data_BC, n_star)
        cols = ["X1","X2","X3","X4"]
        X_AC = _safe_X(data_AC[cols].values)
        A = np.ascontiguousarray(data_AC["trt"].values.astype(np.float64))
        Y = np.ascontiguousarray(data_AC["y"].values.astype(np.float64))
        X_star = _safe_X(x_star[cols].values); n_AC = len(X_AC); n_st = len(X_star)

        X_pool = np.vstack([X_AC, X_star])
        S_pool = np.array([0]*n_AC+[1]*n_st, dtype=np.float64)
        with suppress_all_output():
            dr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            dr.fit(X_pool, S_pool)
            p_bc = np.clip(dr.predict_proba(X_AC)[:,1], 0.01, 0.99)
        r_raw = p_bc/(1-p_bc); r = r_raw/r_raw.mean()

        with suppress_all_output():
            g = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            g.fit(X_AC, A.astype(int))
            g1 = np.clip(g.predict_proba(X_AC)[:,1], 0.05, 0.95)

        sl_Q = SuperLearnerBinary(base_learners=base_learners, K=K_sl, random_state=444)
        XA_in = np.column_stack([A, X_AC])
        with suppress_all_output():
            sl_Q.fit(_safe_X(XA_in), _safe_y(Y.astype(int)))
            X1a = np.column_stack([np.ones(n_AC),  X_AC])
            X0a = np.column_stack([np.zeros(n_AC), X_AC])
            Q1 = np.clip(sl_Q.predict_proba(_safe_X(X1a))[:,1], 1e-6,1-1e-6)
            Q0 = np.clip(sl_Q.predict_proba(_safe_X(X0a))[:,1], 1e-6,1-1e-6)
        QA = np.where(A==1, Q1, Q0)

        H = r*(A/g1 - (1-A)/(1-g1))
        off = logit(QA)
        def nll(eps):
            p = np.clip(expit(off+eps*H), 1e-6,1-1e-6)
            return -np.mean(Y*np.log(p)+(1-Y)*np.log(1-p))
        eps = minimize_scalar(nll, bounds=(-5,5), method='bounded').x
        Q1e = expit(logit(Q1)+eps*r/g1); Q0e = expit(logit(Q0)-eps*r/(1-g1))
        mu1 = np.clip(np.mean(r*Q1e), 1e-6,1-1e-6)
        mu0 = np.clip(np.mean(r*Q0e), 1e-6,1-1e-6)
        tlogOR = logit(mu1)-logit(mu0)
        phi1 = r*(A*(Y-Q1e)/g1 + Q1e - mu1)
        phi0 = r*((1-A)*(Y-Q0e)/(1-g1) + Q0e - mu0)
        d1 = 1/(mu1*(1-mu1)); d0 = 1/(mu0*(1-mu0))
        phi = d1*phi1 - d0*phi0
        var_lor = np.var(phi, ddof=1)/n_AC
        bc, var_bc = calculate_bc_effect(data_BC)
        if np.isnan(bc): return np.nan,np.nan,np.nan,np.nan
        est = tlogOR - bc; var = max(var_lor,1e-10) + var_bc
        return est, var, *_wald_ci(est, var)
    except Exception: return np.nan,np.nan,np.nan,np.nan


def tmle_sl_wrapper(data_AC, data_BC, n_star, base_learners, K_sl=3, K_cf=5):
    try:
        x_star = generate_pseudo_population(data_AC, data_BC, n_star)
        cols = ["X1","X2","X3","X4"]
        X_AC = _safe_X(data_AC[cols].values)
        A = np.ascontiguousarray(data_AC["trt"].values.astype(np.float64))
        Y = np.ascontiguousarray(data_AC["y"].values.astype(np.float64))
        X_star = _safe_X(x_star[cols].values); n_AC = len(X_AC); n_st = len(X_star)

        X_pool = np.vstack([X_AC, X_star])
        S_pool = np.array([0]*n_AC+[1]*n_st, dtype=np.float64)
        sl_dr = SuperLearnerBinary(base_learners=base_learners, K=K_sl, random_state=11)
        with suppress_all_output():
            sl_dr.fit(_safe_X(X_pool), _safe_y(S_pool.astype(int)))
            p_bc = np.clip(sl_dr.predict_proba(X_AC)[:,1], 0.02, 0.98)
        r_raw = p_bc/(1-p_bc); r = r_raw/r_raw.mean()

        Q1 = np.zeros(n_AC); Q0 = np.zeros(n_AC); g1 = np.zeros(n_AC)
        kf = KFold(n_splits=K_cf, shuffle=True, random_state=999)
        for k, (tr, te) in enumerate(kf.split(X_AC)):
            if len(np.unique(Y[tr]))<2 or len(np.unique(A[tr]))<2:
                Q1[te]=Q0[te]=float(np.mean(Y)); g1[te]=float(np.mean(A)); continue
            XA_tr = np.column_stack([A[tr], X_AC[tr]])
            sQ = SuperLearnerBinary(base_learners=base_learners, K=K_sl,
                                    random_state=2024 + 37*k)
            sg = SuperLearnerBinary(base_learners=base_learners, K=K_sl,
                                    random_state=3024 + 37*k)
            with suppress_all_output():
                sQ.fit(_safe_X(XA_tr), _safe_y(Y[tr]))
                sg.fit(_safe_X(X_AC[tr]), _safe_y(A[tr]))
                X1t = np.column_stack([np.ones(len(te)),  X_AC[te]])
                X0t = np.column_stack([np.zeros(len(te)), X_AC[te]])
                Q1[te] = np.clip(sQ.predict_proba(_safe_X(X1t))[:,1], 1e-6,1-1e-6)
                Q0[te] = np.clip(sQ.predict_proba(_safe_X(X0t))[:,1], 1e-6,1-1e-6)
                g1[te] = np.clip(sg.predict_proba(_safe_X(X_AC[te]))[:,1], 0.05, 0.95)
        QA = np.where(A==1, Q1, Q0)
        H = r*(A/g1 - (1-A)/(1-g1))
        off = logit(QA)
        def nll(eps):
            p = np.clip(expit(off+eps*H),1e-6,1-1e-6)
            return -np.mean(Y*np.log(p)+(1-Y)*np.log(1-p))
        eps = minimize_scalar(nll, bounds=(-5,5), method='bounded').x
        Q1e = expit(logit(Q1)+eps*r/g1); Q0e = expit(logit(Q0)-eps*r/(1-g1))
        mu1 = np.clip(np.mean(r*Q1e),1e-6,1-1e-6)
        mu0 = np.clip(np.mean(r*Q0e),1e-6,1-1e-6)
        tlogOR = logit(mu1)-logit(mu0)
        phi1 = r*(A*(Y-Q1e)/g1 + Q1e - mu1)
        phi0 = r*((1-A)*(Y-Q0e)/(1-g1) + Q0e - mu0)
        d1 = 1/(mu1*(1-mu1)); d0 = 1/(mu0*(1-mu0))
        phi = d1*phi1 - d0*phi0
        var_lor = np.var(phi, ddof=1)/n_AC
        bc, var_bc = calculate_bc_effect(data_BC)
        if np.isnan(bc): return np.nan,np.nan,np.nan,np.nan
        est = tlogOR - bc; var = max(var_lor,1e-10) + var_bc
        return est, var, *_wald_ci(est, var)
    except Exception: return np.nan,np.nan,np.nan,np.nan


# =============================================================================
# 6. Diagnostics
# =============================================================================
def compute_overlap_diagnostics(data_AC, data_BC, n_star=2000):
    diag = {"ESS_pct":np.nan, "max_w_ratio":np.nan, "PS_KS":np.nan,
            "PS_AC_mean":np.nan, "PS_AC_pct95":np.nan,
            "g1_trunc_pct":np.nan, "dr_trunc_pct":np.nan}
    cols = ["X1","X2","X3","X4"]
    try:
        target = {"X1": data_BC["X1_mean"], "X2": data_BC["X2_mean"]}
        w = estimate_maic_weights(data_AC[["X1","X2"]], target)
        ess = (w.sum()**2)/np.sum(w**2)
        diag["ESS_pct"] = ess/len(data_AC)*100
        diag["max_w_ratio"] = float(w.max()/w.mean())
    except Exception: pass
    try:
        x_star = generate_pseudo_population(data_AC, data_BC, n_star)
        Xp = np.vstack([data_AC[cols].values, x_star.values])
        Sp = np.array([0]*len(data_AC) + [1]*len(x_star))
        with suppress_all_output():
            ps = LogisticRegression(max_iter=1000).fit(Xp, Sp)
            psA_raw = ps.predict_proba(data_AC[cols].values)[:,1]
            psB_raw = ps.predict_proba(x_star.values)[:,1]
        diag["PS_KS"] = float(ks_2samp(psA_raw, psB_raw).statistic)
        diag["PS_AC_mean"]  = float(np.mean(psA_raw))
        diag["PS_AC_pct95"] = float(np.percentile(psA_raw, 95))
        dr_trunc = np.mean((psA_raw<=0.02) | (psA_raw>=0.98)) * 100
        diag["dr_trunc_pct"] = float(dr_trunc)
    except Exception: pass
    try:
        with suppress_all_output():
            g = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            g.fit(data_AC[cols].values, data_AC["trt"].values.astype(int))
            g1_raw = g.predict_proba(data_AC[cols].values)[:,1]
        g1_trunc = np.mean((g1_raw<=0.05) | (g1_raw>=0.95)) * 100
        diag["g1_trunc_pct"] = float(g1_trunc)
    except Exception: pass
    return diag


def compute_sl_diagnostics(data_AC, base_learners):
    try:
        cols = ["trt","X1","X2","X3","X4"]
        sl = SuperLearnerBinary(base_learners=base_learners, K=3, random_state=444)
        with suppress_all_output(): sl.fit(data_AC[cols].values, data_AC["y"].values)
        return sl.get_diagnostics()
    except Exception:
        return None


# =============================================================================
# 7. Replicate driver
# =============================================================================
def run_one_replicate(j, IPD_AC, ALD_BC, base_learners,
                      resamples, resamples_cf, n_star,
                      run_robustness=False, fast_diag=True):
    os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['NUMEXPR_NUM_THREADS']='1'
    os.environ['XGBOOST_VERBOSITY']='0'
    warnings.filterwarnings("ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            data_AC = IPD_AC[j]; data_BC = ALD_BC[j]
            res = {}
            def timed(fn, *a):
                t0 = time.time()
                try: out = fn(*a)
                except BaseException: out = (np.nan,)*4
                return (*out, time.time()-t0) if len(out)==4 else out+(time.time()-t0,)

            t0 = time.time()
            try: maic_out = maic_wrapper(data_AC, data_BC, resamples)
            except BaseException: maic_out = (np.nan,)*6
            res["maic"] = (maic_out[0], maic_out[1], maic_out[2], maic_out[3], time.time()-t0)
            res["maic_ess"] = (maic_out[4], maic_out[5])

            res["stc"]         = timed(stc_wrapper, data_AC, data_BC)
            res["gcomp_ml"]    = timed(gcomp_ml_wrapper, data_AC, data_BC, resamples, n_star)
            res["gcomp_sl"]    = timed(gcomp_sl_wrapper, data_AC, data_BC, resamples, n_star, base_learners)
            res["gcomp_sl_cf"] = timed(gcomp_sl_crossfit_wrapper, data_AC, data_BC,
                                       resamples_cf, n_star, base_learners)
            res["tmle"]        = timed(tmle_wrapper, data_AC, data_BC, n_star, base_learners)
            res["tmle_sl"]     = timed(tmle_sl_wrapper, data_AC, data_BC, n_star, base_learners)

            try: res["overlap"] = compute_overlap_diagnostics(data_AC, data_BC, n_star)
            except Exception:
                res["overlap"] = {"ESS_pct":np.nan,"max_w_ratio":np.nan,"PS_KS":np.nan,
                                  "PS_AC_mean":np.nan,"PS_AC_pct95":np.nan,
                                  "g1_trunc_pct":np.nan,"dr_trunc_pct":np.nan}

            if j < 50:
                try: res["sl_diag"] = compute_sl_diagnostics(data_AC, base_learners)
                except Exception: res["sl_diag"] = None
            else:
                res["sl_diag"] = None

            if run_robustness:
                res["gcomp_sl_indep"] = timed(gcomp_sl_wrapper, data_AC, data_BC,
                                              resamples, n_star, base_learners, 3, "independent")
            return res
        except BaseException:
            return None


# =============================================================================
# 8. Data-generation stage
# =============================================================================
def load_or_compute_settings():
    b0 = {}
    true_itc_nonzero = {}
    if os.path.exists("binary_settings.pkl"):
        try:
            with open("binary_settings.pkl", "rb") as f:
                existing = pickle.load(f)
            b0 = dict(existing.get("b_0_dict", {}) or {})
            true_itc_nonzero = dict(existing.get("true_itc_nonzero", {}) or {})
            print(f"[settings] loaded existing b_0 = {b0}")
            print(f"[settings] loaded existing true_itc_nonzero = {true_itc_nonzero}")
        except Exception as e:
            print(f"[settings] failed to read binary_settings.pkl: {e}")

    print("\n[settings] b_0 ...")
    for st in SCEN_TYPES:
        if st not in b0:
            np.random.seed(scenario_seed("B0_" + st))
            b0[st] = find_optimal_b0(st)
            print(f"  calculated {st}: b_0 = {b0[st]:.6f}")
        else:
            print(f"  existing     {st}: b_0 = {b0[st]:.6f}")

    print("\n[settings] true_itc_nonzero ...")
    for st in SCEN_TYPES:
        if st not in true_itc_nonzero:
            np.random.seed(88888)
            N_lg = 500000
            dA = gen_data_internal(N_lg, b_trt,   b_X, b_EM, b0[st], meanX_BC, sdX, corX, 1.0, st)
            dB = gen_data_internal(N_lg, b_trt_B, b_X, b_EM, b0[st], meanX_BC, sdX, corX, 1.0, st)
            muA = np.clip(dA["y"].mean(),1e-6,1-1e-6)
            muB = np.clip(dB["y"].mean(),1e-6,1-1e-6)
            true_itc_nonzero[st] = float(logit(muA)-logit(muB))
            print(f"  calculated {st}: ITC = {true_itc_nonzero[st]:.4f}")
        else:
            print(f"  existing     {st}: ITC = {true_itc_nonzero[st]:.4f}")

    settings = {"pc":pc, "pc_nonzero":pc_nonzero,
                "true_itc_nonzero":true_itc_nonzero,
                "b_0_dict":b0, "N_sim":N_sim, "allocation":allocation}
    try:
        with open("binary_settings.pkl","wb") as f:
            pickle.dump(settings, f)
    except Exception as e:
        print(f"[settings] write failed: {e}")

    return b0, true_itc_nonzero


def stage_generate_data():
    print("="*70); print(f"STAGE 1 - data generation (single machine)")
    print("="*70)
    b0, true_itc_nonzero = load_or_compute_settings()

    for i in range(len(pc)):
        sp = pc.iloc[i]; st = sp["type"]; n = int(sp["N_AC"]); m = float(sp["meanX_AC"])
        fid = f"N_AC{n}meanX_AC{m}_{st}"
        if fid in SKIP_FIDS:
            continue
        ipd_p = f"Data/IPD_AC_{fid}.pkl"; ald_p = f"Data/ALD_BC_{fid}.pkl"
        if os.path.exists(ipd_p) and os.path.exists(ald_p):
            print(f"  [exists] {fid}"); continue
        print(f"  generate primary {fid}")
        np.random.seed(scenario_seed("MAIN_"+fid))
        b0_st = b0[st]
        IPD_AC = [gen_data_internal(n, b_trt, b_X, b_EM, b0_st, m, sdX, corX, allocation, st)
                  for _ in range(N_sim)]
        IPD_BC = [gen_data_internal(N_BC, b_trt, b_X, b_EM, b0_st, meanX_BC, sdX, corX, allocation, st)
                  for _ in range(N_sim)]
        ALD_BC = build_ald(IPD_BC)
        with open(ipd_p,"wb") as f: pickle.dump(IPD_AC, f)
        with open(ald_p,"wb") as f: pickle.dump(ALD_BC, f)

    for i in range(len(pc_nonzero)):
        r = pc_nonzero.iloc[i]; st = r["type"]; n = int(r["N_AC"]); m = float(r["meanX_AC"])
        fid = f"NONZERO_N_AC{n}meanX_AC{m}_{st}"
        if fid in SKIP_FIDS:
            continue
        ipd_p = f"Data/IPD_AC_{fid}.pkl"; ald_p = f"Data/ALD_BC_{fid}.pkl"
        if os.path.exists(ipd_p) and os.path.exists(ald_p):
            print(f"  [exists] {fid}"); continue
        print(f"  generate nonzero {fid}")
        np.random.seed(scenario_seed(fid))
        b0_st = b0[st]
        IPD_AC = [gen_data_internal(n, b_trt, b_X, b_EM, b0_st, m, sdX, corX, allocation, st)
                  for _ in range(N_sim)]
        IPD_BC = [gen_data_internal(N_BC, b_trt_B, b_X, b_EM, b0_st, meanX_BC, sdX, corX, allocation, st)
                  for _ in range(N_sim)]
        ALD_BC = build_ald(IPD_BC)
        with open(ipd_p,"wb") as f: pickle.dump(IPD_AC, f)
        with open(ald_p,"wb") as f: pickle.dump(ALD_BC, f)
    print(">>> Data generation complete.")


# =============================================================================
# 9. Simulation stage
# =============================================================================
METHODS = ["maic","stc","gcomp_ml","gcomp_sl","gcomp_sl_cf","tmle","tmle_sl"]
METHODS_ROBUST = ["gcomp_sl_indep"]

def load_scenario_data(fid):
    try:
        with open(f"Data/IPD_AC_{fid}.pkl","rb") as f: ipd = pickle.load(f)
        with open(f"Data/ALD_BC_{fid}.pkl","rb") as f: ald = pickle.load(f)
        ald_list = [normalize_record_keys(r) for r in ald.to_dict(orient="records")]
        return ipd, ald_list
    except Exception: return None, None


def save_scen_results(fid, container, methods):
    for m in methods:
        bp = f"Results/{m.upper()}"; os.makedirs(bp, exist_ok=True)
        with open(f"{bp}/means_{fid}.pkl","wb") as f: pickle.dump(container[m]["means"], f)
        with open(f"{bp}/variances_{fid}.pkl","wb") as f: pickle.dump(container[m]["vars"], f)
        with open(f"{bp}/lcis_{fid}.pkl","wb") as f: pickle.dump(container[m]["lcis"], f)
        with open(f"{bp}/ucis_{fid}.pkl","wb") as f: pickle.dump(container[m]["ucis"], f)
        with open(f"{bp}/times_{fid}.pkl","wb") as f: pickle.dump(container[m]["times"], f)


def _method_done_count(fid, method):
    bp = f"Results/{method.upper()}"
    paths = [f"{bp}/means_{fid}.pkl",
             f"{bp}/variances_{fid}.pkl",
             f"{bp}/lcis_{fid}.pkl",
             f"{bp}/ucis_{fid}.pkl",
             f"{bp}/times_{fid}.pkl"]
    if not all(os.path.exists(p) for p in paths):
        return 0
    try:
        with open(paths[0],"rb") as f:
            n = len(pickle.load(f))
        return n
    except Exception:
        return 0

def _diag_done_count(fid):
    diag_path = f"Results/DIAGNOSTICS/diag_{fid}.pkl"
    if not os.path.exists(diag_path):
        return 0
    try:
        with open(diag_path, "rb") as f:
            d = pickle.load(f)
        n_ov  = len(d.get("overlap",  []))
        n_ess = len(d.get("maic_ess", []))
        return min(n_ov, n_ess)
    except Exception:
        return 0

def _load_existing_container(fid, methods_used):
    container = {m: {"means":[],"vars":[],"lcis":[],"ucis":[],"times":[]}
                 for m in methods_used}
    min_len = None
    for m in methods_used:
        bp = f"Results/{m.upper()}"
        paths = {"means": f"{bp}/means_{fid}.pkl",
                 "vars":  f"{bp}/variances_{fid}.pkl",
                 "lcis":  f"{bp}/lcis_{fid}.pkl",
                 "ucis":  f"{bp}/ucis_{fid}.pkl",
                 "times": f"{bp}/times_{fid}.pkl"}
        if all(os.path.exists(p) for p in paths.values()):
            try:
                for k, p in paths.items():
                    with open(p, "rb") as f:
                        container[m][k] = list(pickle.load(f))
            except Exception:
                container[m] = {"means":[],"vars":[],"lcis":[],"ucis":[],"times":[]}
        cur = len(container[m]["means"])
        min_len = cur if min_len is None else min(min_len, cur)
    if min_len is None:
        min_len = 0

    diag_path = f"Results/DIAGNOSTICS/diag_{fid}.pkl"
    if os.path.exists(diag_path):
        diag_n = _diag_done_count(fid)
        if diag_n < min_len:
            print(f"  [diag-mismatch] {fid}: methods={min_len}, diag={diag_n} "
                  f"-> truncate to {diag_n} rerun to align diagnostics")
            min_len = diag_n

    for m in methods_used:
        for k in ["means","vars","lcis","ucis","times"]:
            container[m][k] = container[m][k][:min_len]
    return container, min_len


def _load_existing_diag(fid, n_done):
    diag_path = f"Results/DIAGNOSTICS/diag_{fid}.pkl"
    diag = {"maic_ess":[], "overlap":[], "sl_diag":[]}
    if os.path.exists(diag_path):
        try:
            with open(diag_path,"rb") as f: d = pickle.load(f)
            diag["maic_ess"] = list(d.get("maic_ess", []))[:n_done]
            diag["overlap"]  = list(d.get("overlap",  []))[:n_done]
            diag["sl_diag"]  = list(d.get("sl_diag",  []))
        except Exception: pass
    return diag


def _scenario_fully_done(fid, methods_used, reps):
    for m in methods_used:
        if _method_done_count(fid, m) < reps:
            return False
    if _diag_done_count(fid) < reps:
        return False
    return True

def run_scenario(fid, ipd, ald_list, reps, base_learners, n_jobs,
                 resamples, resamples_cf, n_star,
                 run_robustness=False, save_diag=True):
    methods_used = METHODS + (METHODS_ROBUST if run_robustness else [])

    container, n_done = _load_existing_container(fid, methods_used)
    diag_container    = _load_existing_diag(fid, n_done)

    if n_done >= reps:
        print(f"  [complete] {fid}: {n_done}/{reps}")
        return
    if n_done > 0:
        print(f"  [resume] {fid}:  {n_done} / {reps} start")

    n_batches   = int(np.ceil(reps/BATCH_SIZE))
    start_batch = n_done // BATCH_SIZE

    pbar = tqdm(range(start_batch, n_batches), desc=fid,
                initial=start_batch, total=n_batches)
    for b in pbar:
        bs = max(b*BATCH_SIZE, n_done)
        be = min((b+1)*BATCH_SIZE, reps)
        if bs >= be: continue

        batch = None
        for attempt, nj in enumerate([n_jobs, max(1, n_jobs//2),
                                      max(1, n_jobs//4), 1]):
            try:
                with Parallel(
                    n_jobs=nj, backend="loky", verbose=0,
                    max_nbytes=None, batch_size=1,
                    pre_dispatch="2*n_jobs", timeout=7200,
                ) as parallel:
                    batch = parallel(
                        delayed(run_one_replicate)(
                            j, ipd, ald_list, base_learners,
                            resamples, resamples_cf, n_star, run_robustness)
                        for j in range(bs, be))
                break
            except Exception as e:
                print(f"\n  [batch {b} attempt {attempt} n_jobs={nj}] "
                      f"{type(e).__name__}: {e}")
                gc.collect(); time.sleep(5)
                if nj == 1:
                    print(f"  [batch {b}] serial fallback failed; fill batch with NaN")
                    batch = [None]*(be-bs)

        if batch is None:
            batch = [None]*(be-bs)

        for res in batch:
            if res is None:
                for m in methods_used:
                    container[m]["means"].append(np.nan)
                    container[m]["vars"].append(np.nan)
                    container[m]["lcis"].append(np.nan)
                    container[m]["ucis"].append(np.nan)
                    container[m]["times"].append(np.nan)
                diag_container["maic_ess"].append((np.nan,np.nan))
                diag_container["overlap"].append({})
                continue
            for m in methods_used:
                v = res.get(m, (np.nan,)*5)
                container[m]["means"].append(v[0])
                container[m]["vars"].append(v[1])
                container[m]["lcis"].append(v[2])
                container[m]["ucis"].append(v[3])
                container[m]["times"].append(v[4])
            diag_container["maic_ess"].append(res.get("maic_ess",(np.nan,np.nan)))
            diag_container["overlap"].append(res.get("overlap", {}))
            sd = res.get("sl_diag", None)
            if sd is not None: diag_container["sl_diag"].append(sd)

        n_done = len(container[methods_used[0]]["means"])

        save_scen_results(fid, container, methods_used)
        if save_diag:
            os.makedirs("Results/DIAGNOSTICS", exist_ok=True)
            with open(f"Results/DIAGNOSTICS/diag_{fid}.pkl","wb") as f:
                pickle.dump(diag_container, f)
        gc.collect()


def stage_run_simulation():
    print("="*70); print(f"STAGE 2 - simulation (single machine)")
    print("="*70)
    base_learners = [
        ("lr_l2", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)),
        ("lr_l1", LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=1000)),
        ("rf_shallow", RandomForestClassifier(n_estimators=100, max_depth=3,
                        max_features="sqrt", min_samples_leaf=10,
                        random_state=444, n_jobs=1)),
        ("xgb_shallow", SafeXGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05)),
        ("rf_deep", RandomForestClassifier(n_estimators=100, max_depth=None,
                        max_features="sqrt", min_samples_leaf=5,
                        random_state=444, n_jobs=1)),
        ("xgb_deep", SafeXGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1)),
        ("gam", LogisticGAMClassifier()),
    ]
    try: n_cpu = os.cpu_count() or 4
    except Exception: n_cpu = 4
    n_jobs = max(1, n_cpu - 2)
    print(f"  n_jobs={n_jobs}, CPU={n_cpu}")

    print("\n>>> Primary analysis")
    for i in range(len(pc)):
        row = pc.iloc[i]
        fid = f"N_AC{int(row['N_AC'])}meanX_AC{row['meanX_AC']}_{row['type']}"
        if fid in SKIP_FIDS: continue

        ipd, ald = load_scenario_data(fid)
        if ipd is None:
            print(f"  [missing data] {fid}")
            continue
        reps = min(N_sim, len(ipd))

        run_rob      = (int(row["N_AC"]) == 400 and row["meanX_AC"] == 0.3)
        methods_used = METHODS + (METHODS_ROBUST if run_rob else [])

        if _scenario_fully_done(fid, methods_used, reps):
            print(f"  [skip] {fid}  {reps}/{reps}")
            continue

        counts = {m: _method_done_count(fid, m) for m in methods_used}
        print(f"\n>>> {fid} (N={reps}, robust={run_rob}) | status: {counts}")
        try:
            run_scenario(fid, ipd, ald, reps, base_learners, n_jobs,
                         RESAMPLES, RESAMPLES_CF, N_STAR,
                         run_robustness=run_rob)
        except Exception as e:
            print(f"  [ERROR] {fid}: {type(e).__name__}: {e}")
            gc.collect()

    print("\n>>> nonzero ITC")
    for i in range(len(pc_nonzero)):
        row = pc_nonzero.iloc[i]
        fid = f"NONZERO_N_AC{int(row['N_AC'])}meanX_AC{row['meanX_AC']}_{row['type']}"
        if fid in SKIP_FIDS: continue

        ipd, ald = load_scenario_data(fid)
        if ipd is None:
            print(f"  [missing data] {fid}")
            continue
        reps = min(N_sim, len(ipd))
        methods_used = METHODS

        if _scenario_fully_done(fid, methods_used, reps):
            print(f"  [skip] {fid}  {reps}/{reps}")
            continue

        counts = {m: _method_done_count(fid, m) for m in methods_used}
        print(f"\n>>> {fid} (N={reps}) | status: {counts}")
        try:
            run_scenario(fid, ipd, ald, reps, base_learners, n_jobs,
                         RESAMPLES, RESAMPLES_CF, N_STAR,
                         run_robustness=False)
        except Exception as e:
            print(f"  [ERROR] {fid}: {type(e).__name__}: {e}")
            gc.collect()

    if len(BOOT_SCENS_ALL) > 0:
        print(f"\n>>> Bootstrap=1000 sensitivity({len(BOOT_SCENS_ALL)} )")
        os.makedirs("Results/BOOT1000", exist_ok=True)

        for kc in BOOT_SCENS_ALL:
            fid       = f"N_AC{kc['N_AC']}meanX_AC{kc['meanX_AC']}_{kc['type']}"
            if fid in SKIP_FIDS: continue
            save_path = f"Results/BOOT1000/boot1000_{fid}.pkl"

            out = []
            if os.path.exists(save_path):
                try:
                    with open(save_path, "rb") as f:
                        out = list(pickle.load(f))
                except Exception:
                    out = []

            ipd, ald = load_scenario_data(fid)
            if ipd is None:
                print(f"  [missing data] {fid}")
                continue
            reps   = min(N_SIM_BOOT1K, len(ipd))
            n_done = len(out)

            if n_done >= reps:
                print(f"  [skip] {fid} boot1k  {n_done}/{reps}")
                continue

            if n_done > 0:
                print(f"  [resume] boot1k {fid}: {n_done}/{reps}")
            else:
                print(f"  >>> boot1k {fid}: 0/{reps}")

            BS_BOOT = max(50, n_jobs*4)
            pbar = tqdm(range(n_done, reps, BS_BOOT), desc=f"boot1k {fid}")
            for bs in pbar:
                be = min(bs + BS_BOOT, reps)
                chunk = None
                for attempt, nj in enumerate([n_jobs, max(1, n_jobs // 2),
                                              max(1, n_jobs // 4), 1]):
                    try:
                        with Parallel(n_jobs=nj, backend="loky", verbose=0,
                                      max_nbytes=None, batch_size=1,
                                      pre_dispatch="2*n_jobs",
                                      timeout=14400) as parallel:
                            chunk = parallel(
                                delayed(run_one_replicate)(
                                    j, ipd, ald, base_learners,
                                    1000, 200, N_STAR, False)
                                for j in range(bs, be))
                        break
                    except Exception as e:
                        print(f"\n  [boot1k batch {bs}-{be} attempt {attempt} "
                              f"n_jobs={nj}] {type(e).__name__}: {e}")
                        gc.collect(); time.sleep(5)
                        if nj == 1:
                            print(f"  [boot1k batch {bs}-{be}] ,fill batch with None")
                            chunk = [None] * (be - bs)

                if chunk is None:
                    chunk = [None] * (be - bs)

                out.extend(chunk)
                with open(save_path, "wb") as f:
                    pickle.dump(out, f)
                gc.collect()

    print(">>> Simulation stage complete.")


# =============================================================================
# 9b. Completeness checks
# =============================================================================
def verify_all_complete(check_full_grid=True):
    missing = []

    def _need(fid):
        if check_full_grid:
            return True
        return fid not in SKIP_FIDS

    for _, row in pc.iterrows():
        fid = f"N_AC{int(row['N_AC'])}meanX_AC{row['meanX_AC']}_{row['type']}"
        if not _need(fid): continue
        run_rob      = (int(row["N_AC"]) == 400 and row["meanX_AC"] == 0.3)
        methods_used = METHODS + (METHODS_ROBUST if run_rob else [])
        for m in methods_used:
            cnt = _method_done_count(fid, m)
            if cnt < N_sim:
                missing.append((fid, m, cnt, N_sim))
        d_cnt = _diag_done_count(fid)
        if d_cnt < N_sim:
            missing.append((fid, "DIAG", d_cnt, N_sim))

    for _, row in pc_nonzero.iterrows():
        fid = f"NONZERO_N_AC{int(row['N_AC'])}meanX_AC{row['meanX_AC']}_{row['type']}"
        if not _need(fid): continue
        for m in METHODS:
            cnt = _method_done_count(fid, m)
            if cnt < N_sim:
                missing.append((fid, m, cnt, N_sim))
        d_cnt = _diag_done_count(fid)
        if d_cnt < N_sim:
            missing.append((fid, "DIAG", d_cnt, N_sim))

    boot_set = KEY_SCEN_1000B if check_full_grid else BOOT_SCENS_ALL
    for kc in boot_set:
        fid = f"N_AC{kc['N_AC']}meanX_AC{kc['meanX_AC']}_{kc['type']}"
        if not _need(fid): continue
        p = f"Results/BOOT1000/boot1000_{fid}.pkl"
        n = 0
        if os.path.exists(p):
            try:
                with open(p,"rb") as f: n = len(pickle.load(f))
            except Exception: n = 0
        if n < N_SIM_BOOT1K:
            missing.append((fid, "BOOT1000", n, N_SIM_BOOT1K))

    return (len(missing) == 0), missing


def find_incomplete_fids():
    incomplete_main = set()
    incomplete_nz   = set()
    incomplete_boot = []
    detail = []

    for _, row in pc.iterrows():
        fid = main_fid(row)
        run_rob      = (int(row["N_AC"]) == 400 and row["meanX_AC"] == 0.3)
        methods_used = METHODS + (METHODS_ROBUST if run_rob else [])
        for m in methods_used:
            cnt = _method_done_count(fid, m)
            if cnt < N_sim:
                incomplete_main.add(fid)
                detail.append((fid, m, cnt, N_sim))
        d_cnt = _diag_done_count(fid)
        if d_cnt < N_sim:
            incomplete_main.add(fid)
            detail.append((fid, "DIAG", d_cnt, N_sim))

    for _, row in pc_nonzero.iterrows():
        fid = nz_fid(row)
        for m in METHODS:
            cnt = _method_done_count(fid, m)
            if cnt < N_sim:
                incomplete_nz.add(fid)
                detail.append((fid, m, cnt, N_sim))
        d_cnt = _diag_done_count(fid)
        if d_cnt < N_sim:
            incomplete_nz.add(fid)
            detail.append((fid, "DIAG", d_cnt, N_sim))

    for kc in KEY_SCEN_1000B:
        fid = boot_fid(kc)
        p = f"Results/BOOT1000/boot1000_{fid}.pkl"
        n = 0
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    n = len(pickle.load(f))
            except Exception:
                n = 0
        if n < N_SIM_BOOT1K:
            incomplete_boot.append(kc)
            detail.append((fid, "BOOT1000", n, N_SIM_BOOT1K))

    return incomplete_main, incomplete_nz, incomplete_boot, detail


# =============================================================================
# 10. Analysis stage
# =============================================================================
methods_map = {
    "MAIC":            "MAIC",
    "STC":             "STC",
    "G-comp (ML)":     "GCOMP_ML",
    "G-comp (SL)":     "GCOMP_SL",
    "G-comp (SL, SA)": "GCOMP_SL_CF",
    "TMLE":            "TMLE",
    "TMLE (SL+CF)":    "TMLE_SL",
}

plot_method_labels = {
    "MAIC": "MAIC",
    "STC": "STC",
    "G-comp (ML)": "G-comp ML",
    "G-comp (SL)": "G-comp SL",
    "G-comp (SL, SA)": "G-comp SL, SA",
    "TMLE": "TMLE",
    "TMLE (SL+CF)": "TMLE SL+CF",
}

def process_metrics(means, variances, lcis, ucis, times, truth):
    valid = ~np.isnan(means) & ~np.isnan(lcis) & ~np.isnan(ucis)
    if not np.any(valid): return {}
    means = means[valid]; variances = np.clip(variances[valid], 0, None)
    lcis = lcis[valid]; ucis = ucis[valid]; n = len(means)
    bias = np.mean(means)-truth
    rmse = np.sqrt(np.mean((means-truth)**2))
    cov  = np.mean((lcis<=truth)&(ucis>=truth))
    ese  = np.std(means, ddof=1)
    ses  = np.sqrt(variances); mean_se = np.mean(ses)
    vr   = (mean_se/ese) if ese>1e-9 else np.nan
    return {"Mean_Est":np.mean(means),"Bias":bias,"Bias_MCSE":ese/np.sqrt(n),
            "RMSE":rmse,"Cov":cov,"Cov_MCSE":np.sqrt(cov*(1-cov)/n),
            "CI_Width":np.mean(ucis-lcis),"VR":vr,
            "Time_Avg": np.mean(times) if len(times)>0 else np.nan,
            "N_valid":n}


def collect_results(pc_df, true_eff_src, is_nonzero=False):
    metrics, ates = [], []
    for i, sc in pc_df.iterrows():
        st = sc.get("type","linear"); cm = sc["meanX_AC"]; cn = int(sc["N_AC"])
        prefix = "NONZERO_" if is_nonzero else ""
        fid = f"{prefix}N_AC{cn}meanX_AC{cm}_{st}"
        truth = (true_eff_src.get(st,0.0) if isinstance(true_eff_src, dict)
                 else float(true_eff_src))
        for mname, folder in methods_map.items():
            bp = f"Results/{folder.upper()}"
            mp = f"{bp}/means_{fid}.pkl"
            if not os.path.exists(mp): continue
            try:
                rm = np.array(pickle.load(open(mp,"rb")))
                rv = np.array(pickle.load(open(f"{bp}/variances_{fid}.pkl","rb")))
                rl = np.array(pickle.load(open(f"{bp}/lcis_{fid}.pkl","rb")))
                ru = np.array(pickle.load(open(f"{bp}/ucis_{fid}.pkl","rb")))
                tm = (np.array(pickle.load(open(f"{bp}/times_{fid}.pkl","rb")))
                      if os.path.exists(f"{bp}/times_{fid}.pkl") else np.array([]))
                m = process_metrics(rm, rv, rl, ru, tm, truth)
                if not m: continue
                m.update({"Method":mname,"N_AC":cn,"meanX_AC":cm,
                          "type":st,"is_nonzero":is_nonzero})
                metrics.append(m)
                vm = rm[~np.isnan(rm)]
                if len(vm)>0:
                    ates.append(pd.DataFrame({"ATE":vm,"Method":mname,
                                              "N_AC":cn,"meanX_AC":cm,
                                              "type":st,"is_nonzero":is_nonzero}))
            except Exception as e:
                print(f"  err {fid} {mname}: {e}")
    return metrics, ates


def collect_diagnostics():
    rows_overlap = []; rows_sl = []
    if not os.path.exists("Results/DIAGNOSTICS"):
        return pd.DataFrame(), pd.DataFrame()
    for f in os.listdir("Results/DIAGNOSTICS"):
        if not f.startswith("diag_"): continue
        fid = f.replace("diag_","").replace(".pkl","")
        try:
            with open(f"Results/DIAGNOSTICS/{f}","rb") as ff: d = pickle.load(ff)
        except Exception: continue
        ov = pd.DataFrame(d["overlap"])
        if len(ov)>0:
            row = {"fid":fid}
            for c in ["ESS_pct","max_w_ratio","PS_KS","PS_AC_mean","PS_AC_pct95",
                      "g1_trunc_pct","dr_trunc_pct"]:
                if c in ov.columns:
                    row[f"{c}_mean"]   = float(np.nanmean(ov[c]))
                    row[f"{c}_median"] = float(np.nanmedian(ov[c]))
            ess = pd.DataFrame(d["maic_ess"], columns=["ess_main","ess_pct"])
            if len(ess)>0:
                row["MAIC_ESS_pct_mean"] = float(np.nanmean(ess["ess_pct"]))
            rows_overlap.append(row)
        for sd in d["sl_diag"]:
            if sd is None: continue
            for nm,w in sd["weights"].items():
                rows_sl.append({"fid":fid, "learner":nm,
                                "weight":w,
                                "cv_risk": sd["cv_risks"].get(nm, np.nan),
                                "dominant": sd["dominant"]})
    return pd.DataFrame(rows_overlap), pd.DataFrame(rows_sl)


def collect_robustness(pc_df):
    rows = []
    for i, sc in pc_df.iterrows():
        st = sc.get("type","linear"); cm = sc["meanX_AC"]; cn = int(sc["N_AC"])
        if cn != 400 or cm != 0.3: continue
        fid = f"N_AC{cn}meanX_AC{cm}_{st}"
        for mname, folder in [("ac_proxy","GCOMP_SL"),
                              ("independent","GCOMP_SL_INDEP")]:
            bp = f"Results/{folder.upper()}"
            mp = f"{bp}/means_{fid}.pkl"
            if not os.path.exists(mp): continue
            try:
                rm = np.array(pickle.load(open(mp,"rb")))
                rv = np.array(pickle.load(open(f"{bp}/variances_{fid}.pkl","rb")))
                rl = np.array(pickle.load(open(f"{bp}/lcis_{fid}.pkl","rb")))
                ru = np.array(pickle.load(open(f"{bp}/ucis_{fid}.pkl","rb")))
                m = process_metrics(rm, rv, rl, ru, np.array([]), 0.0)
                if not m: continue
                m.update({"cor_method":mname,"N_AC":cn,"meanX_AC":cm,"type":st})
                rows.append(m)
            except Exception as e:
                print(f"  robustness err {fid} {mname}: {e}")
    return pd.DataFrame(rows)


def collect_boot1000():
    rows = []
    if not os.path.exists("Results/BOOT1000"):
        return pd.DataFrame()
    for f in os.listdir("Results/BOOT1000"):
        if not f.startswith("boot1000_"): continue
        fid = f.replace("boot1000_","").replace(".pkl","")
        try:
            with open(f"Results/BOOT1000/{f}","rb") as ff: out = pickle.load(ff)
        except Exception: continue
        for m in METHODS:
            ms, vs, ls, us = [], [], [], []
            for r in out:
                if r is None: continue
                v = r.get(m, (np.nan,)*5)
                ms.append(v[0]); vs.append(v[1]); ls.append(v[2]); us.append(v[3])
            if not ms: continue
            ms = np.array(ms); vs = np.array(vs); ls = np.array(ls); us = np.array(us)
            mt = process_metrics(ms, vs, ls, us, np.array([]), 0.0)
            if not mt: continue
            mt.update({"fid":fid,"method":m,"boot":1000})
            rows.append(mt)
    return pd.DataFrame(rows)


def plot_scenario(out_dir, ate_all, metrics_df, cn, cm, ct, is_nz, truth):
    sub = ate_all[(ate_all["N_AC"]==cn)&(ate_all["meanX_AC"]==cm)&
                  (ate_all["type"]==ct)&(ate_all["is_nonzero"]==is_nz)]
    if sub.empty: return
    sub = sub.copy()
    sm = metrics_df[(metrics_df["N_AC"]==cn)&(metrics_df["meanX_AC"]==cm)&
                    (metrics_df["type"]==ct)&(metrics_df["is_nonzero"]==is_nz)
                    ].set_index("Method")
    avail = [m for m in methods_map.keys() if m in sm.index]
    avail_labels = [plot_method_labels.get(m, m) for m in avail]
    sub["Method_label"] = sub["Method"].map(plot_method_labels).fillna(sub["Method"])
    cols = ["Method","Bias","RMSE","Cov","VR","CI_Width","Time(s)"]
    tdat = []
    for m in avail:
        r = sm.loc[m]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        tdat.append([plot_method_labels.get(m, m),
                     f"{r['Bias']:.3f}", f"{r['RMSE']:.3f}", f"{r['Cov']:.3f}",
                     f"{r['VR']:.2f}" if not np.isnan(r['VR']) else "-",
                     f"{r['CI_Width']:.2f}",
                     f"{r['Time_Avg']:.1f}" if not np.isnan(r['Time_Avg']) else "-"])
    fig = plt.figure(figsize=(15,11))
    gs = fig.add_gridspec(2,1,height_ratios=[3,1.2])
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    sns.violinplot(data=sub, y="Method_label", x="ATE", order=avail_labels, ax=ax1,
                   palette="Set3", inner="quartile", cut=0, orient="h")
    ax1.axvline(truth, color="#E74C3C", linestyle="--", linewidth=2.5,
                label=f"True effect ({truth:.3f})")
    pf = "[Sensitivity] " if is_nz else ""
    ax1.set_title(f"{pf}Type={ct} | N_AC={cn} | Overlap Mean={cm}",
                  fontsize=14, weight="bold")
    ax1.set_xlabel("Estimated indirect-comparison log-OR (A vs B)"); ax1.legend(fontsize=11)
    ax2.axis("off")
    tbl = ax2.table(cellText=tdat, colLabels=cols, loc="center",
                    cellLoc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (rr,cc),cell in tbl.get_celld().items():
        if rr==0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(weight="bold", color="white")
            cell.set_height(0.3)
        else:
            cell.set_height(0.18)
            cell.set_facecolor("#f5f5f5" if rr%2==0 else "white")
    tag = "NONZERO_" if is_nz else ""
    name = f"{out_dir}/Plot_{tag}{ct}_N{cn}_M{cm}.png"
    plt.tight_layout(); plt.savefig(name, dpi=300); plt.close()
    print(f"  [plot] {name}")


def write_response_summary(out_dir):
    md = """# Reproducibility Output Checklist

| Output | File / column | Purpose |
|--------|---------------|---------|
| Super Learner TMLE | `final_metrics.csv` row `TMLE (SL+CF)` | Targeted-learning sensitivity analysis |
| Influence-curve variance diagnostics | TMLE columns including `VR` | Interval-calibration check |
| MAIC effective sample size | `overlap_diagnostics.csv` column `MAIC_ESS_pct_mean` | Weighting and overlap diagnostic |
| Propensity-score separation | `overlap_diagnostics.csv` columns `PS_KS_*` | Covariate-support diagnostic |
| Truncation indicators | `overlap_diagnostics.csv` columns `g1_trunc_pct_*`, `dr_trunc_pct_*` | Extrapolation diagnostic |
| Super Learner weights and CV risk | `sl_learner_weights.csv` | Learner-library transparency |
| Covariance sensitivity | `robustness_comparison.csv` | Target-population reconstruction check |
| Bootstrap sensitivity | `boot1000_sensitivity.csv` | Sensitivity to number of bootstrap resamples |
| Nonzero-effect simulations | `Plot_NONZERO_*.png` and `final_metrics.csv` | Sensitivity to the true A versus B effect |
| Split-averaged SL G-computation | `final_metrics.csv` row `G-comp (SL, SA)` | Finite-sample sensitivity implementation |
"""
    with open(f"{out_dir}/REVIEWER_RESPONSE_SUMMARY.md","w",encoding="utf-8") as f:
        f.write(md)
    print(f"  [summary] {out_dir}/REVIEWER_RESPONSE_SUMMARY.md")


def stage_analyze():
    print("="*70); print("STAGE 3 - analysis and plotting"); print("="*70)
    out_dir = "Final_Results_Analysis_ITC"
    os.makedirs(out_dir, exist_ok=True)
    with open("binary_settings.pkl","rb") as f: settings = pickle.load(f)
    pc_loc = settings["pc"]; pc_nz = settings.get("pc_nonzero", pd.DataFrame())
    true_itc_nz = settings.get("true_itc_nonzero", {})

    print(">>> Primary metrics")
    m1, a1 = collect_results(pc_loc, true_eff_src=0.0, is_nonzero=False)
    m2, a2 = collect_results(pc_nz,  true_eff_src=true_itc_nz, is_nonzero=True)
    metrics_df = pd.DataFrame(m1+m2)
    if metrics_df.empty:
        print("  [warn] no metric data"); return
    metrics_df.to_csv(f"{out_dir}/final_metrics.csv", index=False)
    print(f"  saved final_metrics.csv  ({len(metrics_df)} rows)")

    ate_all = pd.concat(a1+a2, ignore_index=True) if (a1+a2) else pd.DataFrame()

    print(">>> Diagnostics")
    df_ov, df_sl = collect_diagnostics()
    if not df_ov.empty:
        df_ov.to_csv(f"{out_dir}/overlap_diagnostics.csv", index=False)
    if not df_sl.empty:
        sl_summary = df_sl.groupby(["fid","learner"]).agg(
            mean_weight=("weight","mean"),
            sd_weight=("weight","std"),
            mean_cv_risk=("cv_risk","mean")).reset_index()
        sl_summary.to_csv(f"{out_dir}/sl_learner_weights.csv", index=False)

    print(">>> Covariance sensitivity")
    df_rob = collect_robustness(pc_loc)
    if not df_rob.empty:
        df_rob.to_csv(f"{out_dir}/robustness_comparison.csv", index=False)

    print(">>> Boot1000")
    df_b1k = collect_boot1000()
    if not df_b1k.empty:
        df_b1k.to_csv(f"{out_dir}/boot1000_sensitivity.csv", index=False)

    print(">>> Plotting")
    sns.set_theme(style="whitegrid", font_scale=1.05)
    if not ate_all.empty:
        for _, s in pc_loc.iterrows():
            plot_scenario(out_dir, ate_all, metrics_df,
                          int(s["N_AC"]), s["meanX_AC"],
                          s.get("type","linear"), False, 0.0)
        for _, s in pc_nz.iterrows():
            t = s.get("type","linear")
            plot_scenario(out_dir, ate_all, metrics_df,
                          int(s["N_AC"]), s["meanX_AC"], t, True,
                          true_itc_nz.get(t, 0.0))

    write_response_summary(out_dir)
    print(f"\n>>> Complete. Results in {out_dir}/")


# =============================================================================
# 11. Main - simulation, checks, and analysis
# =============================================================================
if __name__ == "__main__":
    write_package_versions()
    print("\n" + "#"*70)
    if ANALYSIS_ONLY:
        print(f"#  ITC SIMULATION - ANALYSIS ONLY MODE / N_sim={N_sim}")
    else:
        print(f"#  ITC SIMULATION - SINGLE MACHINE / N_sim={N_sim}")
    print(f"#  N_AC_LIST    = {N_AC_LIST}")
    print(f"#  meanX_AC     = {meanX_AC_LIST}")
    print(f"#  pc       size= {len(pc)} (primary)")
    print(f"#  pc_nonzero size= {len(pc_nonzero)} (full 3x3x2 grid)")
    print("#"*70 + "\n")

    if ANALYSIS_ONLY:
        ok, missing = verify_all_complete(check_full_grid=True)
        if not ok:
            print("\n" + "!"*70)
            print(f"!! The grid still has {len(missing)} incomplete scenario-method combinations:")
            for fid, m, cur, need in missing[:50]:
                print(f"    - {fid:50s}  {m:15s}  {cur}/{need}")
            if len(missing) > 50:
                print(f"    ... (additional {len(missing)-50} )")
            print("!"*70 + "\n")
            sys.exit(1)
        print("\n>>> All scenarios complete; starting analysis.\n")
        stage_analyze()
    else:
        stage_generate_data()
        stage_run_simulation()

        # : grid
        ok, missing = verify_all_complete(check_full_grid=True)
        if not ok:
            print("\n" + "!"*70)
            print(f"!!  {len(missing)} incomplete scenario-method combinations:")
            for fid, m, cur, need in missing[:50]:
                print(f"    - {fid:50s}  {m:15s}  {cur}/{need}")
            if len(missing) > 50:
                print(f"    ... (additional {len(missing)-50} )")
            print("!! Rerun this script to resume incomplete work.")
            print("!"*70 + "\n")
            sys.exit(1)

        print("\n" + "#"*70)
        print("#  Simulation complete; starting final analysis.")
        print("#"*70 + "\n")
        stage_analyze()

        print("\n" + "#"*70)
        print("#  All tasks complete. Results in Final_Results_Analysis_ITC/")
        print("#"*70 + "\n")
