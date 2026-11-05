"""Phase 2 - Machine Learning family: Random Forest, XGBoost, LightGBM, SVR.

All four consume the flattened window (672 x 11 = 7392 features) and predict
the full 96 x 4 output block (in scaled space; the caller inverse-scales).

* RandomForestRegressor : native multi-output
* XGBRegressor          : one-output-per-tree multi-output strategy
* LGBMRegressor         : wrapped by MultiOutputRegressor (parallel over outputs)
* SVR (RBF)             : PCA on features (64 comps) + MultiOutputRegressor,
                          parallelised over the 384 outputs
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

logger = logging.getLogger(__name__)


def _timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def fit_predict_rf(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray
                   ) -> tuple[np.ndarray, int, float]:
    # max_features=0.2 + depth cap: essential with 384 outputs (split
    # criterion cost scales with n_outputs); negligible metric impact
    model = RandomForestRegressor(n_estimators=120, max_features=0.2,
                                  max_depth=24, min_samples_leaf=2,
                                  n_jobs=-1, random_state=42)
    (model, dt) = _timeit(model.fit, Xtr, Ytr)
    pred, _ = _timeit(model.predict, Xte)
    n_params = int(sum(est.tree_.node_count for est in model.estimators_))
    return pred.astype("float32"), n_params, dt


def fit_predict_xgb(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray
                    ) -> tuple[np.ndarray, int, float]:
    """XGBRegressor per output (384 tiny single-output fits, parallelised).

    xgboost's multi_output_tree needs histograms scaling with n_outputs
    (hundreds of MB) and fails on RAM-tight machines; single-output fits
    need only a few MB each.
    """
    from xgboost import XGBRegressor
    base = XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=8,
                        subsample=0.9, colsample_bytree=0.5, tree_method="hist",
                        n_jobs=2, random_state=42)
    model = MultiOutputRegressor(base, n_jobs=4)
    (model, dt) = _timeit(model.fit, Xtr, Ytr)
    pred, _ = _timeit(model.predict, Xte)
    n_params = None
    try:
        n_params = 0
        for est in model.estimators_:
            tdf = est.get_booster().trees_to_dataframe()
            n_params += int(len(tdf[tdf["Feature"] != "Leaf"]))
    except Exception:  # pragma: no cover
        n_params = None
    return pred.astype("float32"), n_params, dt


def fit_predict_lgbm(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray
                     ) -> tuple[np.ndarray, int, float]:
    from lightgbm import LGBMRegressor
    base = LGBMRegressor(n_estimators=250, learning_rate=0.08, num_leaves=63,
                         subsample=0.9, colsample_bytree=0.8, n_jobs=2,
                         random_state=42, verbose=-1)
    model = MultiOutputRegressor(base, n_jobs=4)
    (model, dt) = _timeit(model.fit, Xtr, Ytr)
    pred, _ = _timeit(model.predict, Xte)
    n_params = int(sum(est.booster_.num_trees() * est.booster_.num_model_per_iteration()
                       for est in model.estimators_))
    return pred.astype("float32"), n_params, dt


def fit_predict_svr(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
                    n_components: int = 64) -> tuple[np.ndarray, int, float]:
    """SVR (RBF) on PCA-reduced features; outputs handled in parallel."""
    pca = PCA(n_components=n_components, random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)
    model = MultiOutputRegressor(
        SVR(kernel="rbf", C=10.0, epsilon=0.01, cache_size=500), n_jobs=-1)
    (model, dt) = _timeit(model.fit, Ztr, Ytr)
    pred, _ = _timeit(model.predict, Zte)
    # SVR "parameters": support vectors x (1 + n_features weights)
    n_sv = sum(len(est.support_) for est in model.estimators_)
    n_params = int(n_sv * (n_components + 1))
    return pred.astype("float32"), n_params, dt


MODELS = {
    "RandomForest": fit_predict_rf,
    "XGBoost": fit_predict_xgb,
    "LightGBM": fit_predict_lgbm,
    "SVR": fit_predict_svr,
}
