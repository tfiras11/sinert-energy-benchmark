"""PHASE 2 - Colab benchmark: ALL models that don't run on the local machine.

Run this single file on Google Colab. It trains:
  * Machine Learning   : XGBoost, LightGBM, SVR      (sklearn/xgb/lgbm)
  * Statistical        : SARIMAX                      (statsmodels)
  * Deep Learning (7) : LSTM, Bi-LSTM, GRU, CNN-1D, TCN/ResNet-1D,
                         Transformer, CNN-LSTM       (torch)

Everything consumes ONLY `windows_scaled.npz` (same protocol as the local
benchmark: X = 672x11 scaled week, Y = 96x4 scaled next day; scaler params
included so predictions come back in physical units).

HOW TO USE ON COLAB
-------------------
1. Upload `windows_scaled.npz` (sg_project/output/benchmark/) and this file.
2. Runtime > Change runtime type > GPU (only the DL part benefits; CPU works).
3. Selectively run (fast tests first):
       !python colab_benchmark.py --models ml                # ~5 min on CPU
       !python colab_benchmark.py --models sarimax           # ~5 min
       !python colab_benchmark.py --models dl --epochs 60    # ~20 min on GPU
       !python colab_benchmark.py                            # everything
4. Download `colab_results.csv` + `colab_predictions.npz` and drop them in
   sg_project/output/benchmark/, then locally:
       python -I merge_colab.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

TARGETS = ["dEa_kWh", "dEr_kvarh", "P_max_kW", "S_max_kVA"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_data(path: str):
    d = np.load(path, allow_pickle=False)
    X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
    split = d["split"].astype(str)
    masks = {k: split == k for k in ("train", "val", "test")}
    st = pd.DatetimeIndex(d["sample_time"])
    return {"X": X, "Y": Y, "masks": masks, "sample_time": st,
            "y_mean": d["y_mean"].astype(np.float32),
            "y_std": d["y_std"].astype(np.float32),
            "y_mean64": d["y_mean"].astype(np.float64),
            "y_std64": d["y_std"].astype(np.float64)}


def inverse(Ys, mean, std):
    return Ys * std + mean


# --------------------------------------------------------------------------- #
# metrics (identical to phase2/benchmark.py)
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred) -> dict:
    out = {}
    for k, tgt in enumerate(TARGETS):
        t, p = y_true[:, :, k].ravel(), y_pred[:, :, k].ravel()
        err = p - t
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        out[f"MAE_{tgt}"] = float(np.mean(np.abs(err)))
        out[f"RMSE_{tgt}"] = float(np.sqrt(ss_res / len(t)))
        out[f"R2_{tgt}"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if tgt.startswith("d"):
            d_t, d_p = y_true[:, :, k].sum(1), y_pred[:, :, k].sum(1)
            msk = d_t > 1e-6
            out[f"MAPE_{tgt}_daily"] = float(
                np.mean(np.abs((d_t[msk] - d_p[msk]) / d_t[msk])) * 100)
        else:
            pk_t, pk_p = y_true[:, :, k].max(1), y_pred[:, :, k].max(1)
            out[f"MAE_{tgt}_dailypeak"] = float(np.mean(np.abs(pk_p - pk_t)))
    for k, tgt in ((2, "P"), (3, "S")):
        pk_t = y_true[:, :, k].max(1)
        pk_p = y_pred[:, :, k].max(1)
        out[f"PDE_{tgt}max"] = float(
            np.mean(np.abs(pk_p - pk_t) / np.maximum(pk_t, 1e-6)) * 100)
    return out


class Collector:
    def __init__(self):
        self.rows, self.preds = [], {}

    def add(self, name, family, y_true, pred_val, pred_test, n_params, dt):
        self.preds[f"{name}__val"], self.preds[f"{name}__test"] = pred_val, pred_test
        met = evaluate(y_true["test"], pred_test)
        self.rows.append({"model": name, "family": family, "n_params": n_params,
                          "train_time_s": round(dt, 1), **met})
        print(f"  [{name}] test RMSE_dEa={met['RMSE_dEa_kWh']:.2f}  "
              f"MAPE_dEa={met['MAPE_dEa_kWh_daily']:.2f}%  "
              f"PDE_P={met['PDE_Pmax']:.2f}%")

    def save(self):
        """Save cumulatively: merge with any previous partial run so results
        accumulate across separate sessions (ml / sarimax / dl runs)."""
        import os
        old_rows, old_preds = [], {}
        if os.path.exists("colab_results.csv"):
            old_rows = pd.read_csv("colab_results.csv").to_dict("records")
        if os.path.exists("colab_predictions.npz"):
            old_preds = dict(np.load("colab_predictions.npz"))
        new_names = {r["model"] for r in self.rows}
        rows = [r for r in old_rows if r["model"] not in new_names] + self.rows
        preds = {k: v for k, v in old_preds.items()
                 if k.split("__")[0] not in new_names}
        preds.update(self.preds)
        pd.DataFrame(rows).to_csv("colab_results.csv", index=False)
        np.savez_compressed("colab_predictions.npz", **preds)
        print(f"saved: colab_results.csv ({len(rows)} models), "
              f"colab_predictions.npz ({len(preds)} arrays)")


# --------------------------------------------------------------------------- #
# machine-learning family
# --------------------------------------------------------------------------- #
def run_ml(data, col: Collector):
    from sklearn.decomposition import PCA
    from sklearn.multioutput import MultiOutputRegressor

    Xtr = data["X"][data["masks"]["train"]].reshape(-1, 672 * 11)
    Ytr = data["Y"][data["masks"]["train"]].reshape(len(Xtr), -1)
    Xte = data["X"][data["masks"]["val"] | data["masks"]["test"]].reshape(-1, 672 * 11)
    n_val = int(data["masks"]["val"].sum())
    y_true = {m: inverse(data["Y"][data["masks"][m]], data["y_mean"], data["y_std"])
              for m in ("val", "test")}

    def split_out(p):
        return (inverse(p[:n_val], data["y_mean"], data["y_std"]),
                inverse(p[n_val:], data["y_mean"], data["y_std"]))

    # ---- XGBoost: one-output-per-tree per target (fast, in-process) --------
    # multi_output_tree is experimentally slow here (split cost scales with
    # n_outputs x n_features x n_bins -> minutes per node level on 2 vCPUs).
    # The default strategy builds 96 tiny trees per boosting round: each
    # single-output histogram is only a few MB and fits in milliseconds.
    print("=== XGBoost ===")
    from xgboost import XGBRegressor
    t0 = time.time()
    preds, n_params = [], 0
    for k in range(4):
        m = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=8,
                         subsample=0.9, colsample_bytree=0.5, max_bin=256,
                         tree_method="hist", n_jobs=2, random_state=42)
        m.fit(Xtr, Ytr[:, k * 96:(k + 1) * 96])   # 96 outputs -> 96 trees/round
        preds.append(m.predict(Xte))
        try:
            tdf = m.get_booster().trees_to_dataframe()
            n_params += int(len(tdf[tdf["Feature"] != "Leaf"]))
        except Exception:
            pass
        print(f"  XGB target {k + 1}/4 done ({time.time() - t0:.0f}s cumulated)")
    dt = time.time() - t0
    pv, pt = split_out(np.concatenate(preds, axis=1).astype(np.float32))
    col.add("XGBoost", "Machine Learning", y_true, pv, pt, n_params, dt)

    # ---- LightGBM: sequential per-target wrapper (n_jobs=1 -> in-process) --
    print("=== LightGBM ===")
    from lightgbm import LGBMRegressor
    t0 = time.time()
    preds, n_params = [], 0
    for k in range(4):
        wrap = MultiOutputRegressor(LGBMRegressor(
            n_estimators=300, learning_rate=0.08, num_leaves=63,
            subsample=0.9, colsample_bytree=0.8, n_jobs=2,
            random_state=42, verbose=-1), n_jobs=1)
        wrap.fit(Xtr, Ytr[:, k * 96:(k + 1) * 96])
        preds.append(wrap.predict(Xte))
        n_params += int(sum(b.booster_.num_trees() * b.booster_.num_model_per_iteration()
                            for b in wrap.estimators_))
        print(f"  LGBM target {k + 1}/4 done ({time.time() - t0:.0f}s cumulated)")
    dt = time.time() - t0
    pv, pt = split_out(np.concatenate(preds, axis=1).astype(np.float32))
    col.add("LightGBM", "Machine Learning", y_true, pv, pt, n_params, dt)

    # ---- SVR: PCA features + sequential wrapper ----------------------------
    print("=== SVR ===")
    from sklearn.svm import SVR
    t0 = time.time()
    pca = PCA(n_components=64, random_state=42)
    Ztr, Zte = pca.fit_transform(Xtr), pca.transform(Xte)
    preds, n_sv = [], 0
    for k in range(4):
        wrap = MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.01,
                                        cache_size=500), n_jobs=1)
        wrap.fit(Ztr, Ytr[:, k * 96:(k + 1) * 96])
        preds.append(wrap.predict(Zte))
        n_sv += sum(len(e.support_) for e in wrap.estimators_)
        print(f"  SVR target {k + 1}/4 done ({time.time() - t0:.0f}s cumulated)")
    dt = time.time() - t0
    pv, pt = split_out(np.concatenate(preds, axis=1).astype(np.float32))
    col.add("SVR", "Machine Learning", y_true, pv, pt, int(n_sv * 65), dt)


# --------------------------------------------------------------------------- #
# statistical family: SARIMAX (fits on the continuous train series
# reconstructed from the daily Y blocks; recursive multi-step forecast)
# --------------------------------------------------------------------------- #
def calendar_exog(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour + index.minute / 60.0
    dow = index.dayofweek
    doy = index.dayofyear
    ylen = np.where(index.is_leap_year, 366.0, 365.0)
    cols = []
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * h / 24), np.cos(2 * np.pi * k * h / 24)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * dow / 7), np.cos(2 * np.pi * k * dow / 7)]
    cols += [np.sin(2 * np.pi * doy / ylen), np.cos(2 * np.pi * doy / ylen),
             (dow >= 5).astype(float)]
    return np.stack(cols, axis=1)


def run_sarimax(data, col: Collector):
    """SARIMAX with WALK-FORWARD evaluation (cahier de charges: statistical
    family). One fit per target on the continuous train series (reconstructed
    from the daily Y blocks), then for every val/test day: forecast the next
    96 steps and append the ACTUALS to the Kalman state (refit=False).
    This conditions every prediction on all observed history and avoids the
    drift of a single recursive 6-month forecast."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    st, masks, Y = data["sample_time"], data["masks"], data["Y"]

    # continuous training series: train Y blocks tile the timeline day by day
    tr_days = st[masks["train"]]
    tr_steps = pd.date_range(tr_days.min(),
                             tr_days.max() + pd.Timedelta(minutes=95 * 15),
                             freq="15min")
    series = np.full((len(tr_steps), 4), np.nan)
    pos = {t: i for i, t in enumerate(tr_steps)}
    for d, yblock in zip(tr_days, Y[masks["train"]]):
        series[pos[d]: pos[d] + 96] = yblock
    assert not np.isnan(series).any(), "train series has gaps - unexpected"

    f_mask = masks["val"] | masks["test"]
    f_days = st[f_mask]
    day_split = np.where(masks["val"][f_mask], "val", "test")
    actuals = Y[f_mask]                       # (n_days, 96, 4) for the appends

    preds = {m: np.empty((int(masks[m].sum()), 96, 4), dtype=np.float32)
             for m in ("val", "test")}
    t0, n_params = time.time(), 0
    for k, tgt in enumerate(TARGETS):
        model = SARIMAX(series[:, k], exog=calendar_exog(tr_steps),
                        order=(2, 0, 2), enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=200)
        n_params += len(res.params)
        print(f"  SARIMAX {tgt}: fitted ({time.time()-t0:.0f}s cumulated)")

        ctr, t1 = {"val": 0, "test": 0}, time.time()
        for i, (d, sp) in enumerate(zip(f_days, day_split)):
            day_idx = pd.date_range(d, periods=96, freq="15min")
            fc = res.get_forecast(steps=96, exog=calendar_exog(day_idx))
            preds[sp][ctr[sp], :, k] = \
                np.asarray(fc.predicted_mean).ravel()[:96]
            ctr[sp] += 1
            # feed the actuals of day d back into the state filter
            res = res.append(actuals[i, :, k], exog=calendar_exog(day_idx),
                             refit=False)
            if (i + 1) % 60 == 0:
                print(f"    {tgt}: {i+1}/{len(f_days)} days "
                      f"({time.time()-t1:.0f}s)")
    y_true = {m: Y[masks[m]] for m in ("val", "test")}
    col.add("SARIMAX", "Statistical", y_true, preds["val"], preds["test"],
            n_params, time.time() - t0)


# --------------------------------------------------------------------------- #
# deep-learning family (PyTorch)
# --------------------------------------------------------------------------- #
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(),
                                 nn.Linear(128, 96 * 4))

    def forward(self, z):
        return self.net(z).view(-1, 96, 4)


class LSTMModel(nn.Module):
    def __init__(self, hidden=96, layers=2, bidirectional=False):
        super().__init__()
        self.rnn = nn.LSTM(11, hidden, layers, batch_first=True,
                           bidirectional=bidirectional, dropout=0.2)
        self.head = Head(hidden * (2 if bidirectional else 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class BiLSTMModel(LSTMModel):
    def __init__(self):
        super().__init__(hidden=64, layers=2, bidirectional=True)


class GRUModel(nn.Module):
    def __init__(self, hidden=96, layers=2):
        super().__init__()
        self.rnn = nn.GRU(11, hidden, layers, batch_first=True, dropout=0.2)
        self.head = Head(hidden)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class CNN1DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(11, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(24))
        self.head = Head(128 * 24)

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)).flatten(1))


class TCNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, 5, padding=4 * dilation, dilation=dilation),
            nn.BatchNorm1d(ch_out), nn.ReLU(),
            nn.Conv1d(ch_out, ch_out, 5, padding=4 * dilation, dilation=dilation),
            nn.BatchNorm1d(ch_out), nn.ReLU())
        self.skip = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x):
        return self.conv(x)[:, :, : x.size(2)] + self.skip(x)


class TCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, ch = [], 11
        for c, d in ((32, 1), (64, 2), (64, 4)):
            blocks.append(TCNBlock(ch, c, d))
            ch = c
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.head = Head(ch * 16)

    def forward(self, x):
        return self.head(self.pool(self.body(x.transpose(1, 2))).flatten(1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=672):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(11, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                         batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = Head(d_model)

    def forward(self, x):
        z = self.encoder(self.pos(self.proj(x)))
        return self.head(z.mean(dim=1))


class CNNLSTMModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(11, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 11, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(11, hidden, 1, batch_first=True)
        self.head = Head(hidden)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.head(out[:, -1])


DL_MODELS = {
    "LSTM": lambda: LSTMModel(),
    "Bi_LSTM": lambda: BiLSTMModel(),
    "GRU": lambda: GRUModel(),
    "CNN_1D": lambda: CNN1DModel(),
    "TCN_ResNet1D": lambda: TCNModel(),
    "Transformer": lambda: TransformerModel(),
    "CNN_LSTM": lambda: CNNLSTMModel(),
}


def run_dl(data, col: Collector, epochs: int, device: str):
    import copy
    Ys = (data["Y"] - data["y_mean64"]) / data["y_std64"]  # scaled targets
    Ys = Ys.astype(np.float32)
    for name, maker in DL_MODELS.items():
        print(f"=== {name} ===")
        torch.manual_seed(42)
        model = maker().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                           patience=3)
        lossf = nn.MSELoss()
        Xtr = torch.tensor(data["X"][data["masks"]["train"]])
        Ytr = torch.tensor(Ys[data["masks"]["train"]])
        Xva = torch.tensor(data["X"][data["masks"]["val"]])
        Yva = torch.tensor(Ys[data["masks"]["val"]])
        best_state, best_val, patience = None, float("inf"), 0
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = lossf(model(Xtr[idx].to(device)), Ytr[idx].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = lossf(model(Xva.to(device)), Yva.to(device)).item()
            sched.step(vl)
            if vl < best_val - 1e-5:
                best_val, patience = vl, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
            if ep % 10 == 0 or ep == 1:
                print(f"  epoch {ep:3d}  val MSE(scaled) = {vl:.4f}")
            if patience >= 7:
                print(f"  early stop at epoch {ep}")
                break
        dt = time.time() - t0
        if best_state is not None:
            model.load_state_dict(best_state)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        preds = {}
        model.eval()
        with torch.no_grad():
            for m in ("val", "test"):
                Xm = torch.tensor(data["X"][data["masks"][m]])
                out = [model(Xm[i:i + 256].to(device)).cpu().numpy()
                       for i in range(0, len(Xm), 256)]
                preds[m] = inverse(np.concatenate(out),
                                   data["y_mean"], data["y_std"])
        y_true = {m: data["Y"][data["masks"][m]] for m in ("val", "test")}
        col.add(name, "Deep Learning", y_true, preds["val"], preds["test"],
                n_params, dt)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="windows_scaled.npz")
    ap.add_argument("--models", default="ml,sarimax,dl",
                    help="comma list: ml, sarimax, dl")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    want = {w.strip() for w in args.models.split(",")}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    data = load_data(args.data)
    col = Collector()

    if "ml" in want:
        run_ml(data, col)
        col.save()
    if "sarimax" in want:
        run_sarimax(data, col)
        col.save()
    if "dl" in want:
        run_dl(data, col, args.epochs, device)
        col.save()


if __name__ == "__main__":
    main()
