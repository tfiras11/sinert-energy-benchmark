"""Phase 2 - naive seasonal baselines (cahier de charges: persistance t-96 / t-672).

Both baselines are computed purely from the input window X (no target access):

* ``naive_96``  : the previous day repeats itself       -> last 96 steps of X
* ``naive_672`` : same weekday one week earlier          -> first 96 steps of X

For the energies the per-step differences of the cumulative counter channel
are used (delta over 97 counter points -> 96 per-step values).
"""

from __future__ import annotations

import numpy as np

N_TARGETS = 4


def _from_counter_slice(win: np.ndarray) -> np.ndarray:
    """win: (n, L+1, 11) counter slice -> (n, L, 2) per-step dEa/dEr."""
    return np.diff(win[:, :, 0:2], axis=1)


def naive_predictions(X: np.ndarray, sample_time, masks: dict[str, np.ndarray]
                      ) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{"naive_96": {"val": ..., "test": ...}, "naive_672": ...}``.

    Predictions are in physical units (kWh, kvarh, kW, kVA).
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, c_slice, p_slice in (("naive_96", slice(-97, None), slice(-96, None)),
                                   ("naive_672", slice(0, 97), slice(0, 96))):
        d = _from_counter_slice(X[:, c_slice, :])       # (n, 96, 2)
        pred = np.concatenate([d, X[:, p_slice, 2:4]], axis=2).astype("float32")
        out[name] = {m: pred[mask] for m, mask in
                     (("val", masks["val"]), ("test", masks["test"]))}
    return out
