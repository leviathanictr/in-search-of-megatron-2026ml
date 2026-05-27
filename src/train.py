"""
Тренировка v12: PSD-разметка + двухраундовый ансамбль (RF + GB + CatBoost + SVM).

Раунд 1 (rules-based seed labels):
  - gamma: PSD < 0.13, амплитуда > 400, пик в [95, 115]
  - neutron: PSD ∈ [0.24, 0.32], амплитуда > 400, пик в [95, 115]
  - noise: FLAGS != 0x4000, насыщение, шум >80, пик вне [60, 350], много пиков, низкая амплитуда

  Тренируется RF + GB + CatBoost, soft-voting усреднение.

Раунд 2 (self-training):
  - События с уверенностью soft-voting > 0.85 становятся новой обучающей выборкой.
  - Переобучаются RF + GB + CatBoost + добавляется SVM (RBF, балансированная подвыборка ~7000 на класс).
  - Финальное предсказание: взвешенное soft-voting (1·RF + 1·GB + 3·CatBoost + 3·SVM) / 8.

На выходе: verdict_v12_cat3_svm3.csv в формате (Id, Predicted).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier


BASE_FEATS = [
    "energy", "energyshort", "amp", "area", "log_amp", "log_area", "peak_argmin",
    "psd_caen", "psd_local", "a_over_s",
    "short_q", "mid_q", "tail_q", "tail_far_q",
    "frac_short", "frac_mid", "frac_tail", "frac_tail_far",
    "rise10_90", "fall90_10", "width50", "width20",
    "asymmetry", "baseline_noise", "post_peak_noise", "n_peaks",
    "energy_skew", "energy_kurt",
]


def build_seed_labels(ff: pd.DataFrame) -> np.ndarray:
    N = len(ff)
    psd = ff["psd_caen"].to_numpy()
    amp = ff["amp"].to_numpy()
    peakp = ff["peak_argmin"].to_numpy()
    flag = ff["flags"].to_numpy()
    rise = ff["rise10_90"].to_numpy()
    basen = ff["baseline_noise"].to_numpy()
    satv = ff["sat"].to_numpy()
    npks = ff["n_peaks"].to_numpy()

    anom = ((flag != "0x4000")
            | (satv == 1)
            | (basen > 80)
            | (peakp > 350) | (peakp < 60)
            | ((peakp > 200) & (rise > 20))
            | (npks > 6)
            | (amp < 50))
    clean_peak = (peakp >= 95) & (peakp <= 115)
    gam = ~anom & clean_peak & (psd < 0.13) & (amp > 400)
    neu = ~anom & clean_peak & (psd >= 0.24) & (psd <= 0.32) & (amp > 400)

    y = np.full(N, -1, dtype=np.int8)
    y[neu] = 0
    y[gam] = 1
    y[anom] = 2
    return y


def make_round1_models(class_weights: dict[int, float] | None = None):
    rf = Pipeline([
        ("sc", StandardScaler()),
        ("m", RandomForestClassifier(
            n_estimators=700, max_features="sqrt", n_jobs=-1, random_state=0,
            min_samples_leaf=2, class_weight="balanced",
        )),
    ])
    gb = Pipeline([
        ("sc", StandardScaler()),
        ("m", GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            random_state=0, subsample=0.8,
        )),
    ])
    cat = CatBoostClassifier(
        iterations=600, learning_rate=0.04, depth=5, loss_function="MultiClass",
        random_seed=0, verbose=0,
        class_weights=class_weights or {0: 1.0, 1: 5.0, 2: 4.0},
    )
    return rf, gb, cat


def make_round2_models():
    rf = Pipeline([
        ("sc", StandardScaler()),
        ("m", RandomForestClassifier(
            n_estimators=700, max_features="sqrt", n_jobs=-1, random_state=1,
            min_samples_leaf=2, class_weight="balanced",
        )),
    ])
    gb = Pipeline([
        ("sc", StandardScaler()),
        ("m", GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            random_state=1, subsample=0.8,
        )),
    ])
    cat = CatBoostClassifier(
        iterations=600, learning_rate=0.04, depth=5, loss_function="MultiClass",
        random_seed=1, verbose=0,
    )
    svm = Pipeline([
        ("sc", StandardScaler()),
        ("m", SVC(
            kernel="rbf", C=8, gamma="scale", probability=True, random_state=1,
            class_weight="balanced", cache_size=1000,
        )),
    ])
    return rf, gb, cat, svm


def balanced_subsample(y: np.ndarray, per_class: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = min(per_class, len(idx))
        parts.append(rng.choice(idx, n, replace=False))
    return np.concatenate(parts)


def main(artifacts_dir: Path, weights: tuple[int, int, int, int] = (1, 1, 3, 3)) -> Path:
    full_path = artifacts_dir / "features_full.parquet"
    extra_path = artifacts_dir / "features_extra.parquet"
    if not (full_path.exists() and extra_path.exists()):
        sys.exit("missing feature parquet files. Run src/features.py first.")

    print("loading features ...", flush=True)
    full = pd.read_parquet(full_path)
    extra = pd.read_parquet(extra_path)
    ff = full.merge(extra, on="idx")
    N = len(ff)

    pca_cols = [c for c in extra.columns if c.startswith("pca_")]
    win_cols = [c for c in extra.columns
                if c.startswith("win_") or c.startswith("winf_")
                or c in ["frac_late80", "frac_late200", "post_centroid"]]
    feat_cols = BASE_FEATS + pca_cols + win_cols
    X_all = ff[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    print(f"  N={N}, feature_count={X_all.shape[1]}", flush=True)

    y0 = build_seed_labels(ff)
    n0 = int((y0 == 0).sum())
    n1 = int((y0 == 1).sum())
    n2 = int((y0 == 2).sum())
    print(f"seed labels: neutron={n0}, gamma={n1}, noise={n2}", flush=True)
    mask = y0 >= 0
    X_tr = X_all[mask]
    y_tr = y0[mask]

    rf1, gb1, cat1 = make_round1_models()
    t0 = time.time()
    print("round 1: training RF + GB + CatBoost ...", flush=True)
    rf1.fit(X_tr, y_tr); print(f"  rf   {time.time()-t0:.0f}s", flush=True)
    gb1.fit(X_tr, y_tr); print(f"  gb   {time.time()-t0:.0f}s", flush=True)
    cat1.fit(X_tr, y_tr); print(f"  cat  {time.time()-t0:.0f}s", flush=True)

    p_rf = rf1.predict_proba(X_all)
    p_gb = gb1.predict_proba(X_all)
    p_cat = cat1.predict_proba(X_all)
    ens1 = (p_rf + p_gb + p_cat) / 3.0
    pred1 = ens1.argmax(axis=1).astype(np.int8)
    conf1 = ens1.max(axis=1)
    high = conf1 > 0.85
    y2 = np.where(high, pred1, -1).astype(np.int8)
    print(f"round 2 seeds (conf>0.85): {int((y2>=0).sum())}", flush=True)

    X_tr2 = X_all[y2 >= 0]
    y_tr2 = y2[y2 >= 0]

    rf2, gb2, cat2, svm2 = make_round2_models()
    t0 = time.time()
    print("round 2: training RF + GB + CatBoost ...", flush=True)
    rf2.fit(X_tr2, y_tr2); print(f"  rf2  {time.time()-t0:.0f}s", flush=True)
    gb2.fit(X_tr2, y_tr2); print(f"  gb2  {time.time()-t0:.0f}s", flush=True)
    cat2.fit(X_tr2, y_tr2); print(f"  cat2 {time.time()-t0:.0f}s", flush=True)
    sub = balanced_subsample(y_tr2, 7000)
    svm2.fit(X_tr2[sub], y_tr2[sub])
    print(f"  svm  {time.time()-t0:.0f}s (subsample={len(sub)})", flush=True)

    q_rf = rf2.predict_proba(X_all)
    q_gb = gb2.predict_proba(X_all)
    q_cat = cat2.predict_proba(X_all)
    q_svm = svm2.predict_proba(X_all)

    w_rf, w_gb, w_cat, w_svm = weights
    ens = (w_rf * q_rf + w_gb * q_gb + w_cat * q_cat + w_svm * q_svm) / sum(weights)
    pred = ens.argmax(axis=1).astype(np.int8)

    out_path = artifacts_dir / "verdict_v12_cat3_svm3.csv"
    pd.DataFrame({"Id": np.arange(N), "Predicted": pred}).to_csv(out_path, index=False)
    print(f"saved {out_path}", flush=True)
    print(f"distribution: neutron={int((pred==0).sum())}, gamma={int((pred==1).sum())}, noise={int((pred==2).sum())}", flush=True)
    return out_path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    main(here / "artifacts")
