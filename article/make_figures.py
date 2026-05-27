"""
Генерация графиков для статьи.

Требует:
  - artifacts/features_full.parquet
  - artifacts/features_extra.parquet
  - artifacts/adc_raw.npy (если есть; иначе пропускает sample-signals)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)


def load_features() -> pd.DataFrame:
    f = pd.read_parquet(ARTIFACTS / "features_full.parquet")
    e = pd.read_parquet(ARTIFACTS / "features_extra.parquet")
    return f.merge(e, on="idx")


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


def fig_psd_distribution(ff: pd.DataFrame) -> Path:
    psd = ff["psd_caen"].to_numpy()
    amp = ff["amp"].to_numpy()
    mask = (amp > 50) & (psd > -0.1) & (psd < 1.0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(psd[mask], bins=200, color="#4c72b0", alpha=0.85)
    ax.axvspan(-0.1, 0.13, color="#dd8452", alpha=0.15, label="Гамма-кванты (PSD < 0,13)")
    ax.axvspan(0.13, 0.24, color="#999999", alpha=0.15, label="Зона неопределённости")
    ax.axvspan(0.24, 0.32, color="#55a868", alpha=0.15, label="Нейтроны (0,24 ≤ PSD ≤ 0,32)")
    ax.axvline(0.13, color="#dd8452", linestyle="--", linewidth=1.0)
    ax.axvline(0.24, color="#55a868", linestyle="--", linewidth=1.0)
    ax.axvline(0.32, color="#55a868", linestyle="--", linewidth=1.0)
    ax.set_xlabel("PSD$_\\mathrm{CAEN}$")
    ax.set_ylabel("Число событий")
    ax.set_title("Распределение PSD по выборке (амплитуда > 50)")
    ax.set_xlim(-0.05, 0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    out = FIGURES / "fig1_psd_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_feature_importance(ff: pd.DataFrame, y: np.ndarray) -> Path:
    from catboost import CatBoostClassifier

    base_feats = [
        "energy", "energyshort", "amp", "area", "log_amp", "log_area", "peak_argmin",
        "psd_caen", "psd_local", "a_over_s",
        "short_q", "mid_q", "tail_q", "tail_far_q",
        "frac_short", "frac_mid", "frac_tail", "frac_tail_far",
        "rise10_90", "fall90_10", "width50", "width20",
        "asymmetry", "baseline_noise", "post_peak_noise", "n_peaks",
        "energy_skew", "energy_kurt",
    ]
    pca_cols = [c for c in ff.columns if c.startswith("pca_")]
    win_cols = [c for c in ff.columns
                if c.startswith("win_") or c.startswith("winf_")
                or c in ["frac_late80", "frac_late200", "post_centroid"]]
    feat_cols = base_feats + pca_cols + win_cols
    X = ff[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    mask = y >= 0
    X_tr, y_tr = X[mask], y[mask]
    print(f"  training CatBoost on {len(X_tr)} samples × {X_tr.shape[1]} features ...", flush=True)
    cat = __import__("catboost").CatBoostClassifier(
        iterations=400, learning_rate=0.04, depth=5, loss_function="MultiClass",
        random_seed=0, verbose=0, thread_count=-1,
    )
    cat.fit(X_tr, y_tr)
    imp = cat.get_feature_importance()
    order = np.argsort(imp)[::-1][:18]
    names = [feat_cols[i] for i in order]
    vals = imp[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(names)), vals, color="#4c72b0", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Важность по CatBoost (нормированная)")
    ax.set_title("Топ-18 признаков по важности")
    ax.grid(True, axis="x", alpha=0.3)
    out = FIGURES / "fig2_feature_importance.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_sample_signals(ff: pd.DataFrame, y: np.ndarray) -> Path | None:
    adc_path = ARTIFACTS / "adc_raw.npy"
    if not adc_path.exists():
        print("  (skipping sample signals: adc_raw.npy not in artifacts/)", flush=True)
        return None
    adc = np.load(adc_path).astype(np.float32)
    base = np.median(adc[:, :80], axis=1)
    sig = base[:, None] - adc
    peak = sig.argmax(axis=1)
    amp = sig.max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    L = sig.shape[1]
    anchor = 100
    POST = 300
    span = anchor + POST

    def aligned_mean(class_label: int, color: str, label: str) -> None:
        idx = np.where((y == class_label) & (amp > 200))[0]
        if len(idx) == 0:
            return
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, size=min(500, len(idx)), replace=False)
        acc = np.zeros(span, dtype=np.float32)
        cnt = np.zeros(span, dtype=np.int32)
        for i in pick:
            p = int(peak[i])
            a = float(amp[i])
            if a <= 1:
                continue
            src_lo = max(0, p - anchor)
            src_hi = min(L, p + POST)
            dst_lo = anchor - (p - src_lo)
            dst_hi = dst_lo + (src_hi - src_lo)
            acc[dst_lo:dst_hi] += sig[i, src_lo:src_hi] / a
            cnt[dst_lo:dst_hi] += 1
        mean = acc / np.maximum(cnt, 1)
        ax.plot(np.arange(span) - anchor, mean, color=color, linewidth=1.6, label=label)

    aligned_mean(0, "#55a868", "Нейтрон (среднее)")
    aligned_mean(1, "#dd8452", "Гамма-квант (среднее)")

    ax.set_xlabel("Время от пика (отсчёты, 1 отсчёт = 2 нс)")
    ax.set_ylabel("Амплитуда (нормированная)")
    ax.set_title("Усреднённые формы сигналов (выровнены по пику, нормированы по амплитуде)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlim(-50, 250)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = FIGURES / "fig3_sample_signals.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    print("loading features ...", flush=True)
    ff = load_features()
    print(f"  ff {ff.shape}", flush=True)
    y = build_seed_labels(ff)

    print("fig 1: PSD distribution ...", flush=True)
    p1 = fig_psd_distribution(ff)
    print(f"  saved {p1}", flush=True)

    print("fig 2: feature importance ...", flush=True)
    p2 = fig_feature_importance(ff, y)
    print(f"  saved {p2}", flush=True)

    print("fig 3: sample signals ...", flush=True)
    p3 = fig_sample_signals(ff, y)
    if p3:
        print(f"  saved {p3}", flush=True)


if __name__ == "__main__":
    main()
