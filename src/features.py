"""
Полная параметризация сигналов сцинтиллятора.

Реализует все физические и формообразующие признаки v12:
  - амплитуда, площадь, отношение amp/area
  - PSD CAEN и локальное (по форме)
  - сегментные интегралы (short/mid/tail/tail_far) и их доли
  - времена нарастания и спада (10%-90%)
  - ширины на 50% и 20% амплитуды
  - асимметрия, шум базовой линии и хвоста
  - число пиков, скос и эксцесс энергетического распределения

Дополнительные признаки (extra):
  - PCA-проекция выровненных по пику волн (10 компонент)
  - 8 окон-интегралов после пика и их доли
  - доля энергии в позднем хвосте (>80, >200 после пика)
  - центроид распределения энергии
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


WINDOW_STARTS = [0, 5, 10, 20, 40, 80, 160, 240]
WINDOW_ENDS = [5, 10, 20, 40, 80, 160, 240, 380]


def compute_shape_features(adc: np.ndarray, basic: pd.DataFrame) -> pd.DataFrame:
    N, L = adc.shape
    base = np.median(adc[:, :80], axis=1).astype(np.float32)
    sig = base[:, None] - adc.astype(np.float32)
    peak_argmin = adc.argmin(axis=1)
    peak_min = adc.min(axis=1)
    amp = (base - peak_min).astype(np.float32)
    area = sig.clip(min=0).sum(axis=1)

    prepeak = np.zeros(N, dtype=np.float32)
    short = np.zeros(N, dtype=np.float32)
    mid = np.zeros(N, dtype=np.float32)
    tail = np.zeros(N, dtype=np.float32)
    tail_far = np.zeros(N, dtype=np.float32)
    rise10_90 = np.zeros(N, dtype=np.float32)
    fall90_10 = np.zeros(N, dtype=np.float32)
    width50 = np.zeros(N, dtype=np.float32)
    width20 = np.zeros(N, dtype=np.float32)
    asymmetry = np.zeros(N, dtype=np.float32)
    post_peak_noise = np.zeros(N, dtype=np.float32)
    n_peaks = np.zeros(N, dtype=np.float32)
    energy_kurt = np.zeros(N, dtype=np.float32)
    energy_skew = np.zeros(N, dtype=np.float32)

    baseline_noise = np.std(adc[:, :80] - base[:, None], axis=1).astype(np.float32)

    for i in range(N):
        s = sig[i]
        p = int(peak_argmin[i])
        a = float(amp[i])
        if a <= 1:
            continue
        prepeak[i] = float(s[max(0, p-40):p].clip(min=0).sum()) if p > 0 else 0.0
        short[i] = float(s[p:min(L, p+20)].clip(min=0).sum())
        mid[i] = float(s[p+20:min(L, p+60)].clip(min=0).sum()) if p + 20 < L else 0.0
        tail[i] = float(s[p+60:min(L, p+200)].clip(min=0).sum()) if p + 60 < L else 0.0
        tail_far[i] = float(s[p+150:min(L, p+400)].clip(min=0).sum()) if p + 150 < L else 0.0

        half, twenty, ten, ninety = a * 0.5, a * 0.2, a * 0.1, a * 0.9
        width50[i] = float((s > half).sum())
        width20[i] = float((s > twenty).sum())
        rise_idx = np.where(s[max(0, p-80):p+1] >= ten)[0]
        rise_top = np.where(s[max(0, p-80):p+1] >= ninety)[0]
        if rise_idx.size and rise_top.size:
            rise10_90[i] = float(rise_top[0] - rise_idx[0])
        fall_idx = np.where(s[p:min(L, p+300)] >= ninety)[0]
        fall_top = np.where(s[p:min(L, p+300)] <= ten)[0]
        if fall_idx.size and fall_top.size:
            ftop = fall_top[fall_top > fall_idx[-1]] if (fall_top > fall_idx[-1]).any() else fall_top
            if ftop.size:
                fall90_10[i] = float(ftop[0] - fall_idx[-1])

        postpeak = float(s[p:min(L, p+200)].clip(min=0).sum())
        if postpeak > 0 and prepeak[i] >= 0:
            asymmetry[i] = (postpeak - prepeak[i]) / (postpeak + prepeak[i] + 1e-6)

        end = min(L, p + 200)
        if end - p > 20:
            post_peak_noise[i] = float(adc[i, p+20:end].std())
        if a > 50:
            from_p = s[max(0, p-3):min(L, p+200)]
            n_peaks[i] = float(((from_p[:-1] < half) & (from_p[1:] >= half)).sum())

        if area[i] > 0:
            x = s.clip(min=0)
            w = x / x.sum()
            idx = np.arange(L)
            mean = (idx * w).sum()
            var = ((idx - mean) ** 2 * w).sum()
            std = np.sqrt(var) + 1e-9
            energy_skew[i] = float(((idx - mean) ** 3 * w).sum() / std ** 3)
            energy_kurt[i] = float(((idx - mean) ** 4 * w).sum() / std ** 4 - 3)

    energies = basic["energy"].to_numpy()
    energyshorts = basic["energyshort"].to_numpy()
    psd_caen = (energies - energyshorts) / np.maximum(energies, 1)
    psd_local = (area - (short + prepeak)) / np.maximum(area, 1.0)

    return pd.DataFrame({
        "idx": np.arange(N),
        "energy": energies,
        "energyshort": energyshorts,
        "flags": basic["flags"].to_numpy(),
        "baseline": base,
        "amp": amp,
        "area": area,
        "peak_argmin": peak_argmin,
        "psd_caen": psd_caen,
        "psd_local": psd_local,
        "log_amp": np.log1p(amp),
        "log_area": np.log1p(area),
        "a_over_s": amp / np.maximum(area, 1.0),
        "short_q": short,
        "mid_q": mid,
        "tail_q": tail,
        "tail_far_q": tail_far,
        "frac_short": short / np.maximum(area, 1.0),
        "frac_mid": mid / np.maximum(area, 1.0),
        "frac_tail": tail / np.maximum(area, 1.0),
        "frac_tail_far": tail_far / np.maximum(area, 1.0),
        "rise10_90": rise10_90,
        "fall90_10": fall90_10,
        "width50": width50,
        "width20": width20,
        "asymmetry": asymmetry,
        "baseline_noise": baseline_noise,
        "post_peak_noise": post_peak_noise,
        "n_peaks": n_peaks,
        "energy_skew": energy_skew,
        "energy_kurt": energy_kurt,
        "sat": (adc.max(axis=1) >= 16383).astype(np.int8),
    })


def compute_extra_features(adc: np.ndarray) -> pd.DataFrame:
    N, L = adc.shape
    base = np.median(adc[:, :80], axis=1).astype(np.float32)
    sig = base[:, None] - adc.astype(np.float32)
    peak = sig.argmax(axis=1)

    anchor = 100
    post_len = 300
    shifted = np.zeros((N, anchor + post_len), dtype=np.float32)
    for i in range(N):
        p = int(peak[i])
        src_lo = max(0, p - anchor)
        src_hi = min(L, p + post_len)
        dst_lo = anchor - (p - src_lo)
        dst_hi = dst_lo + (src_hi - src_lo)
        shifted[i, dst_lo:dst_hi] = sig[i, src_lo:src_hi]

    amp_norm = np.maximum(shifted.max(axis=1, keepdims=True), 1.0)
    shifted_norm = shifted / amp_norm

    pca = PCA(n_components=10, random_state=0)
    pca_feats = pca.fit_transform(shifted_norm)

    post = shifted[:, anchor:]
    post_wins = np.stack([post[:, a:b].clip(min=0).sum(axis=1) for a, b in zip(WINDOW_STARTS, WINDOW_ENDS)], axis=1)
    post_norms = post_wins / np.maximum(post_wins.sum(axis=1, keepdims=True), 1.0)

    out = pd.DataFrame(pca_feats, columns=[f"pca_{i}" for i in range(10)])
    for i, (a, b) in enumerate(zip(WINDOW_STARTS, WINDOW_ENDS)):
        out[f"win_{a}_{b}"] = post_wins[:, i]
        out[f"winf_{a}_{b}"] = post_norms[:, i]
    out["frac_late80"] = post[:, 80:].clip(min=0).sum(axis=1) / np.maximum(post.clip(min=0).sum(axis=1), 1.0)
    out["frac_late200"] = post[:, 200:].clip(min=0).sum(axis=1) / np.maximum(post.clip(min=0).sum(axis=1), 1.0)
    out["post_centroid"] = (np.arange(post.shape[1])[None, :] * post.clip(min=0)).sum(axis=1) / np.maximum(post.clip(min=0).sum(axis=1), 1.0)
    out["idx"] = np.arange(N)
    return out


def main(artifacts_dir: Path) -> None:
    adc_path = artifacts_dir / "adc_raw.npy"
    basic_path = artifacts_dir / "features_basic.parquet"
    if not (adc_path.exists() and basic_path.exists()):
        sys.exit(f"missing inputs. Run src/load_data.py first.")

    print("loading raw ADC and basic features ...", flush=True)
    adc = np.load(adc_path).astype(np.float32)
    basic = pd.read_parquet(basic_path)

    print(f"  adc {adc.shape}", flush=True)

    t0 = time.time()
    print("computing shape features ...", flush=True)
    full = compute_shape_features(adc, basic)
    full_path = artifacts_dir / "features_full.parquet"
    full.to_parquet(full_path, index=False)
    print(f"  saved {full_path} ({full.shape}) in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    print("computing extra features (PCA + windows) ...", flush=True)
    extra = compute_extra_features(adc)
    extra_path = artifacts_dir / "features_extra.parquet"
    extra.to_parquet(extra_path, index=False)
    print(f"  saved {extra_path} ({extra.shape}) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    main(here / "artifacts")
