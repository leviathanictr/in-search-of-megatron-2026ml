"""
Загружает сырые данные соревнования (data_set_1.csv) и сохраняет:
  - adc_raw.npy:           массив АЦП (N, 496)
  - features_basic.parquet: метаданные + базовые признаки (амплитуда, площадь, PSD)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd


def parse_compass_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    boards, channels, tts, energies, energyshorts, flags, samples = [], [], [], [], [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        _ = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split(";")
            boards.append(int(parts[0]))
            channels.append(int(parts[1]))
            tts.append(int(parts[2]))
            energies.append(int(parts[3]))
            energyshorts.append(int(parts[4]))
            flags.append(parts[5])
            samples.append(parts[6:])
    adc = np.array(samples, dtype=np.int32)
    meta = pd.DataFrame({
        "board": boards,
        "channel": channels,
        "timetag": tts,
        "energy": energies,
        "energyshort": energyshorts,
        "flags": flags,
    })
    return meta, adc


def build_basic_features(meta: pd.DataFrame, adc: np.ndarray) -> pd.DataFrame:
    eps = 1e-9
    energies = meta["energy"].to_numpy()
    energyshorts = meta["energyshort"].to_numpy()
    baseline = np.median(adc[:, :80], axis=1).astype(np.int32)
    peak_min = adc.min(axis=1)
    peak_argmin = adc.argmin(axis=1)
    amplitude = baseline - peak_min
    area = (baseline[:, None] - adc).clip(min=0).sum(axis=1)
    sat_max = adc.max(axis=1)
    a_over_s = amplitude / (area + eps)
    psd = (energies - energyshorts) / (energies + eps)
    return pd.DataFrame({
        "idx": np.arange(len(adc)),
        "energy": energies,
        "energyshort": energyshorts,
        "flags": meta["flags"].to_numpy(),
        "baseline": baseline,
        "amplitude": amplitude,
        "area": area,
        "peak_argmin": peak_argmin,
        "psd": psd,
        "sat_max": sat_max,
        "a_over_s": a_over_s,
    })


def main(data_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"loading {data_path} ...", flush=True)
    meta, adc = parse_compass_csv(data_path)
    print(f"  parsed {adc.shape[0]} events x {adc.shape[1]} samples in {time.time()-t0:.1f}s", flush=True)

    print("computing basic features ...", flush=True)
    feats = build_basic_features(meta, adc)
    feats_path = out_dir / "features_basic.parquet"
    feats.to_parquet(feats_path, index=False)
    print(f"  saved {feats_path} ({feats.shape})", flush=True)

    adc_path = out_dir / "adc_raw.npy"
    np.save(adc_path, adc.astype(np.int16))
    print(f"  saved {adc_path}", flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    data = here / "data" / "data_set_1.csv"
    artifacts = here / "artifacts"
    if not data.exists():
        sys.exit(f"data_set_1.csv not found at {data}. See data/README.md for download instructions.")
    main(data, artifacts)
