"""Reusable PCA butterfly and relative-value utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class PCAFit:
    loadings: pd.DataFrame
    explained_variance: pd.Series
    scores: pd.DataFrame


def _find_file(root: Path, filename: str) -> Path:
    candidates = [
        root / filename,
        root / "data" / filename,
        root / "data" / "raw" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Put it in the repo root or data/raw/."
    )


def normalize_maturity_columns(columns: Iterable[object]) -> list[object]:
    out = []
    for col in columns:
        if isinstance(col, str) and col.isdigit():
            out.append(int(col))
        elif isinstance(col, float) and col.is_integer():
            out.append(int(col))
        elif isinstance(col, (int, np.integer)):
            out.append(int(col))
        else:
            out.append(col)
    return out


def load_gsw_yields(root: str | Path = ".") -> pd.DataFrame:
    path = _find_file(Path(root), "gsw_yields.xlsx")
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = normalize_maturity_columns(df.columns)
    df = df.dropna(how="all")
    # GSW yields are in percent. Convert to bp for risk/PnL readability.
    return df.astype(float) * 100.0


def load_treasury_panel(root: str | Path = ".") -> pd.DataFrame:
    path = _find_file(Path(root), "treasury_panel_pca.xlsx")
    panel = pd.read_excel(path, sheet_name="panel")
    for col in ["caldt", "issue_date", "maturity_date"]:
        panel[col] = pd.to_datetime(panel[col])
    return panel.sort_values(["caldt", "ttm", "kytreasno"])


def orient_loadings(loadings: pd.DataFrame) -> pd.DataFrame:
    out = loadings.copy()
    maturities = pd.Series(out.index.astype(float), index=out.index)
    if out["PC1"].sum() < 0:
        out["PC1"] *= -1
    if out["PC2"].corr(maturities) < 0:
        out["PC2"] *= -1
    if 5 in out.index and {2, 10}.issubset(set(out.index)):
        wings = 0.5 * (out.loc[2, "PC3"] + out.loc[10, "PC3"])
        if wings - out.loc[5, "PC3"] < 0:
            out["PC3"] *= -1
    return out


def fit_pca(yields_bp: pd.DataFrame, method: str = "covariance", n_components: int = 3) -> PCAFit:
    # Rolling Treasury panels can have incomplete long-maturity histories.
    # Keep maturities with enough observations, forward-fill within the
    # estimation window, then require complete rows for PCA.
    clean = yields_bp.sort_index().copy()
    min_obs = max(n_components + 2, int(len(clean) * 0.80))
    clean = clean.dropna(axis=1, thresh=min_obs).ffill().dropna(how="any")
    changes = clean.diff().dropna(how="any")
    if len(changes) < n_components + 1:
        raise ValueError("Not enough complete yield-change observations for PCA.")

    if method == "correlation":
        matrix = (changes - changes.mean()) / changes.std(ddof=0)
    elif method == "covariance":
        matrix = changes - changes.mean()
    else:
        raise ValueError("method must be 'covariance' or 'correlation'")

    pca = PCA(n_components=n_components)
    scores = pd.DataFrame(
        pca.fit_transform(matrix),
        index=matrix.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=changes.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings = orient_loadings(loadings)
    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=loadings.columns,
        name="explained_variance",
    )
    return PCAFit(loadings=loadings, explained_variance=explained, scores=scores)


def solve_butterfly_weights(
    loadings: pd.DataFrame,
    maturities: tuple[int, int, int] = (2, 5, 10),
    belly_weight: float = -1.0,
) -> pd.Series:
    front, belly, back = maturities
    sub = loadings.loc[[front, belly, back], ["PC1", "PC2"]].astype(float)
    lhs = np.array(
        [
            [sub.loc[front, "PC1"], sub.loc[back, "PC1"]],
            [sub.loc[front, "PC2"], sub.loc[back, "PC2"]],
        ]
    )
    rhs = -belly_weight * np.array([sub.loc[belly, "PC1"], sub.loc[belly, "PC2"]])
    front_back = np.linalg.solve(lhs, rhs)
    return pd.Series(
        {front: front_back[0], belly: belly_weight, back: front_back[1]},
        name="weight",
    )


def compute_spread(yields_bp: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return (yields_bp[weights.index] * weights).sum(axis=1).rename("spread_bp")


def adf_summary(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    stat, pvalue, usedlag, nobs, *_ = adfuller(s, autolag="AIC")
    return {
        "adf_stat": stat,
        "pvalue": pvalue,
        "used_lag": usedlag,
        "nobs": nobs,
    }


def half_life_days(series: pd.Series) -> float:
    s = series.dropna()
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna().loc[lagged.index]
    x = np.column_stack([np.ones(len(lagged)), lagged.values])
    beta = np.linalg.lstsq(x, delta.values, rcond=None)[0][1]
    if beta >= 0:
        return np.inf
    return float(-np.log(2.0) / beta)


def rolling_zscore(series: pd.Series, lookback: int = 126) -> pd.Series:
    mean = series.shift(1).rolling(lookback, min_periods=max(20, lookback // 3)).mean()
    std = series.shift(1).rolling(lookback, min_periods=max(20, lookback // 3)).std()
    return ((series - mean) / std).rename("zscore")


def backtest_mean_reversion(
    spread: pd.Series,
    zscore: pd.Series,
    entry_z: float = 1.25,
    exit_z: float = 0.25,
    cost_bp: float = 0.02,
    max_hold_days: int = 63,
) -> pd.DataFrame:
    data = pd.DataFrame({"spread": spread, "zscore": zscore}).dropna()
    position = []
    current = 0
    hold = 0
    for z in data["zscore"]:
        if current == 0:
            hold = 0
            if z <= -entry_z:
                current = 1
            elif z >= entry_z:
                current = -1
        else:
            hold += 1
            if abs(z) <= exit_z or hold >= max_hold_days:
                current = 0
                hold = 0
        position.append(current)

    data["position"] = position
    data["gross_pnl"] = data["position"].shift(1).fillna(0.0) * data["spread"].diff().fillna(0.0)
    turnover = data["position"].diff().abs().fillna(data["position"].abs())
    data["cost"] = turnover * cost_bp
    data["net_pnl"] = data["gross_pnl"] - data["cost"]
    data["cum_net_pnl"] = data["net_pnl"].cumsum()
    return data


def performance_stats(backtest: pd.DataFrame, pnl_col: str = "net_pnl") -> dict[str, float]:
    pnl = backtest[pnl_col].dropna()
    active = backtest["position"].shift(1).fillna(0) != 0
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252.0) if pnl.std() > 0 else np.nan
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    trades = int((backtest["position"].diff().abs().fillna(backtest["position"].abs()) > 0).sum())
    active_pnl = pnl[active.reindex(pnl.index).fillna(False)]
    return {
        "total_pnl_bp": pnl.sum(),
        "ann_sharpe": sharpe,
        "max_drawdown_bp": drawdown.min(),
        "hit_rate_active": (active_pnl > 0).mean() if len(active_pnl) else np.nan,
        "active_days": int(active.sum()),
        "trades": trades,
    }


def monthly_rolling_weights(
    yields_bp: pd.DataFrame,
    window: int = 756,
    maturities: tuple[int, int, int] = (2, 5, 10),
) -> pd.DataFrame:
    rebalance_dates = pd.date_range(yields_bp.index.min(), yields_bp.index.max(), freq="BMS")
    rows = []
    for date in rebalance_dates:
        if date not in yields_bp.index:
            loc = yields_bp.index.searchsorted(date)
            if loc >= len(yields_bp.index):
                continue
            date = yields_bp.index[loc]
        history = yields_bp.loc[:date].iloc[:-1]
        if len(history) < window:
            continue
        fit = fit_pca(history.iloc[-window:], method="covariance")
        if not set(maturities).issubset(set(fit.loadings.index)):
            continue
        weights = solve_butterfly_weights(fit.loadings, maturities=maturities)
        rows.append({"date": date, **weights.to_dict()})
    if not rows:
        raise ValueError("Not enough history to estimate rolling PCA weights.")
    weights = pd.DataFrame(rows).set_index("date").sort_index()
    weights.columns = weights.columns.astype(int)
    return weights


def spread_from_weight_history(yields_bp: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    daily_weights = weights.reindex(yields_bp.index).ffill()
    spread = (yields_bp[daily_weights.columns] * daily_weights).sum(axis=1)
    spread = spread.where(daily_weights.notna().all(axis=1))
    return spread.dropna().rename("rolling_spread_bp")


def regime(date: pd.Timestamp) -> str:
    date = pd.Timestamp(date)
    if date < pd.Timestamp("2020-01-01"):
        return "Pre-COVID"
    if date < pd.Timestamp("2022-01-01"):
        return "COVID/QE"
    if date < pd.Timestamp("2024-01-01"):
        return "Hiking"
    return "Recent"


def pick_nearest_bonds(panel: pd.DataFrame, targets: Iterable[int], rebalance_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for date in rebalance_dates:
        day = panel.loc[panel["caldt"] == date].copy()
        if day.empty:
            continue
        for target in targets:
            eligible = day[(day["ttm"] >= max(0.5, target - 1.0)) & (day["ttm"] <= target + 1.0)]
            if eligible.empty:
                continue
            idx = (eligible["ttm"] - target).abs().idxmin()
            row = eligible.loc[idx]
            rows.append(
                {
                    "rebalance_date": date,
                    "target_tenor": target,
                    "kytreasno": row["kytreasno"],
                    "cusip": row["cusip"],
                    "ttm": row["ttm"],
                    "duration": row["duration"],
                    "dirty_price": row["dirty_price"],
                    "ytm": row["ytm"],
                }
            )
    return pd.DataFrame(rows)
