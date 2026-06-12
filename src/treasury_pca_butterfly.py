"""Reusable PCA butterfly and relative-value utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class PCAFit:
    loadings: pd.DataFrame
    explained_variance: pd.Series
    scores: pd.DataFrame


@dataclass(frozen=True)
class RelativeValueConfig:
    entry_z: float = 1.50
    exit_z: float = 0.25
    lookback: int = 126
    max_hold_days: int = 42
    transaction_cost_bp: float = 0.03
    target_daily_vol_bp: float = 2.50
    max_abs_position: float = 1.50
    carry_tolerance_bp: float = 0.05


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


def factor_exposures(loadings: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Return PC exposures of a yield-space butterfly."""

    common = weights.index
    exposures = loadings.loc[common].T @ weights.loc[common].astype(float)
    exposures.name = "exposure"
    return exposures


def solve_regularized_butterfly_weights(
    loadings: pd.DataFrame,
    maturities: tuple[int, int, int] = (2, 5, 10),
    belly_weight: float = -1.0,
    gross_cap: float = 3.0,
    ridge: float = 0.10,
    turnover_penalty: float = 0.0,
    previous_weights: pd.Series | None = None,
) -> pd.Series:
    """Solve stable butterfly weights with soft PC neutrality and gross cap.

    Exact PC1/PC2 neutrality is attractive in a static classroom exercise but
    can create explosive wing weights in rolling samples.  This optimizer keeps
    the belly anchor, penalizes residual level/slope exposure, discourages
    large notionals, and optionally adds a turnover penalty.
    """

    front, belly, back = maturities
    pcs = ["PC1", "PC2"]
    sub = loadings.loc[[front, belly, back], pcs].astype(float)

    if previous_weights is None:
        previous = np.array([0.5, belly_weight, 0.5])
    else:
        previous = previous_weights.reindex([front, belly, back]).fillna(0.0).values

    def objective(x: np.ndarray) -> float:
        w = np.array([x[0], belly_weight, x[1]])
        exposure = sub.T.values @ w
        size_penalty = ridge * float(np.dot(w, w))
        turn_penalty = turnover_penalty * float(np.dot(w - previous, w - previous))
        return float(np.dot(exposure, exposure) + size_penalty + turn_penalty)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: gross_cap
            - (abs(x[0]) + abs(belly_weight) + abs(x[1])),
        }
    ]
    start = np.array([0.5, 0.5])
    result = minimize(objective, start, method="SLSQP", constraints=constraints)
    if not result.success:
        # Fall back to the ridge closed form without gross constraint.
        a = sub.loc[[front, back]].T.values
        b = belly_weight * sub.loc[belly].values
        x = np.linalg.solve(a.T @ a + ridge * np.eye(2), -a.T @ b)
    else:
        x = result.x

    weights = pd.Series(
        {front: x[0], belly: belly_weight, back: x[1]},
        name="regularized_weight",
    )
    gross = weights.abs().sum()
    if gross > gross_cap:
        scale = max((gross_cap - abs(belly_weight)) / weights.drop(belly).abs().sum(), 0.0)
        weights.loc[[front, back]] *= scale
    return weights


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


def monthly_regularized_weights(
    yields_bp: pd.DataFrame,
    window: int = 756,
    maturities: tuple[int, int, int] = (2, 5, 10),
    gross_cap: float = 3.0,
    ridge: float = 0.10,
    turnover_penalty: float = 0.05,
) -> pd.DataFrame:
    """Monthly walk-forward PCA weights with stability penalties."""

    rebalance_dates = pd.date_range(yields_bp.index.min(), yields_bp.index.max(), freq="BMS")
    rows = []
    previous = None
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
        weights = solve_regularized_butterfly_weights(
            fit.loadings,
            maturities=maturities,
            gross_cap=gross_cap,
            ridge=ridge,
            turnover_penalty=turnover_penalty,
            previous_weights=previous,
        )
        previous = weights
        exposures = factor_exposures(fit.loadings, weights)
        rows.append(
            {
                "date": date,
                **weights.to_dict(),
                "PC1_exposure": exposures.get("PC1", np.nan),
                "PC2_exposure": exposures.get("PC2", np.nan),
                "gross": weights.abs().sum(),
            }
        )
    if not rows:
        raise ValueError("Not enough history to estimate regularized rolling weights.")
    weights = pd.DataFrame(rows).set_index("date").sort_index()
    for col in maturities:
        weights[col] = weights[col].astype(float)
    return weights


def spread_from_weight_history(yields_bp: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    weight_cols = [col for col in weights.columns if isinstance(col, (int, np.integer))]
    daily_weights = weights[weight_cols].reindex(yields_bp.index).ffill()
    spread = (yields_bp[daily_weights.columns] * daily_weights).sum(axis=1)
    spread = spread.where(daily_weights.notna().all(axis=1))
    return spread.dropna().rename("rolling_spread_bp")


def interpolate_curve(row: pd.Series, maturities: Iterable[float]) -> pd.Series:
    """Linearly interpolate a yield curve row in maturity space."""

    clean = row.dropna().sort_index()
    x = clean.index.astype(float).values
    y = clean.astype(float).values
    targets = np.asarray(list(maturities), dtype=float)
    values = np.interp(targets, x, y)
    return pd.Series(values, index=targets)


def rolldown_spread(
    yields_bp: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    horizon_days: int = 21,
) -> pd.Series:
    """Approximate unchanged-curve rolldown of the weighted yield spread.

    Positive values mean the spread should rise if the curve is unchanged over
    the horizon.  This is a yield-space carry proxy, not full bond P&L.
    """

    years = horizon_days / 252.0
    if isinstance(weights, pd.Series):
        maturities = [float(col) for col in weights.index]
        rolled_maturities = [max(min(yields_bp.columns.astype(float)), m - years) for m in maturities]
        rows = []
        for date, row in yields_bp.iterrows():
            current = row[weights.index].astype(float)
            rolled = interpolate_curve(row, rolled_maturities)
            rolled.index = weights.index
            rows.append((date, float((rolled - current) @ weights.astype(float))))
        return pd.Series(dict(rows), name="rolldown_spread_bp")

    weight_cols = [col for col in weights.columns if isinstance(col, (int, np.integer))]
    daily_weights = weights[weight_cols].reindex(yields_bp.index).ffill()
    rows = []
    min_maturity = min(yields_bp.columns.astype(float))
    for date, row in yields_bp.iterrows():
        w = daily_weights.loc[date].dropna()
        if len(w) != len(weight_cols):
            rows.append((date, np.nan))
            continue
        maturities = [float(col) for col in w.index]
        rolled_maturities = [max(min_maturity, m - years) for m in maturities]
        current = row[w.index].astype(float)
        rolled = interpolate_curve(row, rolled_maturities)
        rolled.index = w.index
        rows.append((date, float((rolled - current) @ w.astype(float))))
    return pd.Series(dict(rows), name="rolldown_spread_bp")


def robust_zscore(series: pd.Series, lookback: int = 126) -> pd.Series:
    """Lagged robust z-score using rolling median and MAD."""

    lagged = series.shift(1)
    median = lagged.rolling(lookback, min_periods=max(30, lookback // 3)).median()
    mad = (lagged - median).abs().rolling(lookback, min_periods=max(30, lookback // 3)).median()
    sigma = 1.4826 * mad.replace(0.0, np.nan)
    return ((series - median) / sigma).rename("robust_zscore")


def enhanced_relative_value_backtest(
    spread: pd.Series,
    zscore: pd.Series,
    carry: pd.Series | None = None,
    config: RelativeValueConfig | None = None,
) -> pd.DataFrame:
    """Backtest a gated, volatility-targeted curvature mean-reversion rule."""

    cfg = config or RelativeValueConfig()
    data = pd.DataFrame({"spread": spread, "zscore": zscore}).dropna()
    if carry is not None:
        data["carry"] = carry.reindex(data.index)
    else:
        data["carry"] = 0.0

    desired = []
    current = 0.0
    hold = 0
    for _, row in data.iterrows():
        z = row["zscore"]
        carry_bp = row["carry"]
        if current == 0:
            hold = 0
            candidate = 0.0
            if z <= -cfg.entry_z:
                candidate = 1.0
            elif z >= cfg.entry_z:
                candidate = -1.0

            carry_ok = candidate == 0.0 or candidate * carry_bp >= -cfg.carry_tolerance_bp
            current = candidate if carry_ok else 0.0
        else:
            hold += 1
            if abs(z) <= cfg.exit_z or hold >= cfg.max_hold_days:
                current = 0.0
                hold = 0
        desired.append(current)

    data["direction"] = desired
    spread_vol = data["spread"].diff().rolling(63, min_periods=20).std()
    scale = (cfg.target_daily_vol_bp / spread_vol).replace([np.inf, -np.inf], np.nan)
    data["position"] = data["direction"] * scale.clip(0.0, cfg.max_abs_position).fillna(0.0)
    data["gross_pnl"] = data["position"].shift(1).fillna(0.0) * data["spread"].diff().fillna(0.0)
    data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())
    data["cost"] = data["turnover"] * cfg.transaction_cost_bp
    data["net_pnl"] = data["gross_pnl"] - data["cost"]
    data["cum_net_pnl"] = data["net_pnl"].cumsum()
    return data


def curvature_momentum_carry_backtest(
    spread: pd.Series,
    carry: pd.Series,
    momentum_lookback: int = 63,
    carry_agreement: bool = True,
    transaction_cost_bp: float = 0.03,
    target_daily_vol_bp: float = 2.50,
    max_abs_position: float = 1.50,
) -> pd.DataFrame:
    """Test a carry-aligned curvature momentum alternative.

    The preliminary mean-reversion evidence is weak.  This rule asks whether
    curvature shocks trend when the curve's own rolldown points in the same
    direction.  It is still a research proxy, but it is economically distinct
    from repeatedly fading a non-stationary residual.
    """

    data = pd.DataFrame({"spread": spread, "carry": carry}).dropna()
    data["momentum"] = data["spread"] - data["spread"].shift(momentum_lookback)
    momentum_signal = np.sign(data["momentum"]).fillna(0.0)
    carry_signal = np.sign(data["carry"]).fillna(0.0)
    if carry_agreement:
        direction = momentum_signal.where(momentum_signal == carry_signal, 0.0)
    else:
        direction = momentum_signal

    spread_vol = data["spread"].diff().rolling(63, min_periods=20).std()
    scale = (target_daily_vol_bp / spread_vol).replace([np.inf, -np.inf], np.nan)
    data["position"] = direction * scale.clip(0.0, max_abs_position).fillna(0.0)
    data["gross_pnl"] = data["position"].shift(1).fillna(0.0) * data["spread"].diff().fillna(0.0)
    data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())
    data["cost"] = data["turnover"] * transaction_cost_bp
    data["net_pnl"] = data["gross_pnl"] - data["cost"]
    data["cum_net_pnl"] = data["net_pnl"].cumsum()
    return data


def strategy_grid(
    spread: pd.Series,
    carry: pd.Series | None = None,
    lookbacks: Iterable[int] = (63, 126, 252),
    entries: Iterable[float] = (1.0, 1.5, 2.0),
    costs: Iterable[float] = (0.01, 0.03, 0.05),
) -> pd.DataFrame:
    """Evaluate a small parameter grid for robustness, not curve fitting."""

    rows = []
    for lookback in lookbacks:
        z = robust_zscore(spread, lookback=lookback)
        for entry in entries:
            for cost in costs:
                cfg = RelativeValueConfig(entry_z=entry, lookback=lookback, transaction_cost_bp=cost)
                bt = enhanced_relative_value_backtest(spread, z, carry=carry, config=cfg)
                stats = performance_stats(bt)
                rows.append(
                    {
                        "lookback": lookback,
                        "entry_z": entry,
                        "cost_bp": cost,
                        **stats,
                    }
                )
    return pd.DataFrame(rows).sort_values(["ann_sharpe", "total_pnl_bp"], ascending=False)


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
