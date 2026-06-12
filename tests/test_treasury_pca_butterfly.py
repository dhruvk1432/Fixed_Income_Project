from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.treasury_pca_butterfly import (
    RelativeValueConfig,
    backtest_mean_reversion,
    compute_spread,
    curvature_momentum_carry_backtest,
    enhanced_relative_value_backtest,
    factor_exposures,
    fit_pca,
    load_gsw_yields,
    rolldown_spread,
    rolling_zscore,
    robust_zscore,
    solve_regularized_butterfly_weights,
    solve_butterfly_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def _has_data() -> bool:
    return (ROOT / "gsw_yields.xlsx").exists()


def test_regularized_weights_control_exposure_and_gross():
    index = pd.Index([1, 2, 3, 5, 7, 10], name="maturity")
    loadings = pd.DataFrame(
        {
            "PC1": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45],
            "PC2": [-0.50, -0.30, -0.10, 0.05, 0.22, 0.45],
            "PC3": [0.30, 0.10, -0.05, -0.20, -0.05, 0.25],
        },
        index=index,
    )
    weights = solve_regularized_butterfly_weights(loadings, gross_cap=2.5, ridge=0.10)
    exposures = factor_exposures(loadings, weights)
    assert weights.abs().sum() <= 2.5 + 1e-8
    assert abs(weights.loc[5] + 1.0) < 1e-12
    assert exposures.loc[["PC1", "PC2"]].abs().max() < 0.20


def test_enhanced_relative_value_backtest_has_costs_and_carry_filter():
    dates = pd.date_range("2021-01-01", periods=320, freq="B")
    curve = pd.DataFrame(
        {
            2: 100 + np.sin(np.linspace(0, 20, len(dates))) * 12,
            5: 130 + np.sin(np.linspace(0, 20, len(dates)) + 0.5) * 10,
            10: 160 + np.sin(np.linspace(0, 20, len(dates)) + 1.0) * 8,
        },
        index=dates,
    )
    weights = pd.Series({2: 0.55, 5: -1.0, 10: 0.45})
    spread = compute_spread(curve, weights)
    zscore = robust_zscore(spread, lookback=63)
    carry = rolldown_spread(curve, weights, horizon_days=21)
    bt = enhanced_relative_value_backtest(
        spread,
        zscore,
        carry=carry,
        config=RelativeValueConfig(entry_z=1.0, transaction_cost_bp=0.01),
    )
    momentum_bt = curvature_momentum_carry_backtest(spread, carry, momentum_lookback=21)
    assert {"position", "gross_pnl", "cost", "net_pnl", "cum_net_pnl"}.issubset(bt.columns)
    assert {"position", "gross_pnl", "cost", "net_pnl", "cum_net_pnl"}.issubset(momentum_bt.columns)
    assert len(bt) > 100


@pytest.mark.skipif(not _has_data(), reason="local raw data not present")
def test_static_butterfly_is_pc1_pc2_neutral():
    yields = load_gsw_yields(ROOT)
    fit = fit_pca(yields, method="covariance")
    weights = solve_butterfly_weights(fit.loadings)
    sub = fit.loadings.loc[weights.index, ["PC1", "PC2"]]
    exposure = sub.mul(weights, axis=0).sum()
    assert abs(exposure["PC1"]) < 1e-10
    assert abs(exposure["PC2"]) < 1e-10


@pytest.mark.skipif(not _has_data(), reason="local raw data not present")
def test_backtest_produces_accounting_columns():
    yields = load_gsw_yields(ROOT)
    fit = fit_pca(yields, method="covariance")
    weights = solve_butterfly_weights(fit.loadings)
    spread = compute_spread(yields, weights)
    zscore = rolling_zscore(spread, lookback=126)
    bt = backtest_mean_reversion(spread, zscore)
    assert {"spread", "zscore", "position", "gross_pnl", "cost", "net_pnl", "cum_net_pnl"}.issubset(bt.columns)
    assert len(bt) > 100
