from pathlib import Path

import pytest

from src.treasury_pca_butterfly import (
    backtest_mean_reversion,
    compute_spread,
    fit_pca,
    load_gsw_yields,
    rolling_zscore,
    solve_butterfly_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def _has_data() -> bool:
    return (ROOT / "gsw_yields.xlsx").exists()


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
