# Treasury PCA Butterfly and Relative Value

This repository asks whether a PCA-neutral 2s5s10s Treasury butterfly isolates a
tradable curvature residual or only creates a clean-looking factor signal. The
project estimates Treasury principal components on daily yield changes,
constructs exact and regularized butterflies, tests stationarity and
walk-forward stability, and then adds carry/rolldown, transaction costs,
volatility targeting, and an actual-bond implementation check.

The important result is not that a simple butterfly z-score is production alpha.
The cleaned-up evidence rejects the naive mean-reversion story. The static
spread is not strongly stationary, exact rolling hedge ratios become unstable,
and the stricter mean-reversion strategy loses money after implementation-aware
filters. The stronger research contribution is the pivot: regularized hedge
construction stabilizes the portfolio, and carry-aligned curvature momentum is a
more plausible empirical branch to build on. The robustness layer keeps that
claim honest: purged CPCV, nested blocked selection, residual block bootstrap,
and full-curve covariance Monte Carlo support continued research but do not
promote the strategy to production alpha.

## What is in the repo

- `Treasury_PCA_Butterfly_Relative_Value.ipynb`: executed research notebook.
- `paper/Treasury_PCA_Butterfly_Relative_Value_Mini_Paper.pdf`: single-column
  empirical paper explaining the research object, methodology, results,
  limitations, and claim audit.
- `paper/figures/`: vector figures used in the paper.
- `src/treasury_pca_butterfly.py`: reusable PCA, butterfly, backtest,
  carry/rolldown, and bond-implementation utilities.
- `scripts/fetch_fred_series.py`: optional public-data downloader for market
  stress, credit, and curve-regime context. Downloaded CSV files stay under
  ignored `data/raw/`.
- `docs/knowledge_base_protocol.md`: local NotebookLM/knowledge-base protocol
  used to ground the research extension without shipping private research
  material.
- `tests/test_treasury_pca_butterfly.py`: smoke tests for the reusable
  implementation.
- `data/README.md`: data contract and public-repo policy.

## Claim hierarchy

The repo is written to make the strength of each claim clear:

1. **Implemented and validated:** yield-change PCA recovers the usual Treasury
   level, slope, and curvature factors.
2. **Construction result:** exact 2s5s10s weights neutralize PC1 and PC2 while
   retaining curvature exposure.
3. **Negative strategy result:** the residual is not strongly stationary and
   enhanced mean reversion loses money after realistic filters.
4. **Improved research branch:** regularized hedge construction prevents
   unstable wing weights, and carry-aligned momentum is more plausible than
   naive z-score fading.
5. **Anti-overfit evidence:** CPCV and Monte Carlo checks reject the tuned
   mean-reversion story and leave carry-aligned momentum as a candidate branch,
   not a proven edge.
6. **Not yet claimed:** executable cash-bond or futures alpha. A production
   version needs DV01 sizing, financing, bid/ask, roll mechanics, benchmark or
   CTD selection, and portfolio-level risk controls.

## Research design

1. Use yield changes rather than yield levels for PCA.
2. Compare correlation PCA for factor interpretation with covariance PCA for
   hedging and P&L units.
3. Solve butterfly weights that neutralize level and slope while keeping the
   belly short.
4. Evaluate stationarity, half-life, and a lagged z-score signal.
5. Compare static in-sample weights with rolling walk-forward PCA weights.
6. Replace fragile exact rolling neutrality with a regularized gross-capped
   hedge solver.
7. Add carry/rolldown as a filter so the strategy does not blindly fight the
   curve's natural pull.
8. Stress the signal across lookbacks, entry thresholds, and transaction costs.
9. Map fixed-maturity targets to actual Treasury bonds to quantify the
   implementation gap.
10. Validate strategy families with purged/embargoed CPCV, nested blocked
    model selection, factor-residual block bootstrap, and full-curve covariance
    Monte Carlo.

## Data

Raw Excel workbooks are not tracked. Place these files in the repo root or
`data/raw/` before running the notebook:

- `gsw_yields.xlsx`
- `treasury_panel_pca.xlsx`

The notebook reports only aggregate diagnostics and plots. Optional public FRED
series can be downloaded with:

```bash
python scripts/fetch_fred_series.py
```

Those files are written to `data/raw/fred/` and are not tracked.

## How to review

Start with the PDF paper for the claim hierarchy. Then open the notebook to see
the executed calculations and figures. The reusable implementation is kept in
`src/`, and the test suite provides quick checks that the public code still
loads and preserves key numerical behavior.
