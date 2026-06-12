# Treasury PCA Butterfly and Relative Value

This project builds a 2s5s10s Treasury butterfly that is neutral to the first two
yield-curve principal components and evaluates whether the residual curvature
spread is tradable after realistic walk-forward and implementation checks.  The
upgraded strategy layer adds regularized PCA hedge weights, carry/rolldown,
volatility-targeted sizing, transaction costs, and parameter-robustness checks.

## What is in the repo

- `Treasury_PCA_Butterfly_Relative_Value.ipynb`: presentation-ready research
  notebook.
- `paper/Treasury_PCA_Butterfly_Relative_Value_Mini_Paper.pdf`: short mini-paper for first-pass reading.
- `paper/figures/`: vector figures used in the mini-paper.
- `src/treasury_pca_butterfly.py`: reusable PCA, butterfly, backtest, and bond
  implementation/strategy utilities.
- `scripts/fetch_fred_series.py`: optional public-data downloader for market
  stress, credit, and curve-regime context.  Downloaded CSV files stay under
  ignored `data/raw/`.
- `docs/knowledge_base_protocol.md`: local NotebookLM/knowledge-base protocol
  used to ground the research extension without shipping private notes.
- `tests/test_treasury_pca_butterfly.py`: smoke tests that run when the local
  data workbooks are present.
- `data/README.md`: data contract and public-repo policy.

## Research design

1. Use yield changes rather than yield levels for PCA.
2. Compare correlation PCA for factor interpretation with covariance PCA for
   hedging and P&L units.
3. Solve butterfly weights that neutralize level and slope while keeping the
   belly short.
4. Evaluate stationarity, half-life, and a lagged z-score signal.
5. Compare static in-sample weights with rolling walk-forward PCA weights.
6. Map the fixed-maturity signal to nearest tradable Treasury bonds to quantify
   the gap between a clean factor signal and an implementable portfolio.
7. Replace fragile exact rolling neutrality with a regularized gross-capped
   hedge solver.
8. Add carry/rolldown as a filter so the strategy does not blindly fight the
   curve's natural pull.
9. Stress the signal across lookbacks, entry thresholds, and transaction costs
   before calling it researchable.

## Data

Raw Excel workbooks are not tracked. Place these files in the repo root or
`data/raw/` before running the notebook:

- `gsw_yields.xlsx`
- `treasury_panel_pca.xlsx`

The notebook reports only aggregate diagnostics and plots.
Optional public FRED series can be downloaded with:

```bash
python scripts/fetch_fred_series.py
```

Those files are written to `data/raw/fred/` and are not tracked.
