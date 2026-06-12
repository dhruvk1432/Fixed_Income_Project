# Data contract

Expected local files:

- `gsw_yields.xlsx`: Gurkaynak-Sack-Wright fixed-maturity Treasury yields.
- `treasury_panel_pca.xlsx`: Treasury bond panel used for implementation checks.

These workbooks are intentionally ignored by git. Keep raw vendor/course data
local and commit only code, notebooks, documentation, and aggregate diagnostics.

Optional public FRED context downloaded by `scripts/fetch_fred_series.py` is
written to `data/raw/fred/`.  That directory is also ignored by git.
