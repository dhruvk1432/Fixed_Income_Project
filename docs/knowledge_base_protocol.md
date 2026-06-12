# Knowledge-Base Protocol

This repo can use the local NotebookLM mirror established in the sibling
`Options_Portfolio_Model` project.  The hook is configured in
`.codex/config.toml` and points to the local MCP server and source manifest in
that sibling repo.

The public repo does not include the private note corpus, credentials, or raw
data.  The goal is reproducible research hygiene: each extension should state
which theoretical idea it is testing, which data fields are needed, and how the
strategy would fail.

## Research Questions Used For This Extension

- Does the 2s5s10s butterfly represent stable curvature risk, or does PCA
  instability create hidden leverage and turnover?
- Does carry/rolldown improve a pure mean-reversion signal, or does it reveal
  that apparent cheapness is compensation for adverse curve shape?
- How much performance survives lagged signals, transaction costs,
  volatility-targeted sizing, and walk-forward weight estimation?
- Can a fixed-maturity factor signal be mapped into actual Treasury bonds
  without losing the statistical edge to duration, liquidity, and roll effects?

## Implementation Discipline

- Keep all raw workbooks and downloaded public data under `data/raw/`.
- Do not commit `.env`, API keys, provider tokens, or raw market data.
- Keep strategy functions in `src/` and call them from notebooks.
- Treat notebook outputs as research diagnostics, not as proof of deployability.
