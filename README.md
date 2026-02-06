# Self-Storage CMBS Analysis

Analysis of self-storage properties in the TREPP CMBS loan universe, focusing on:
1. **Vacancy** - Occupancy trends and prediction
2. **Revenue / NOI** - Cash flow analysis and growth patterns
3. **Distress Signals** - Watchlist, special servicing, delinquency, foreclosure

## Project Structure

```
self_storage_trepp/
├── scripts/
│   └── data_exploration.py     # Main data pull and exploration
├── src/
│   ├── __init__.py
│   ├── config.py               # Snowflake configuration
│   └── snowflake_io.py         # Database connection helpers
├── docs/
│   └── AI_GUIDE.md             # Context for AI agents
├── data/                       # Data files (gitignored)
├── output/
│   ├── figures/                # Charts and visualizations
│   ├── models/                 # Saved model artifacts
│   └── results/                # JSON results
├── .env                        # Snowflake credentials (gitignored)
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run data exploration
python scripts/data_exploration.py
```

## Data Source

- **TREPP CMBS Database** via Snowflake
- Tables: `RAW.ANALYSIS.TREPP_CMBS_LOAN` + `RAW.ANALYSIS.TREPP_CMBS_PROPERTY`
- Filter: `CREFCPROPERTYTYPE = 'Self Storage'` or `NORMALIZEDPROPERTYTYPE LIKE 'SS%'`

## Key Metrics

| Category | Metrics |
|----------|---------|
| **Vacancy** | Current occupancy, historical lags (1Y, 2Y), vacancy change, vacancy vs securitization |
| **Revenue/NOI** | Current NOI, NOI growth YoY, NOI vs securitization, revenue per unit, OpEx ratio, NOI margin |
| **DSCR** | Current DSCR, DSCR change, DSCR vs securitization, distressed flag (<1.0x) |
| **Distress** | Watchlist, special servicing, delinquency, modification, foreclosure, REO, composite score |
| **Property** | Units, sqft, year built, property condition, MSA, state |
| **Loan** | LTV, note rate, remaining term, maturity date, interest-only flag, debt yield |
