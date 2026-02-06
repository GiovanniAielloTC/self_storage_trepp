# AI Agent Guide: Self-Storage CMBS Analysis

**Purpose:** Context for AI agents working on this project.  
**Created:** February 5, 2026

---

## 1. Project Overview

### 1.1 Objective
Analyze self-storage properties in the TREPP CMBS loan universe across three dimensions:
1. **Vacancy** - Occupancy trends, predictions, and drivers
2. **Revenue / NOI** - Cash flow health, growth patterns, efficiency metrics
3. **Distress Signals** - Watchlist, special servicing, delinquency, foreclosure tracking

### 1.2 Why Self-Storage?
Self-storage is a distinct CRE asset class with unique characteristics:
- High NOI margins (~55-65% typical)
- Short "lease" terms (month-to-month)
- Lower capex vs multifamily
- Highly sensitive to local supply/demand
- Growing institutional interest
- Different vacancy dynamics than multifamily (the vacancy_pulse_trepp project focuses on MF)

---

## 2. Data Architecture

### 2.1 Source Tables (Snowflake)
```
Database: RAW
Schema:   ANALYSIS
Tables:
  - TREPP_CMBS_LOAN     (365 columns) - Loan-level financial data
  - TREPP_CMBS_PROPERTY  (295 columns) - Property-level characteristics
```

### 2.2 Connection
```python
# Uses .env file with:
SNOWFLAKE_USER=gaiello
SNOWFLAKE_ACCOUNT=wt74173.us-east-2.aws
SNOWFLAKE_WAREHOUSE=RESEARCH
SNOWFLAKE_DATABASE=RAW
SNOWFLAKE_SCHEMA=ANALYSIS
```

### 2.3 Self-Storage Filter
```sql
WHERE (
    p.CREFCPROPERTYTYPE = 'Self Storage'
    OR p.NORMALIZEDPROPERTYTYPE LIKE 'SS%'
    OR LOWER(p.PROPERTYTYPELONG) LIKE '%self%storage%'
    OR LOWER(p.PROPERTYTYPELONG) LIKE '%self-storage%'
)
```

### 2.4 Deduplication
Take latest month per property-year:
```sql
ROW_NUMBER() OVER (
    PARTITION BY p.TREPPMASTERPROPERTYID, l.YEAR 
    ORDER BY l.MONTH DESC
) AS rn
-- Then: WHERE rn = 1
```

---

## 3. Feature Catalog

### 3.1 Vacancy / Occupancy
| Feature | Source Column | Description |
|---------|-------------|-------------|
| OCCUPANCY_CURRENT | MOSTRECENTPHYSICALOCCUPANCY | Current occupancy % |
| VACANCY_RATE | 100 - occupancy | Current vacancy % |
| OCCUPANCY_LAG1 | PRECEDINGFISCALYEARPHYSICALOCCUPANCY | 1 year ago |
| OCCUPANCY_LAG2 | SECONDPRECEDINGFISCALYEARPHYSICALOCCUPANCY | 2 years ago |
| OCCUPANCY_AT_SEC | SECURITIZATIONOCCUPANCY | At loan securitization |
| VACANCY_CHANGE_1Y | Derived | YoY vacancy change |
| VACANCY_VS_SEC | Derived | Change from securitization |

### 3.2 Revenue / NOI
| Feature | Source Column | Description |
|---------|-------------|-------------|
| REVENUE_CURRENT | MOSTRECENTREVENUE | Current revenue |
| REVENUE_PRIOR | PRECEDINGFISCALYEARREVENUE | Prior year revenue |
| NOI_CURRENT | MOSTRECENTNOI | Current NOI |
| NOI_PRIOR | PRECEDINGFISCALYEARNOI | Prior year NOI |
| NOI_AT_SEC | SECURITIZATIONNOI | NOI at securitization |
| NCF_CURRENT | MOSTRECENTNCF | Net cash flow |
| OPEX_CURRENT | MOSTRECENTOPERATINGEXPENSES | Operating expenses |
| REVENUE_GROWTH_1Y | Derived | Revenue growth YoY % |
| NOI_GROWTH_1Y | Derived | NOI growth YoY % |
| NOI_VS_SEC | Derived | NOI change from securitization % |
| OPEX_RATIO | Derived | OpEx / Revenue % |
| NOI_MARGIN | Derived | NOI / Revenue % |
| REVENUE_PER_UNIT | Derived | Revenue / units |
| NOI_PER_UNIT | Derived | NOI / units |

### 3.3 Credit / Leverage
| Feature | Source Column | Description |
|---------|-------------|-------------|
| DSCR_CURRENT | MOSTRECENTDSCR_NOI | Current DSCR |
| DSCR_PRIOR | PRECEDINGFISCALYEARDSCR_NOI | Prior year DSCR |
| DSCR_AT_SEC | SECURITZATIONDSCR_NOI | DSCR at securitization |
| LTV_AT_SEC | SECURITIZATIONLTV | LTV at securitization |
| LTV_CURRENT | DERIVEDLTV | Current LTV |
| DEBT_YIELD_NOI | DERIVEDCURRENTDEBTYIELDNOI | Debt yield (NOI/debt) |
| DSCR_CHANGE_1Y | Derived | DSCR change YoY |
| IS_DSCR_DISTRESSED | Derived | 1 if DSCR < 1.0 |

### 3.4 Distress Signals
| Feature | Source Column | Description |
|---------|-------------|-------------|
| IS_WATCHLIST | WATCHLIST | On servicer watchlist |
| IS_SPECIAL_SERVICING | SPECIALSERVICE | In special servicing |
| IS_DELINQUENT | MONTHSDELINQUENT > 0 | Currently delinquent |
| MONTHS_DELINQUENT | MONTHSDELINQUENT | Months delinquent |
| IS_MODIFIED | MODIFICATIONINDICATOR | Loan modified |
| IS_FORECLOSURE | FORECLOSURESTARTDATE not null | In foreclosure |
| IS_REO | REODATE not null | Real estate owned |
| NEWLY_WATCHLIST | NEWLYONWATCHLIST | Just added to watchlist |
| NEWLY_SS | NEWLYSENTTOSPECIALSERVICING | Just sent to SS |
| DISTRESS_SCORE | Derived | Composite score (0-12) |

### 3.5 Property & Loan
| Feature | Description |
|---------|-------------|
| UNITS, SQFT | Property size |
| YEAR_BUILT, PROPERTY_AGE | Age |
| PROPERTY_CONDITION | Physical condition |
| LOAN_ORIGINAL_BALANCE | Loan size |
| NOTE_RATE, CURRENT_NOTE_RATE | Interest rate |
| REMAINING_TERM, MATURITY_DATE | Term |
| IS_INTEREST_ONLY | IO flag |
| IS_NEAR_MATURITY | < 24 months to maturity |

---

## 4. Distress Score Methodology

Composite distress score (0-12 scale):
```python
DISTRESS_SCORE = (
    IS_WATCHLIST * 1 +
    IS_SPECIAL_SERVICING * 2 +
    IS_DELINQUENT * 2 +
    IS_MODIFIED * 1 +
    IS_FORECLOSURE * 3 +
    IS_REO * 3 +
    IS_DSCR_DISTRESSED * 1
)
```

---

## 5. Relationship to vacancy_pulse_trepp

This project is the self-storage counterpart to the `vacancy_pulse_trepp` project:

| Aspect | vacancy_pulse_trepp | self_storage_trepp |
|--------|--------------------|--------------------|
| Property Type | Multifamily (MF) | Self Storage (SS) |
| Filter | `NORMALIZEDPROPERTYTYPE LIKE 'MF%'` | `CREFCPROPERTYTYPE = 'Self Storage'` |
| Focus | Vacancy prediction | Vacancy + Revenue/NOI + Distress |
| Models | OLS, LightGBM, CatBoost | TBD |
| Shared Code | src/config.py, src/snowflake_io.py | Same pattern |

---

## 6. Potential Analysis Directions

1. **Vacancy Prediction Model** - Port the vacancy_pulse approach to self-storage
2. **NOI Stress Testing** - Model NOI under different vacancy/rate scenarios
3. **Distress Early Warning** - Predict which loans will enter watchlist/SS
4. **Geographic Heat Maps** - Map vacancy and distress by MSA
5. **Supply/Demand Analysis** - Cross-reference with market-level supply data
6. **Vintage Analysis** - How do different origination vintages perform?
7. **Rate Sensitivity** - Impact of rising rates on SS DSCR

---

*This guide enables any AI agent to understand the project context and continue work.*
