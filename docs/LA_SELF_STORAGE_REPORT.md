# Self-Storage CMBS: Vacancy, Revenue & Distress Analysis
### Los Angeles Deep Dive — TerraCotta Research

**Date:** February 2026  
**Data Source:** Trepp CMBS (LOAN + PROPERTY tables)  
**Coverage:** 11,154 unique self-storage properties nationally | 101 active in LA (2025)  
**Data Freshness:** Most recent reporting through September 2025  

---

## Table of Contents

1. [National Landscape](#1-national-landscape)
2. [Los Angeles: Market Position](#2-los-angeles-market-position)
3. [LA 2025 Snapshot](#3-la-2025-snapshot)
4. [Operating Efficiency & OPEX Analysis](#4-operating-efficiency--opex-analysis)
5. [Distress Signals](#5-distress-signals)
6. [Interactive Map](#6-interactive-map)
7. [Data Notes](#7-data-notes)

---

## 1. National Landscape

The U.S. self-storage CMBS universe grew from ~70 properties in 1998 to over 3,600 by 2022, reflecting the sector's emergence as an institutional asset class. Today, approximately **3,357 properties** remain active in CMBS conduit and single-borrower deals.

![National Overview](output/figures/report_01_national_overview.png)

### Key National Metrics (2025)

| Metric | Value |
|--------|-------|
| Active properties | 3,357 |
| Average vacancy | 13.0% |
| Median vacancy | 12.0% |
| Median NOI | $6.97M |
| Median DSCR | 2.1x |
| On watchlist | 38.0% |
| Special servicing | 0.2% |
| Delinquent | 0.8% |

**Cycle history:** Vacancy peaked at **20.3%** during the GFC (2010), dropped to a historic low of **9.0%** in 2022 (post-COVID demand surge), and has since normalized back to **13.0%** in 2025. The current watchlist rate of 38% is notably elevated — largely driven by upcoming loan maturities and tighter refinancing conditions rather than operational distress.

---

## 2. Los Angeles: Market Position

LA consistently outperforms the national average on nearly every metric. The market benefits from high population density, constrained supply, and strong underlying demand for storage.

![LA vs National](output/figures/report_02_la_vs_national.png)

### LA vs National Side-by-Side (2025)

| Metric | Los Angeles | National | LA Advantage |
|--------|-------------|----------|--------------|
| Properties | 101 | 3,357 | — |
| Avg vacancy | **10.5%** | 13.0% | 2.5pp lower ✅ |
| Median vacancy | **10.0%** | 12.0% | 2.0pp lower ✅ |
| Median NOI | **$2.97M** | $6.97M | Smaller assets |
| Avg DSCR | **3.24x** | 2.3x | +0.94x stronger ✅ |
| Watchlist | **19.8%** | 38.0% | 18pp lower ✅ |
| Special servicing | **0.0%** | 0.2% | None ✅ |
| Delinquent | **0.0%** | 0.8% | None ✅ |

**Notable:** LA has zero properties in special servicing, delinquency, foreclosure, or REO status. The 20 watchlist-flagged properties (19.8%) are entirely from servicer monitoring — none have progressed to payment distress.

### Historical Context

LA's CMBS self-storage universe peaked at **227 properties** in 2009 and has steadily declined to 101 as loans matured or defeased. Over the full dataset history, **486 unique properties** have appeared in LA CMBS deals.

---

## 3. LA 2025 Snapshot

The 101 active LA properties show a concentrated, well-performing portfolio with a few pockets of elevated vacancy.

![LA Snapshot](output/figures/report_03_la_snapshot.png)

### Vacancy Distribution

| Bucket | Count | Share |
|--------|-------|-------|
| 0–5% | 18 | 17.8% |
| 5–10% | 32 | 31.7% |
| 10–15% | 24 | 23.8% |
| 15–20% | 18 | 17.8% |
| 20–30% | 6 | 5.9% |
| 30%+ | 3 | 3.0% |

Nearly **half** of LA properties have vacancy below 10%, and only 9 properties (8.9%) exceed 20% vacancy.

### Top Cities by Property Count

| City | Properties | Median Vacancy | Median NOI Margin |
|------|-----------|----------------|-------------------|
| Los Angeles | 14 | 11.0% | 69.0% |
| Westminster | 5 | 11.0% | 50.7% |
| Inglewood | 4 | 11.0% | 50.7% |
| Pasadena | 4 | 6.5% | 68.0% |
| North Hills | 3 | 11.0% | 50.7% |
| Lancaster | 3 | 7.0% | 61.7% |
| Van Nuys | 3 | 9.0% | 67.0% |
| Norwalk | 3 | 10.0% | 71.0% |
| Glendora | 3 | 6.0% | 44.4% |

**Pasadena** and **Lancaster** stand out with the lowest vacancy among multi-property cities. **Glendora** has low vacancy but also a lower NOI margin (44%), suggesting higher relative operating costs.

---

## 4. Operating Efficiency & OPEX Analysis

Self-storage is known for its capital-light, high-margin operating model. CMBS data confirms this — but reveals meaningful variation across properties and markets.

![OPEX & Efficiency](output/figures/report_04_opex_efficiency.png)

### Key Efficiency Metrics (2025)

| Metric | LA | National | Interpretation |
|--------|-----|---------|----------------|
| **OpEx Ratio** (OpEx/Revenue) | **34.6%** (median) | 37.1% | LA is leaner ✅ |
| **NOI Margin** (NOI/Revenue) | **65.4%** (median) | 62.9% | LA converts more revenue to NOI ✅ |
| **Revenue Growth** (YoY) | −22.4% (median) | −28.4% | Normalizing from 2022 peak |
| **OpEx Growth** (YoY) | −22.7% (median) | −26.5% | Costs declining with revenue |
| **OpEx vs Securitization** | +5.5% (median) | −9.2% | LA costs grew vs underwriting |

### What the OpEx Ratio Tells Us

The OpEx ratio measures how much of each revenue dollar goes to operating costs. For self-storage:

- **< 30%**: Highly efficient — likely newer, well-managed facilities
- **30–40%**: Typical — the healthy middle ground (33 LA properties)
- **40–50%**: Above average costs (22 LA properties)
- **> 50%**: Elevated — may signal deferred maintenance, older buildings, or management issues (9 LA properties)

**LA's OpEx Ratio Distribution:**

| Bucket | Count |
|--------|-------|
| < 20% | 7 |
| 20–30% | 30 |
| 30–40% | 33 |
| 40–50% | 22 |
| 50–60% | 7 |
| 60%+ | 2 |

### Vacancy × NOI Margin Relationship

The scatter plot (Figure 4, bottom-right) reveals a clear pattern: **higher vacancy erodes NOI margins**, but the relationship isn't perfectly linear. Some watchlist properties maintain decent margins despite elevated vacancy, while others with moderate vacancy show compressed margins — suggesting cost management issues rather than just occupancy problems.

### Revenue Decline Context

The −22% YoY revenue decline is **not a crisis** — it reflects normalization from the extraordinary 2021–2022 period when storage demand surged post-COVID and operators pushed rates aggressively. Current revenue levels remain well above pre-pandemic baselines for most properties.

---

## 5. Distress Signals

Despite elevated watchlist rates, LA's self-storage market shows remarkably low actual credit distress.

### LA Distress Summary (2025)

| Signal | Count | % of 101 | Assessment |
|--------|-------|----------|------------|
| On watchlist | 20 | 19.8% | ⚠️ Monitoring |
| Special servicing | 0 | 0.0% | ✅ Clean |
| Delinquent | 0 | 0.0% | ✅ Clean |
| Modified | 0 | 0.0% | ✅ Clean |
| Foreclosure | 0 | 0.0% | ✅ Clean |
| REO | 0 | 0.0% | ✅ Clean |

**Why watchlist but no distress?** Watchlist designation in CMBS doesn't necessarily mean payment problems. Common triggers for self-storage include:
- Upcoming loan maturity with refinancing uncertainty
- DSCR dipping below deal-specific thresholds (even though well above 1.0x)
- Revenue decline from prior peaks triggering servicer monitoring
- Insurance cost increases in fire-prone LA areas

The fact that **zero** LA properties have progressed from watchlist to actual payment distress (delinquency, special servicing, foreclosure) is a strong positive signal.

### Historical Distress Patterns

LA self-storage distress peaked during the GFC:
- **2010:** 17.8% vacancy, 13.8% on watchlist
- **2011–2012:** ~13% watchlist, ~16% vacancy

The sector recovered quickly — vacancy returned below 10% by 2015 and DSCR climbed from 1.6x to over 3.0x. The current cycle (post-2022 normalization) has pushed vacancy back to 10.5% but DSCR remains a robust 2.9x median, well above the GFC trough.

---

## 6. Interactive Map

The interactive map shows all 101 active LA self-storage properties with CMBS financing. Each marker encodes three dimensions:

- **Color** = Vacancy rate (green → red gradient)
- **Size** = Square footage of the facility
- **Shape** = ● Circle for healthy | ▲ Triangle for distressed (watchlist/SS/delinquent)

Click any marker for property details: vacancy, NOI, DSCR, loan balance, year built, and distress flags.

![Static Map](output/figures/report_05_la_map.png)

**👉 [Open Interactive Map](https://giovanniaiellotc.github.io/self_storage_trepp/)**

### Geographic Observations

- **Highest concentration:** Central LA, San Fernando Valley (Van Nuys, North Hills), and the 605/710 corridor (Norwalk, Whittier, Santa Fe Springs)
- **Lowest vacancy cluster:** Pasadena / San Gabriel Valley (median ~6.5%)
- **Watchlist clusters:** Scattered — no single sub-market dominates distress risk
- **Largest facilities** (by SQFT): tend to be in suburban/exurban locations (Lancaster, Palmdale)

---

## 7. Data Notes

### Source & Methodology

- **Database:** Trepp CMBS via Snowflake (`RAW.ANALYSIS.TREPP_CMBS_LOAN` + `TREPP_CMBS_PROPERTY`)
- **Filter:** `CREFCPROPERTYTYPE = 'Self Storage'` — verified 100% of properties
- **Deduplication:** `ROW_NUMBER()` over `(TREPPMASTERPROPERTYID, YEAR, MONTH)` to handle cross-collateralized loans
- **Time granularity:** Monthly reporting, though occupancy/financials typically update quarterly or annually

### Data Freshness

The 2025 snapshot uses each property's **most recent reporting month**. Report months represented:

| Month | Properties |
|-------|-----------|
| September | 65 |
| March | 20 |
| January | 5 |
| April | 5 |
| February | 2 |
| May | 2 |
| July | 1 |
| August | 1 |

Most data (65/101) is as of September 2025. Financial figures (NOI, Revenue, OpEx) are typically filed annually, so "MOSTRECENTNOI" may reflect full-year 2024 even in a September 2025 reporting month.

### SQFT Coverage

- 83/101 (82%) have `SQFT_CURRENT` — median 65,997 SF
- 92/101 (91%) have `SQFT_AT_SEC` as fallback
- 72/101 (71%) have `UNITS_CURRENT` — median 699 storage rooms
- Properties without SQFT data receive a default marker size on the map

### Historical Universe

486 unique LA self-storage properties have appeared in CMBS deals since 1998. The universe has shrunk from a peak of 227 (2009) to 101 (2025) as loans matured, defeased, or were refinanced outside CMBS. The current 101 represent the **actively-financed CMBS subset**, not the total LA self-storage market.

---

![Historical Trends](output/figures/report_06_la_historical.png)

---

*Report generated by TerraCotta Research — February 2026*  
*Data: Trepp CMBS | Analysis: Python (pandas, matplotlib, folium)*
