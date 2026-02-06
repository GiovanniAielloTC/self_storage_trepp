"""
demand_analysis.py — Self-Storage Demand & ML Prediction for 222 E 7th St
===========================================================================
Inspired by github.com/GiovanniAielloTC/LA_Multifamily_Demand_Analysis

Pipeline:
1. Pull national self-storage CMBS universe (2025 snapshot) from Snowflake
2. Join each property to its census tract via PROPERTY_CENSUS_TRACTS
3. Enrich with ACS demographics + LODES employment data
   - Job diversity index (Herfindahl), pct_jobs_young, sector shares
4. Train ML models:  Y = VACANCY_RATE  and  Y = NOI_CURRENT
   - OLS baseline (linear) vs GradientBoosting (non-linear)
   - SHAP feature importance to identify demand drivers
5. Predict vacancy & NOI for the proposed 222 E 7th St site trade area
6. Demand-score composite for investment decision support

Author: Giovanni Aiello — TerraCotta Capital
"""

import os
import sys
import warnings
from pathlib import Path
from decimal import Decimal

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── project paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
for d in [DATA_DIR, OUTPUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

# ── proposed site ──
SITE_LAT, SITE_LON = 34.0438, -118.2500
SITE_ADDRESS = "222 E 7th St, Downtown Los Angeles, CA 90014"
PRIMARY_RADIUS_MI = 3.0
SECONDARY_RADIUS_MI = 5.0

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 0 — Snowflake connection
# ═══════════════════════════════════════════════════════════════════════════
import snowflake.connector

def get_conn():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        login_timeout=60,
    )

# Shared connection (opened once, reused across queries)
_CONN = None

def _get_shared_conn():
    global _CONN
    if _CONN is None or _CONN.is_closed():
        _CONN = get_conn()
    return _CONN

def fetch_df(sql: str) -> pd.DataFrame:
    conn = _get_shared_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        # Decimal → float
        for c in df.columns:
            if df[c].dtype == object and len(df) > 0 and isinstance(df[c].iloc[0], Decimal):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        # If connection went stale, retry once
        print(f"  ⚠ Query failed ({e}), retrying with fresh connection...")
        _CONN = None
        conn = _get_shared_conn()
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Pull CMBS Self-Storage universe (latest observation per property)
# ═══════════════════════════════════════════════════════════════════════════
def pull_cmbs_universe():
    """Pulls the most-recent observation for each self-storage property."""
    print("Pulling CMBS self-storage universe...")
    sql = """
    WITH ranked AS (
        SELECT
            p.TREPPMASTERPROPERTYID,
            p.PROPNAME                    AS PROPERTY_NAME,
            p.ADDRESS,
            p.CITY,
            p.STATE,
            p.ZIPCODE,
            p.CREFCPROPERTYTYPE,
            TRY_CAST(p.YEARBUILT AS INT)  AS YEAR_BUILT,
            TRY_CAST(p.UNITS AS INT)      AS UNITS,
            COALESCE(
                TRY_CAST(p.CURRENTNETRENTABLESQUAREFEET AS FLOAT),
                TRY_CAST(p.NETSQUAREFEETATSECURITIZATION AS FLOAT)
            )                             AS SQFT,
            TRY_CAST(l.MOSTRECENTPHYSICALOCCUPANCY AS FLOAT)  AS OCCUPANCY_CURRENT,
            100 - TRY_CAST(l.MOSTRECENTPHYSICALOCCUPANCY AS FLOAT)  AS VACANCY_RATE,
            TRY_CAST(l.MOSTRECENTNOI AS FLOAT)                AS NOI_CURRENT,
            TRY_CAST(l.MOSTRECENTDSCR_NOI AS FLOAT)           AS DSCR_CURRENT,
            TRY_CAST(l.ACTUALBALANCE AS FLOAT)                AS LOAN_CURRENT_BALANCE,
            TRY_CAST(l.MOSTRECENTREVENUE AS FLOAT)            AS REVENUE_CURRENT,
            TRY_CAST(l.MOSTRECENTOPERATINGEXPENSES AS FLOAT)  AS OPEX_CURRENT,
            CASE WHEN UPPER(l.WATCHLIST) = 'Y' THEN 1 ELSE 0 END           AS IS_WATCHLIST,
            CASE WHEN UPPER(l.SPECIALSERVICE) = 'Y' THEN 1 ELSE 0 END      AS IS_SPECIAL_SERVICING,
            CASE WHEN TRY_CAST(l.MONTHSDELINQUENT AS INT) > 0 THEN 1 ELSE 0 END AS IS_DELINQUENT,
            CAST(l.YEAR AS INT) AS REPORT_YEAR,
            CAST(l.MONTH AS INT) AS REPORT_MONTH,
            ROW_NUMBER() OVER (
                PARTITION BY p.TREPPMASTERPROPERTYID
                ORDER BY l.YEAR DESC, l.MONTH DESC
            ) AS rn
        FROM RAW.ANALYSIS.TREPP_CMBS_LOAN l
        JOIN RAW.ANALYSIS.TREPP_CMBS_PROPERTY p
            ON l.TREPPMASTERLOANID = p.TREPPMASTERLOANID
            AND l.YEAR = p.YEAR
            AND l.MONTH = p.MONTH
        WHERE p.CREFCPROPERTYTYPE = 'Self Storage'
          AND l.YEAR >= 2024
    )
    SELECT * FROM ranked WHERE rn = 1
    """
    df = fetch_df(sql)
    df.drop(columns=["RN"], inplace=True, errors="ignore")

    # Convert Decimal columns
    num_cols = ["OCCUPANCY_CURRENT", "VACANCY_RATE", "NOI_CURRENT", "DSCR_CURRENT",
                "LOAN_CURRENT_BALANCE", "REVENUE_CURRENT", "OPEX_CURRENT", "SQFT", "UNITS"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived features
    df["OPEX_RATIO"] = np.where(
        df["REVENUE_CURRENT"] > 0,
        df["OPEX_CURRENT"] / df["REVENUE_CURRENT"] * 100, np.nan
    )
    df["NOI_PER_SQFT"] = np.where(df["SQFT"] > 0, df["NOI_CURRENT"] / df["SQFT"], np.nan)
    df["REV_PER_SQFT"] = np.where(df["SQFT"] > 0, df["REVENUE_CURRENT"] / df["SQFT"], np.nan)

    # Distress score
    df["DISTRESS_SCORE"] = (
        df["IS_WATCHLIST"].astype(int) +
        df["IS_SPECIAL_SERVICING"].astype(int) * 2 +
        df["IS_DELINQUENT"].astype(int) * 2
    )

    print(f"  → {len(df)} properties pulled")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Map properties to census tracts (via ZIP→Tract crosswalk)
# ═══════════════════════════════════════════════════════════════════════════
def join_property_tracts(cmbs: pd.DataFrame) -> pd.DataFrame:
    """
    Map CMBS properties to census tracts using the LODES crosswalk.
    PROPERTY_CENSUS_TRACTS is CoStar-only; Trepp IDs don't match.
    Instead we use CENSUS_LODES_XWALK_2022 which has ZCTA→TRACT.
    For each ZIP, we pick the most-populated tract as the representative.
    Also pull tract centroids for lat/lon.
    """
    print("Mapping properties to census tracts via ZIP crosswalk...")

    # Build ZIP→Tract mapping from LODES crosswalk (pick primary tract per ZIP)
    sql_xwalk = """
    SELECT
        ZCTA                          AS ZIPCODE,
        SUBSTR(TABBLK2020, 1, 11)     AS CENSUS_TRACT_GEOID,
        COUNT(*)                       AS BLOCK_COUNT
    FROM RAW.ANALYSIS.CENSUS_LODES_XWALK_2022
    WHERE ZCTA IS NOT NULL
    GROUP BY ZCTA, SUBSTR(TABBLK2020, 1, 11)
    ORDER BY ZCTA, BLOCK_COUNT DESC
    """
    xwalk = fetch_df(sql_xwalk)
    # Keep the tract with the most blocks per ZIP (dominant tract)
    xwalk = xwalk.sort_values(["ZIPCODE", "BLOCK_COUNT"], ascending=[True, False])
    xwalk = xwalk.drop_duplicates(subset=["ZIPCODE"], keep="first")
    print(f"  ZIP→Tract crosswalk: {len(xwalk)} ZIPs")

    # Get tract centroids for lat/lon
    sql_centroids = """
    SELECT
        LPAD(STATEFP, 2, '0') || LPAD(COUNTYFP, 3, '0') || LPAD(TRACTCE, 6, '0')
            AS CENSUS_TRACT_GEOID,
        LATITUDE   AS PROP_LAT,
        LONGITUDE  AS PROP_LON
    FROM RAW.ANALYSIS.CENSUS_TRACT_POP_CENTROIDS
    """
    centroids = fetch_df(sql_centroids)
    print(f"  Tract centroids: {len(centroids)} tracts")

    # Merge CMBS → ZIP → Tract
    cmbs["ZIPCODE"] = cmbs["ZIPCODE"].astype(str).str.strip().str[:5]
    merged = cmbs.merge(xwalk[["ZIPCODE", "CENSUS_TRACT_GEOID"]], on="ZIPCODE", how="left")
    tract_match = merged["CENSUS_TRACT_GEOID"].notna().sum()
    print(f"  Tract match via ZIP: {tract_match}/{len(merged)} ({tract_match/len(merged)*100:.0f}%)")

    # Add centroid coordinates
    merged = merged.merge(centroids, on="CENSUS_TRACT_GEOID", how="left")
    coord_match = merged["PROP_LAT"].notna().sum()
    print(f"  With coordinates: {coord_match}/{len(merged)} ({coord_match/len(merged)*100:.0f}%)")

    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Enrich with ACS demographics
# ═══════════════════════════════════════════════════════════════════════════

# ACS columns we want (demand-relevant for self-storage)
ACS_FEATURES = [
    "CENSUS_ID",
    "POPULATION",
    "TOT_HH",
    "POP_DENS",
    "LAND_AREA_METERS",
    "POP_MHI",                   # Median household income
    "RENTER_MHI",                # Renter median household income
    "OWNER_MHI",                 # Owner median household income
    "MED_RENT",                  # Median rent
    "RENTER_PCT",                # % renters
    "OWNER_PCT",                 # % owners
    "RENTER_AVG_HH_SIZE",
    "POP_MEDIAN_AGE",
    "POP_POVERTY_PCT",
    "POP_COLLEGE_PCT",           # % with college education
    "POP_HS_GRAD_PCT",
    "POP_0_CAR_PCT",             # % zero-car households
    "POP_1_CAR_PCT",
    "POP_2_CAR_PCT",
    "POP_3_CAR_PCT",
    "POP_PUBLIC_TRANSIT_PCT",
    "POP_DROVE_CAR_PCT",
    "MEDIAN_HOME_VALUE",
    "MED_HOUSE_VAL",
    "RENT_TO_HOUSE_VAL",
    "POP_FAMILY_HH_PCT",
    "POP_SINGLE_MOM_PCT",
    "POP_RETIREE_PCT",
    "POP_OCC_TOTAL_EMPLOYED",
    "POP_AGE_25_29_PCT",
    "POP_AGE_30_34_PCT",
    "POP_AGE_35_39_PCT",
    "POP_AGE_65_66_PCT",
    "POP_AGE_67_69_PCT",
    "POP_AGE_70_74_PCT",
    "RENTER_RENT_50PLUS_PCT",    # % rent-burdened (>50% of income)
    "RENTER_RENT_30_35PCT",
]

def pull_acs_demographics() -> pd.DataFrame:
    """Pulls ACS tract-level demographics."""
    print("Pulling ACS demographics (tract level)...")
    cols_sql = ",\n    ".join(ACS_FEATURES)
    sql = f"SELECT {cols_sql} FROM RAW.ANALYSIS.ACS_CENSUS_TRACT_2023_CLEAN"
    df = fetch_df(sql)
    print(f"  → {len(df)} tracts pulled, {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Enrich with LODES employment data
# ═══════════════════════════════════════════════════════════════════════════
def pull_lodes_employment() -> pd.DataFrame:
    """
    Aggregates LODES WAC (Workplace Area Characteristics) from block to tract level.
    C000 = total jobs, CNS07 = retail, CNS18 = services, etc.
    """
    print("Pulling LODES employment data (aggregated to tract)...")
    sql = """
    SELECT
        LEFT(w.W_GEOCODE, 11)  AS TRACT_GEOID,
        SUM(w.C000)            AS TOTAL_JOBS,
        SUM(w.CA01)            AS JOBS_AGE_29_OR_LESS,
        SUM(w.CA02)            AS JOBS_AGE_30_54,
        SUM(w.CA03)            AS JOBS_AGE_55_PLUS,
        SUM(w.CE01)            AS JOBS_EARN_1250_OR_LESS,
        SUM(w.CE02)            AS JOBS_EARN_1251_3333,
        SUM(w.CE03)            AS JOBS_EARN_3334_PLUS,
        SUM(w.CNS01)           AS JOBS_AGRICULTURE,
        SUM(w.CNS04)           AS JOBS_CONSTRUCTION,
        SUM(w.CNS05)           AS JOBS_MANUFACTURING,
        SUM(w.CNS07)           AS JOBS_RETAIL,
        SUM(w.CNS08)           AS JOBS_TRANSPORT_WAREHOUSE,
        SUM(w.CNS09)           AS JOBS_INFORMATION,
        SUM(w.CNS10)           AS JOBS_FINANCE,
        SUM(w.CNS11)           AS JOBS_REAL_ESTATE,
        SUM(w.CNS12)           AS JOBS_PROFESSIONAL,
        SUM(w.CNS13)           AS JOBS_MANAGEMENT,
        SUM(w.CNS15)           AS JOBS_EDUCATION,
        SUM(w.CNS16)           AS JOBS_HEALTHCARE,
        SUM(w.CNS17)           AS JOBS_ENTERTAINMENT,
        SUM(w.CNS18)           AS JOBS_FOOD_SERVICES,
        SUM(w.CNS20)           AS JOBS_GOVERNMENT
    FROM RAW.ANALYSIS.CENSUS_LODES_WAC_2022 w
    GROUP BY LEFT(w.W_GEOCODE, 11)
    """
    df = fetch_df(sql)
    # Jobs density: compute ratios
    for c in df.columns:
        if c != "TRACT_GEOID":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["JOBS_HIGH_EARN_PCT"] = np.where(
        df["TOTAL_JOBS"] > 0,
        df["JOBS_EARN_3334_PLUS"] / df["TOTAL_JOBS"] * 100, np.nan
    )
    df["JOBS_RETAIL_PCT"] = np.where(
        df["TOTAL_JOBS"] > 0,
        df["JOBS_RETAIL"] / df["TOTAL_JOBS"] * 100, np.nan
    )
    df["PCT_JOBS_YOUNG"] = np.where(
        df["TOTAL_JOBS"] > 0,
        df["JOBS_AGE_29_OR_LESS"] / df["TOTAL_JOBS"] * 100, np.nan
    )
    df["PCT_JOBS_PROFESSIONAL"] = np.where(
        df["TOTAL_JOBS"] > 0,
        df["JOBS_PROFESSIONAL"] / df["TOTAL_JOBS"] * 100, np.nan
    )

    # Herfindahl-style job diversity index (from multifamily repo)
    # Higher = more diverse local economy (max ≈ 0.95), lower = concentrated
    naics_cols = [c for c in df.columns
                  if c.startswith("JOBS_") and c not in (
                      "JOBS_HIGH_EARN_PCT", "JOBS_RETAIL_PCT", "JOBS_PER_HH",
                      "PCT_JOBS_YOUNG", "PCT_JOBS_PROFESSIONAL",
                      "JOBS_EARN_1250_OR_LESS", "JOBS_EARN_1251_3333", "JOBS_EARN_3334_PLUS",
                      "JOBS_AGE_29_OR_LESS", "JOBS_AGE_30_54", "JOBS_AGE_55_PLUS",
                  ) and c != "TOTAL_JOBS"]
    shares = df[naics_cols].div(df["TOTAL_JOBS"].replace(0, np.nan), axis=0)
    df["JOB_DIVERSITY_INDEX"] = 1 - (shares ** 2).sum(axis=1)

    print(f"  → {len(df)} tracts with employment data")
    print(f"    Job diversity index range: {df['JOB_DIVERSITY_INDEX'].min():.3f} – "
          f"{df['JOB_DIVERSITY_INDEX'].max():.3f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5 — Merge everything
# ═══════════════════════════════════════════════════════════════════════════
def build_ml_dataset(cmbs_tracts, acs, lodes):
    """Merge CMBS properties + ACS + LODES into a single ML-ready dataset."""
    print("\nBuilding ML dataset...")

    # Merge ACS
    df = cmbs_tracts.merge(acs, left_on="CENSUS_TRACT_GEOID", right_on="CENSUS_ID", how="left")
    acs_match = df["CENSUS_ID"].notna().sum()
    print(f"  ACS match: {acs_match}/{len(df)} ({acs_match/len(df)*100:.0f}%)")

    # Merge LODES
    df = df.merge(lodes, left_on="CENSUS_TRACT_GEOID", right_on="TRACT_GEOID", how="left")
    lodes_match = df["TRACT_GEOID"].notna().sum()
    print(f"  LODES match: {lodes_match}/{len(df)} ({lodes_match/len(df)*100:.0f}%)")

    # Compute derived features
    df["JOBS_PER_HH"] = np.where(df["TOT_HH"] > 0, df["TOTAL_JOBS"] / df["TOT_HH"], np.nan)
    df["PROPERTY_AGE"] = np.where(
        df["YEAR_BUILT"].notna() & (df["YEAR_BUILT"] > 1900),
        2025 - df["YEAR_BUILT"], np.nan
    )
    df["LTV"] = np.where(
        (df["NOI_CURRENT"] > 0) & (df["LOAN_CURRENT_BALANCE"] > 0),
        df["LOAN_CURRENT_BALANCE"] / (df["NOI_CURRENT"] / 0.065),  # implied cap 6.5%
        np.nan
    )
    df["SQFT_PER_UNIT"] = np.where(
        df["UNITS"] > 0, df["SQFT"] / df["UNITS"], np.nan
    )

    print(f"  Final dataset: {len(df)} rows × {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6 — ML Models: OLS baseline + CatBoost  (matches MF vacancy repo)
# ═══════════════════════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor, Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── Feature taxonomy ─────────────────────────────────────────────────────
# EXOGENOUS (what you can control / observe at site selection time)
#   Building features:  physical property characteristics
#   Demographic features: census-tract level ACS
#   Employment features:  LODES workplace data
# ENDOGENOUS (financial outcomes — correlated with vacancy, not causal)
#   OPEX_RATIO, DSCR, LOAN_BALANCE  ← these dominated the old model (90%+)
#   but they are OUTCOMES, not CAUSES.  "High OPEX ↔ High Vacancy" is
#   simultaneity bias, not actionable insight.
# ─────────────────────────────────────────────────────────────────────────

BUILDING_FEATURES = [
    "SQFT", "UNITS", "PROPERTY_AGE", "SQFT_PER_UNIT",
]

DEMOGRAPHIC_FEATURES = [
    "POPULATION", "POP_DENS", "TOT_HH", "POP_MHI", "RENTER_MHI",
    "MED_RENT", "RENTER_PCT", "POP_MEDIAN_AGE", "POP_POVERTY_PCT",
    "POP_COLLEGE_PCT", "POP_0_CAR_PCT", "POP_PUBLIC_TRANSIT_PCT",
    "MEDIAN_HOME_VALUE", "RENT_TO_HOUSE_VAL",
    "POP_FAMILY_HH_PCT", "POP_RETIREE_PCT",
    "POP_AGE_25_29_PCT", "POP_AGE_30_34_PCT", "POP_AGE_35_39_PCT",
    "RENTER_RENT_50PLUS_PCT",
]

EMPLOYMENT_FEATURES = [
    "TOTAL_JOBS", "JOBS_PER_HH", "JOBS_HIGH_EARN_PCT", "JOBS_RETAIL_PCT",
    "JOBS_RETAIL", "JOBS_TRANSPORT_WAREHOUSE", "JOBS_PROFESSIONAL",
    "JOBS_FOOD_SERVICES", "JOB_DIVERSITY_INDEX", "PCT_JOBS_YOUNG",
    "PCT_JOBS_PROFESSIONAL",
]

FINANCIAL_FEATURES = [
    "DSCR_CURRENT", "LOAN_CURRENT_BALANCE", "OPEX_RATIO",
]

# Grouped feature sets
EXOGENOUS_FEATURES = BUILDING_FEATURES + DEMOGRAPHIC_FEATURES + EMPLOYMENT_FEATURES
ALL_FEATURES = EXOGENOUS_FEATURES + FINANCIAL_FEATURES


def train_models(df, target: str, model_name: str, feature_set: list = None):
    """
    Train OLS (linear baseline) + CatBoost (non-linear) side-by-side.
    Matches methodology from LA_Multifamily_Demand_Analysis.

    feature_set: list of feature names. If None, uses ALL_FEATURES.
    Returns the CatBoost model (better for prediction) + metadata.
    """
    features = feature_set if feature_set is not None else ALL_FEATURES
    if target == "NOI_CURRENT":
        features = [f for f in features if f != "OPEX_RATIO"]  # avoid leakage

    # Filter to usable rows
    avail = [f for f in features if f in df.columns]
    subset = df.dropna(subset=[target])
    subset = subset[avail + [target]].dropna()

    if len(subset) < 100:
        print(f"  ⚠ Only {len(subset)} rows available for {target} — too few.")
        return None, None, None

    X = subset[avail].copy()
    y = subset[target].values.copy()

    # Log-transform NOI (keep positives only)
    log_target = False
    if target == "NOI_CURRENT":
        mask = y > 0
        X, y = X[mask].reset_index(drop=True), y[mask]
        y = np.log1p(y)
        log_target = True

    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"  Target: {target}  |  Samples: {len(y)}  |  Features: {len(avail)}")
    print(f"{'='*70}")

    X_arr = X.values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── OLS baseline ─────────────────────────────────────────────────────
    ols = LinearRegression()
    ols_r2 = cross_val_score(ols, X_arr, y, cv=kf, scoring="r2")
    ols_mae = -cross_val_score(ols, X_arr, y, cv=kf, scoring="neg_mean_absolute_error")
    ols.fit(X_arr, y)
    ols_train_r2 = r2_score(y, ols.predict(X_arr))

    print(f"\n  ┌─ OLS (Linear Baseline) ──────────────────────────────┐")
    print(f"  │  Train R²:  {ols_train_r2*100:6.2f}%                          │")
    print(f"  │  CV R²:     {ols_r2.mean()*100:6.2f}% ± {ols_r2.std()*100:.2f}%                  │")
    if log_target:
        print(f"  │  CV MAE:    {ols_mae.mean():.3f} (log-scale)                │")
    else:
        print(f"  │  CV MAE:    {ols_mae.mean():.2f}                             │")
    print(f"  └────────────────────────────────────────────────────┘")

    # OLS coefficients (top drivers)
    ols_coefs = pd.DataFrame({
        "feature": avail,
        "coef": ols.coef_,
        "abs_coef": np.abs(ols.coef_),
    }).sort_values("abs_coef", ascending=False)
    print(f"\n  OLS Top Coefficients:")
    for _, row in ols_coefs.head(10).iterrows():
        sign = "+" if row["coef"] > 0 else "−"
        print(f"    {sign} {row['feature']:30s} {row['coef']:+.6f}")

    # ── CatBoost ─────────────────────────────────────────────────────────
    cb = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
        loss_function="RMSE",
    )
    cb_r2 = cross_val_score(cb, X_arr, y, cv=kf, scoring="r2")
    cb_mae = -cross_val_score(cb, X_arr, y, cv=kf, scoring="neg_mean_absolute_error")
    cb.fit(Pool(X_arr, y))
    cb_train_r2 = r2_score(y, cb.predict(X_arr))

    print(f"\n  ┌─ CatBoost (Non-Linear) ──────────────────────────────┐")
    print(f"  │  Train R²:  {cb_train_r2*100:6.2f}%                          │")
    print(f"  │  CV R²:     {cb_r2.mean()*100:6.2f}% ± {cb_r2.std()*100:.2f}%                  │")
    if log_target:
        print(f"  │  CV MAE:    {cb_mae.mean():.3f} (log-scale)                │")
    else:
        print(f"  │  CV MAE:    {cb_mae.mean():.2f}                             │")
    print(f"  └────────────────────────────────────────────────────┘")

    # Delta
    delta = (cb_r2.mean() - ols_r2.mean()) * 100
    print(f"\n  CatBoost vs OLS:  {delta:+.2f} pp R² improvement")

    # ── CatBoost feature importance ──────────────────────────────────────
    cb_imp = pd.DataFrame({
        "feature": avail,
        "importance": cb.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    print(f"\n  CatBoost Feature Importance:")
    for _, row in cb_imp.head(15).iterrows():
        bar = "█" * int(row["importance"] / cb_imp["importance"].max() * 30)
        print(f"    {row['feature']:30s} {bar} {row['importance']:.1f}%")

    # ── SHAP beeswarm ────────────────────────────────────────────────────
    print(f"\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(cb)
    shap_values = explainer.shap_values(X_arr)
    shap_imp = pd.DataFrame({
        "feature": avail,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    safe_name = model_name.lower().replace(" ", "_").replace(":", "")

    # SHAP beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_arr, feature_names=avail, show=False, max_display=20)
    plt.title(f"SHAP — {model_name} (CatBoost)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / f"shap_{safe_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # Feature importance bar chart (cleaner for presentations)
    fig, ax = plt.subplots(figsize=(10, 7))
    top = cb_imp.head(15).iloc[::-1]  # reverse for horizontal bar
    colors = ["#2ecc71" if f in DEMOGRAPHIC_FEATURES
              else "#e74c3c" if f in EMPLOYMENT_FEATURES
              else "#e67e22" if f in FINANCIAL_FEATURES
              else "#3498db" for f in top["feature"]]
    ax.barh(top["feature"], top["importance"], color=colors, alpha=0.85)
    ax.set_xlabel("Feature Importance (%)", fontsize=12)
    ax.set_title(f"CatBoost Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.85, label="Building"),
        Patch(facecolor="#2ecc71", alpha=0.85, label="Demographic"),
        Patch(facecolor="#e74c3c", alpha=0.85, label="Employment"),
        Patch(facecolor="#e67e22", alpha=0.85, label="Financial (endogenous)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    fig_path2 = FIG_DIR / f"feature_importance_{safe_name}.png"
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path2}")

    # ── Summary comparison table ─────────────────────────────────────────
    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  MODEL COMPARISON — {model_name:32s}│")
    print(f"  ├──────────────┬───────────┬───────────┬──────────────┤")
    print(f"  │  Model        │ Train R²  │ CV R²     │ CV MAE       │")
    print(f"  ├──────────────┼───────────┼───────────┼──────────────┤")
    print(f"  │  OLS          │ {ols_train_r2*100:6.2f}%   │ {ols_r2.mean()*100:6.2f}%   │ {ols_mae.mean():12.3f} │")
    print(f"  │  CatBoost     │ {cb_train_r2*100:6.2f}%   │ {cb_r2.mean()*100:6.2f}%   │ {cb_mae.mean():12.3f} │")
    print(f"  └──────────────┴───────────┴───────────┴──────────────┘")

    return cb, avail, log_target


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7 — 222 E 7th St Prediction
# ═══════════════════════════════════════════════════════════════════════════
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def get_trade_area_features(df, site_lat, site_lon, radius_mi):
    """
    Compute average demographic features for tracts whose properties
    fall within `radius_mi` of the proposed site.
    Also compute the direct trade-area demographics from nearby tracts.
    """
    # Find properties within radius
    df["_dist"] = haversine_miles(site_lat, site_lon, df["PROP_LAT"], df["PROP_LON"])
    nearby = df[df["_dist"] <= radius_mi].copy()
    df.drop(columns=["_dist"], inplace=True)

    if len(nearby) == 0:
        print(f"  ⚠ No properties within {radius_mi} mi — expanding to 10 mi")
        df["_dist"] = haversine_miles(site_lat, site_lon, df["PROP_LAT"], df["PROP_LON"])
        nearby = df[df["_dist"] <= 10].copy()
        df.drop(columns=["_dist"], inplace=True)

    return nearby


def predict_for_site(model, feature_names, log_target, trade_area_df):
    """Build a feature vector from the trade area and predict."""
    # Use median of the trade-area features as the "site" profile
    feature_vec = []
    for f in feature_names:
        if f in trade_area_df.columns:
            col = pd.to_numeric(trade_area_df[f], errors="coerce")
            val = col.median()
        else:
            val = 0
        feature_vec.append(val if pd.notna(val) else 0)

    X_site = np.array(feature_vec).reshape(1, -1)
    pred = model.predict(X_site)[0]
    if log_target:
        pred = np.expm1(pred)
    return pred, dict(zip(feature_names, feature_vec))


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 8 — Competitive Landscape Summary
# ═══════════════════════════════════════════════════════════════════════════
def competitive_analysis(df):
    """Print competitive analysis around the proposed site."""
    df["dist_to_site"] = haversine_miles(
        SITE_LAT, SITE_LON, df["PROP_LAT"].values, df["PROP_LON"].values
    )
    df_sorted = df.sort_values("dist_to_site")

    print("\n" + "=" * 70)
    print("  COMPETITIVE LANDSCAPE: 222 E 7th St, DTLA")
    print("=" * 70)

    for label, radius in [("3-MILE PRIMARY", PRIMARY_RADIUS_MI),
                           ("5-MILE SECONDARY", SECONDARY_RADIUS_MI)]:
        ring = df_sorted[df_sorted["dist_to_site"] <= radius]
        print(f"\n  {label} TRADE AREA ({radius} mi)")
        print(f"  {'─'*50}")
        print(f"    Properties:       {len(ring)}")
        if len(ring) > 0:
            print(f"    Total SQFT:       {ring['SQFT'].sum():,.0f}")
            print(f"    Avg Vacancy:      {ring['VACANCY_RATE'].mean():.1f}%")
            print(f"    Avg Occupancy:    {ring['OCCUPANCY_CURRENT'].mean():.1f}%")
            print(f"    Avg NOI:          ${ring['NOI_CURRENT'].mean():,.0f}")
            print(f"    Avg DSCR:         {ring['DSCR_CURRENT'].mean():.2f}")
            print(f"    Distressed:       {(ring['DISTRESS_SCORE']>=3).sum()}/{len(ring)}")
            print()
            for _, r in ring.iterrows():
                dist = r["dist_to_site"]
                print(f"    [{dist:.1f} mi]  {r['PROPERTY_NAME']}, {r['CITY']}")
                print(f"              SQFT={r['SQFT']:,.0f}  Vacancy={r['VACANCY_RATE']:.0f}%"
                      f"  NOI=${r['NOI_CURRENT']:,.0f}  DSCR={r['DSCR_CURRENT']:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # 1. Pull CMBS universe
    cmbs = pull_cmbs_universe()

    # 2. Join to tracts
    cmbs_tracts = join_property_tracts(cmbs)

    # 3. Pull demographics
    acs = pull_acs_demographics()
    lodes = pull_lodes_employment()

    # 4. Build ML dataset
    ml_df = build_ml_dataset(cmbs_tracts, acs, lodes)

    # Save enriched dataset
    save_path = DATA_DIR / "ml_dataset_national.csv"
    ml_df.to_csv(save_path, index=False)
    print(f"\nSaved ML dataset: {save_path}")

    # 5. Competitive analysis (only for properties with coordinates)
    has_coords = ml_df.dropna(subset=["PROP_LAT", "PROP_LON"])
    competitive_analysis(has_coords)

    # 6. Train models — TWO VARIANTS per the MF vacancy repo pattern:
    #    (A) Exogenous-only: building + demographics + employment
    #        → answers "if the building is X, in an area like Y, expect Z vacancy"
    #        → avoids simultaneity bias, provides causal / actionable insight
    #    (B) Full model: adds endogenous financial metrics (DSCR, LOAN_BALANCE, OPEX)
    #        → higher R² but not causal; useful for nowcasting
    print("\n\n" + "█" * 70)
    print("  TRAINING ML MODELS — OLS + CatBoost")
    print("█" * 70)

    # ── (A) EXOGENOUS-ONLY (causal / actionable) ────────────────────────
    print("\n\n" + "─" * 70)
    print("  MODEL VARIANT A: EXOGENOUS ONLY (Building + Location)")
    print("  → Avoids simultaneity bias: NO financial metrics")
    print("  → Use this to answer: 'if the building is like X, then...'")
    print("─" * 70)

    vac_model_exog, vac_feat_exog, vac_log_exog = train_models(
        ml_df, "VACANCY_RATE", "Vacancy (Exogenous Only)",
        feature_set=EXOGENOUS_FEATURES,
    )
    noi_model_exog, noi_feat_exog, noi_log_exog = train_models(
        ml_df, "NOI_CURRENT", "NOI (Exogenous Only)",
        feature_set=EXOGENOUS_FEATURES,
    )

    # ── (B) FULL MODEL (including financials — higher R² but endogenous) ─
    print("\n\n" + "─" * 70)
    print("  MODEL VARIANT B: FULL (includes financial metrics)")
    print("  → Higher R² but OPEX/DSCR are outcomes, not causes")
    print("  → Use for nowcasting / benchmarking, not causal insight")
    print("─" * 70)

    vac_model_full, vac_feat_full, vac_log_full = train_models(
        ml_df, "VACANCY_RATE", "Vacancy (Full Model)",
        feature_set=ALL_FEATURES,
    )
    noi_model_full, noi_feat_full, noi_log_full = train_models(
        ml_df, "NOI_CURRENT", "NOI (Full Model)",
        feature_set=ALL_FEATURES,
    )

    # 7. Predict for 222 E 7th St  — use EXOGENOUS model (causal)
    #    The exogenous model is the one that answers the user's question:
    #    "if the building is like this, in an area like that → expect X"
    vac_model = vac_model_exog
    vac_features = vac_feat_exog
    vac_log = vac_log_exog
    noi_model = noi_model_exog
    noi_features = noi_feat_exog
    noi_log = noi_log_exog

    # 7. Predict for 222 E 7th St
    if vac_model and noi_model:
        print("\n\n" + "★" * 70)
        print(f"  PREDICTION FOR: {SITE_ADDRESS}")
        print("★" * 70)

        # Get trade-area features (use 5-mile radius for more data)
        trade_area = get_trade_area_features(
            has_coords.copy(), SITE_LAT, SITE_LON, SECONDARY_RADIUS_MI
        )
        if len(trade_area) == 0:
            trade_area = get_trade_area_features(
                has_coords.copy(), SITE_LAT, SITE_LON, 10
            )

        print(f"\n  Trade area properties used: {len(trade_area)}")
        print(f"  Trade area tracts: {trade_area['CENSUS_TRACT_GEOID'].nunique()}")

        # Vacancy prediction
        vac_pred, vac_profile = predict_for_site(vac_model, vac_features, vac_log, trade_area)
        print(f"\n  ┌───────────────────────────────────────────┐")
        print(f"  │  PREDICTED VACANCY RATE:  {vac_pred:6.1f}%          │")
        print(f"  │  (trade-area avg:         {trade_area['VACANCY_RATE'].mean():6.1f}%)         │")
        print(f"  └───────────────────────────────────────────┘")

        # NOI prediction
        noi_pred, noi_profile = predict_for_site(noi_model, noi_features, noi_log, trade_area)
        print(f"\n  ┌───────────────────────────────────────────┐")
        print(f"  │  PREDICTED NOI:  ${noi_pred:>14,.0f}           │")
        print(f"  │  (trade-area avg: ${trade_area['NOI_CURRENT'].mean():>13,.0f})          │")
        print(f"  └───────────────────────────────────────────┘")

        # Key trade-area demographics
        print(f"\n  KEY TRADE AREA DEMOGRAPHICS (input to model):")
        demo_keys = [
            ("Population density", "POP_DENS", "{:.0f} /sq mi"),
            ("Median HH income", "POP_MHI", "${:,.0f}"),
            ("Renter %", "RENTER_PCT", "{:.1f}%"),
            ("Median age", "POP_MEDIAN_AGE", "{:.1f}"),
            ("College %", "POP_COLLEGE_PCT", "{:.1f}%"),
            ("Median home value", "MEDIAN_HOME_VALUE", "${:,.0f}"),
            ("Rent-burdened (>50%)", "RENTER_RENT_50PLUS_PCT", "{:.1f}%"),
            ("Total jobs", "TOTAL_JOBS", "{:,.0f}"),
            ("Jobs/HH ratio", "JOBS_PER_HH", "{:.2f}"),
            ("High-earn jobs %", "JOBS_HIGH_EARN_PCT", "{:.1f}%"),
        ]
        for label, key, fmt in demo_keys:
            if key in trade_area.columns:
                val = trade_area[key].median()
                if pd.notna(val):
                    print(f"    {label:28s}  {fmt.format(val)}")

        # Save predictions
        pred_df = pd.DataFrame({
            "site": [SITE_ADDRESS],
            "predicted_vacancy_pct": [round(vac_pred, 1)],
            "predicted_noi": [round(noi_pred, 0)],
            "trade_area_avg_vacancy": [round(trade_area["VACANCY_RATE"].mean(), 1)],
            "trade_area_avg_noi": [round(trade_area["NOI_CURRENT"].mean(), 0)],
            "n_competitors_3mi": [(trade_area["dist_to_site"].le(3) if "dist_to_site" in trade_area else trade_area["_dist"].le(3)).sum()],
            "n_competitors_5mi": [len(trade_area)],
        })
        pred_path = OUTPUT_DIR / "site_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"\n  Saved predictions: {pred_path}")

    print("\n\n" + "=" * 70)
    print("  DEMAND ANALYSIS COMPLETE")
    print("=" * 70)

    # Close shared Snowflake connection
    if _CONN and not _CONN.is_closed():
        _CONN.close()
        print("  ✓ Snowflake connection closed")


if __name__ == "__main__":
    main()
