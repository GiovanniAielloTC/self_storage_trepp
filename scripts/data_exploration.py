"""
Self-Storage CMBS Data Exploration
===================================
Pull and explore self-storage properties from the TREPP CMBS universe.
Focus areas: vacancy, revenue/NOI, and distress signals.

Property Type Filter: CREFCPROPERTYTYPE = 'Self Storage'
or NORMALIZEDPROPERTYTYPE LIKE 'SS%'
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.snowflake_io import SnowflakeConnection

import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path('/home/bizon/giovanni/self_storage_trepp/output')
DATA_DIR = Path('/home/bizon/giovanni/self_storage_trepp/data')
FIGURES_DIR = OUTPUT_DIR / 'figures'


def fetch_self_storage_universe():
    """
    Fetch the full self-storage CMBS loan universe from Trepp.
    Returns panel data: property x year with vacancy, NOI, revenue, distress signals.
    """
    
    query = """
    WITH ss_data AS (
        SELECT 
            -- IDs
            l.TREPPMASTERLOANID,
            p.TREPPMASTERPROPERTYID,
            p.PROPERTYID,
            p.PROPNAME AS PROPERTY_NAME,
            p.ADDRESS,
            p.CITY,
            p.STATE,
            p.ZIPCODE,
            p.COUNTY,
            p.MSACODE,
            p.MSAABBREVIATION AS MSA_NAME,
            CAST(l.YEAR AS INT) AS REPORT_YEAR,
            CAST(l.MONTH AS INT) AS REPORT_MONTH,
            
            -- ============================================
            -- VACANCY / OCCUPANCY
            -- ============================================
            TRY_CAST(l.MOSTRECENTPHYSICALOCCUPANCY AS FLOAT) AS OCCUPANCY_CURRENT,
            100 - TRY_CAST(l.MOSTRECENTPHYSICALOCCUPANCY AS FLOAT) AS VACANCY_RATE,
            TRY_CAST(l.PRECEDINGFISCALYEARPHYSICALOCCUPANCY AS FLOAT) AS OCCUPANCY_LAG1,
            TRY_CAST(l.SECONDPRECEDINGFISCALYEARPHYSICALOCCUPANCY AS FLOAT) AS OCCUPANCY_LAG2,
            TRY_CAST(l.SECURITIZATIONOCCUPANCY AS FLOAT) AS OCCUPANCY_AT_SEC,
            
            -- ============================================
            -- REVENUE / NOI / CASH FLOW
            -- ============================================
            TRY_CAST(l.MOSTRECENTREVENUE AS FLOAT) AS REVENUE_CURRENT,
            TRY_CAST(l.PRECEDINGFISCALYEARREVENUE AS FLOAT) AS REVENUE_PRIOR,
            TRY_CAST(l.REVENUEATSECURITIZATION AS FLOAT) AS REVENUE_AT_SEC,
            
            TRY_CAST(l.MOSTRECENTOPERATINGEXPENSES AS FLOAT) AS OPEX_CURRENT,
            TRY_CAST(l.PRECEDINGFISCALYEAROPERATINGEXPENSES AS FLOAT) AS OPEX_PRIOR,
            TRY_CAST(l.OPERATINGEXPENSESATSECURITIZATION AS FLOAT) AS OPEX_AT_SEC,
            
            TRY_CAST(l.MOSTRECENTNOI AS FLOAT) AS NOI_CURRENT,
            TRY_CAST(l.PRECEDINGFISCALYEARNOI AS FLOAT) AS NOI_PRIOR,
            TRY_CAST(l.SECONDPRECEDINGFISCALYEARNOI AS FLOAT) AS NOI_2YR_AGO,
            TRY_CAST(l.SECURITIZATIONNOI AS FLOAT) AS NOI_AT_SEC,
            
            TRY_CAST(l.MOSTRECENTNCF AS FLOAT) AS NCF_CURRENT,
            TRY_CAST(l.PRECEDINGFISCALYEARNCF AS FLOAT) AS NCF_PRIOR,
            TRY_CAST(l.SECURITIZATIONNCF AS FLOAT) AS NCF_AT_SEC,
            
            -- ============================================
            -- CREDIT / LEVERAGE METRICS
            -- ============================================
            TRY_CAST(l.MOSTRECENTDSCR_NOI AS FLOAT) AS DSCR_CURRENT,
            TRY_CAST(l.PRECEDINGFISCALYEARDSCR_NOI AS FLOAT) AS DSCR_PRIOR,
            TRY_CAST(l.SECURITZATIONDSCR_NOI AS FLOAT) AS DSCR_AT_SEC,
            TRY_CAST(l.MOSTRECENTDSCR_NCF AS FLOAT) AS DSCR_NCF_CURRENT,
            
            TRY_CAST(l.SECURITIZATIONLTV AS FLOAT) AS LTV_AT_SEC,
            TRY_CAST(l.DERIVEDLTV AS FLOAT) AS LTV_CURRENT,
            
            TRY_CAST(l.DERIVEDCURRENTDEBTYIELDNOI AS FLOAT) AS DEBT_YIELD_NOI,
            TRY_CAST(l.DERIVEDCURRENTDEBTYIELDNCF AS FLOAT) AS DEBT_YIELD_NCF,
            
            -- ============================================
            -- DISTRESS SIGNALS
            -- ============================================
            CASE WHEN UPPER(l.WATCHLIST) = 'Y' THEN 1 ELSE 0 END AS IS_WATCHLIST,
            CASE WHEN UPPER(l.SPECIALSERVICE) = 'Y' THEN 1 ELSE 0 END AS IS_SPECIAL_SERVICING,
            CASE WHEN UPPER(l.NEWLYONWATCHLIST) = 'Y' THEN 1 ELSE 0 END AS NEWLY_WATCHLIST,
            CASE WHEN UPPER(l.NEWLYSENTTOSPECIALSERVICING) = 'Y' THEN 1 ELSE 0 END AS NEWLY_SS,
            
            TRY_CAST(l.MONTHSDELINQUENT AS INT) AS MONTHS_DELINQUENT,
            CASE WHEN TRY_CAST(l.MONTHSDELINQUENT AS INT) > 0 THEN 1 ELSE 0 END AS IS_DELINQUENT,
            l.DERIVEDDELINQUENCYSTATUS AS DELINQUENCY_STATUS,
            
            CASE WHEN UPPER(l.MODIFICATIONINDICATOR) = 'Y' THEN 1 ELSE 0 END AS IS_MODIFIED,
            l.WORKOUTSTRATEGY AS WORKOUT_STRATEGY,
            
            l.FORECLOSURESTARTDATE,
            l.REODATE,
            CASE WHEN l.FORECLOSURESTARTDATE IS NOT NULL THEN 1 ELSE 0 END AS IS_FORECLOSURE,
            CASE WHEN l.REODATE IS NOT NULL THEN 1 ELSE 0 END AS IS_REO,
            
            -- ============================================
            -- PROPERTY CHARACTERISTICS
            -- ============================================
            TRY_CAST(p.UNITS AS INT) AS UNITS,
            TRY_CAST(p.CURRENTNUMBEROFUNITSBEDSROOMS AS INT) AS UNITS_CURRENT,
            TRY_CAST(p.NETSQUAREFEETATSECURITIZATION AS FLOAT) AS SQFT_AT_SEC,
            TRY_CAST(p.CURRENTNETRENTABLESQUAREFEET AS FLOAT) AS SQFT_CURRENT,
            TRY_CAST(p.YEARBUILT AS INT) AS YEAR_BUILT,
            p.PROPERTYCONDITION AS PROPERTY_CONDITION,
            p.NORMALIZEDPROPERTYTYPE AS PROPERTY_TYPE_CODE,
            p.CREFCPROPERTYTYPE AS PROPERTY_TYPE_LONG,
            
            -- ============================================
            -- LOAN STRUCTURE
            -- ============================================
            TRY_CAST(l.ORIGINALBALANCE AS FLOAT) AS LOAN_ORIGINAL_BALANCE,
            TRY_CAST(l.ACTUALBALANCE AS FLOAT) AS LOAN_CURRENT_BALANCE,
            TRY_CAST(l.ORIGINALNOTERATE AS FLOAT) AS NOTE_RATE,
            TRY_CAST(l.CURRENTNOTERATE AS FLOAT) AS CURRENT_NOTE_RATE,
            TRY_CAST(l.REMAININGTERM AS INT) AS REMAINING_TERM,
            TRY_CAST(l.MATURITYDATE AS DATE) AS MATURITY_DATE,
            TRY_CAST(l.ORIGINATIONDATE AS DATE) AS ORIGINATION_DATE,
            CASE WHEN UPPER(l.INTERESTONLY_YN) = 'Y' THEN 1 ELSE 0 END AS IS_INTEREST_ONLY,
            l.AMORTIZATIONTYPE,
            
            -- ============================================
            -- VALUATION
            -- ============================================
            TRY_CAST(l.MOSTRECENTVALUE AS FLOAT) AS VALUE_CURRENT,
            TRY_CAST(l.SECURITIZATIONAPPRAISEDVALUE AS FLOAT) AS VALUE_AT_SEC,
            TRY_CAST(l.CURRENTBALANCEPERSQFTORUNIT AS FLOAT) AS BALANCE_PER_UNIT,
            
            -- ============================================
            -- SERVICER / DEAL
            -- ============================================
            l.TREPPDEALNAME AS DEAL_NAME,
            l.MASTERSERVICER AS MASTER_SERVICER,
            l.CURLOANSPECIALSERVICER AS SPECIAL_SERVICER,
            l.ORIGINATOR,
            
            -- Dedup: take latest month per property-year
            ROW_NUMBER() OVER (
                PARTITION BY p.TREPPMASTERPROPERTYID, l.YEAR 
                ORDER BY l.MONTH DESC
            ) AS rn
            
        FROM RAW.ANALYSIS.TREPP_CMBS_LOAN l
        JOIN RAW.ANALYSIS.TREPP_CMBS_PROPERTY p 
            ON l.TREPPMASTERLOANID = p.TREPPMASTERLOANID
            AND l.YEAR = p.YEAR 
            AND l.MONTH = p.MONTH
        WHERE (
            p.CREFCPROPERTYTYPE = 'Self Storage'
            OR p.NORMALIZEDPROPERTYTYPE LIKE 'SS%'
        )
        AND l.MOSTRECENTPHYSICALOCCUPANCY IS NOT NULL
        AND TRY_CAST(l.MOSTRECENTPHYSICALOCCUPANCY AS FLOAT) > 0
        AND TRY_CAST(l.MOSTRECENTPHYSICALOCCUPANCY AS FLOAT) <= 100
    )
    SELECT * EXCLUDE (rn) FROM ss_data WHERE rn = 1
    ORDER BY TREPPMASTERPROPERTYID, REPORT_YEAR
    """
    
    sf = SnowflakeConnection()
    df = sf.fetch_df(query)
    sf.close()
    
    return df


def compute_derived_features(df):
    """Engineer features from raw data."""
    
    # Convert all numeric columns from Decimal to float
    numeric_cols = [
        'OCCUPANCY_CURRENT', 'VACANCY_RATE', 'OCCUPANCY_LAG1', 'OCCUPANCY_LAG2', 'OCCUPANCY_AT_SEC',
        'REVENUE_CURRENT', 'REVENUE_PRIOR', 'REVENUE_AT_SEC',
        'OPEX_CURRENT', 'OPEX_PRIOR', 'OPEX_AT_SEC',
        'NOI_CURRENT', 'NOI_PRIOR', 'NOI_2YR_AGO', 'NOI_AT_SEC',
        'NCF_CURRENT', 'NCF_PRIOR', 'NCF_AT_SEC',
        'DSCR_CURRENT', 'DSCR_PRIOR', 'DSCR_AT_SEC', 'DSCR_NCF_CURRENT',
        'LTV_AT_SEC', 'LTV_CURRENT', 'DEBT_YIELD_NOI', 'DEBT_YIELD_NCF',
        'LOAN_ORIGINAL_BALANCE', 'LOAN_CURRENT_BALANCE', 'NOTE_RATE', 'CURRENT_NOTE_RATE',
        'VALUE_CURRENT', 'VALUE_AT_SEC', 'BALANCE_PER_UNIT',
        'SQFT_AT_SEC', 'SQFT_CURRENT',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    int_cols = ['UNITS', 'UNITS_CURRENT', 'YEAR_BUILT', 'REMAINING_TERM',
                'IS_WATCHLIST', 'IS_SPECIAL_SERVICING', 'IS_DELINQUENT',
                'MONTHS_DELINQUENT', 'IS_MODIFIED', 'IS_FORECLOSURE', 'IS_REO',
                'IS_INTEREST_ONLY', 'NEWLY_WATCHLIST', 'NEWLY_SS',
                'REPORT_YEAR', 'REPORT_MONTH']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Vacancy features
    df['VACANCY_LAG1'] = 100 - df['OCCUPANCY_LAG1'].fillna(df['OCCUPANCY_CURRENT'])
    df['VACANCY_LAG2'] = 100 - df['OCCUPANCY_LAG2'].fillna(df['OCCUPANCY_LAG1'])
    df['VACANCY_AT_SEC'] = 100 - df['OCCUPANCY_AT_SEC']
    df['VACANCY_CHANGE_1Y'] = df['VACANCY_RATE'] - df['VACANCY_LAG1']
    df['VACANCY_VS_SEC'] = df['VACANCY_RATE'] - df['VACANCY_AT_SEC']
    
    # Revenue / NOI growth
    df['REVENUE_GROWTH_1Y'] = np.where(
        df['REVENUE_PRIOR'] > 0,
        (df['REVENUE_CURRENT'] - df['REVENUE_PRIOR']) / df['REVENUE_PRIOR'] * 100,
        np.nan
    )
    df['NOI_GROWTH_1Y'] = np.where(
        df['NOI_PRIOR'] > 0,
        (df['NOI_CURRENT'] - df['NOI_PRIOR']) / df['NOI_PRIOR'] * 100,
        np.nan
    )
    df['NOI_VS_SEC'] = np.where(
        df['NOI_AT_SEC'] > 0,
        (df['NOI_CURRENT'] - df['NOI_AT_SEC']) / df['NOI_AT_SEC'] * 100,
        np.nan
    )
    
    # Expense efficiency
    df['OPEX_RATIO'] = np.where(
        df['REVENUE_CURRENT'] > 0,
        df['OPEX_CURRENT'] / df['REVENUE_CURRENT'] * 100,
        np.nan
    )
    df['NOI_MARGIN'] = np.where(
        df['REVENUE_CURRENT'] > 0,
        df['NOI_CURRENT'] / df['REVENUE_CURRENT'] * 100,
        np.nan
    )
    
    # Revenue per unit / per SF
    df['REVENUE_PER_UNIT'] = np.where(
        df['UNITS'] > 0, df['REVENUE_CURRENT'] / df['UNITS'], np.nan
    )
    df['NOI_PER_UNIT'] = np.where(
        df['UNITS'] > 0, df['NOI_CURRENT'] / df['UNITS'], np.nan
    )
    df['REVENUE_PER_SF'] = np.where(
        df['SQFT_CURRENT'] > 0, df['REVENUE_CURRENT'] / df['SQFT_CURRENT'], np.nan
    )
    
    # DSCR momentum
    df['DSCR_CHANGE_1Y'] = df['DSCR_CURRENT'] - df['DSCR_PRIOR']
    df['DSCR_VS_SEC'] = df['DSCR_CURRENT'] - df['DSCR_AT_SEC']
    df['IS_DSCR_DISTRESSED'] = (df['DSCR_CURRENT'] < 1.0).astype(int)
    
    # Property age
    df['PROPERTY_AGE'] = df['REPORT_YEAR'].astype(int) - df['YEAR_BUILT']
    
    # Maturity risk
    max_year = int(df['REPORT_YEAR'].max())
    df['MONTHS_TO_MATURITY'] = (
        pd.to_datetime(df['MATURITY_DATE']) - 
        pd.Timestamp(f'{max_year}-12-31')
    ).dt.days / 30
    df['IS_NEAR_MATURITY'] = (df['MONTHS_TO_MATURITY'] < 24).astype(int)
    
    # Composite distress score
    df['DISTRESS_SCORE'] = (
        df['IS_WATCHLIST'] + 
        df['IS_SPECIAL_SERVICING'] * 2 + 
        df['IS_DELINQUENT'] * 2 + 
        df['IS_MODIFIED'] +
        df['IS_FORECLOSURE'] * 3 +
        df['IS_REO'] * 3 +
        df['IS_DSCR_DISTRESSED']
    )
    
    return df


def print_universe_summary(df):
    """Print summary statistics of the self-storage universe."""
    
    print("=" * 80)
    print("SELF-STORAGE CMBS UNIVERSE SUMMARY")
    print("=" * 80)
    
    latest_year = df['REPORT_YEAR'].max()
    latest = df[df['REPORT_YEAR'] == latest_year]
    
    print(f"\n--- Universe Size ---")
    print(f"Total property-year observations: {len(df):,}")
    print(f"Unique properties: {df['TREPPMASTERPROPERTYID'].nunique():,}")
    print(f"Year range: {df['REPORT_YEAR'].min()} - {df['REPORT_YEAR'].max()}")
    print(f"States covered: {df['STATE'].nunique()}")
    print(f"MSAs covered: {df['MSA_NAME'].nunique()}")
    
    print(f"\n--- Latest Year ({latest_year}) ---")
    print(f"Properties: {len(latest):,}")
    
    print(f"\n--- Vacancy (Latest Year) ---")
    print(f"  Median: {latest['VACANCY_RATE'].median():.1f}%")
    print(f"  Mean:   {latest['VACANCY_RATE'].mean():.1f}%")
    print(f"  25th:   {latest['VACANCY_RATE'].quantile(0.25):.1f}%")
    print(f"  75th:   {latest['VACANCY_RATE'].quantile(0.75):.1f}%")
    
    print(f"\n--- NOI (Latest Year) ---")
    noi_valid = latest[latest['NOI_CURRENT'] > 0]
    print(f"  Properties with NOI: {len(noi_valid):,}")
    print(f"  Median NOI: ${noi_valid['NOI_CURRENT'].median():,.0f}")
    print(f"  Mean NOI:   ${noi_valid['NOI_CURRENT'].mean():,.0f}")
    
    print(f"\n--- NOI Growth 1Y (Latest Year) ---")
    noi_growth = latest['NOI_GROWTH_1Y'].dropna()
    print(f"  Properties with growth data: {len(noi_growth):,}")
    print(f"  Median growth: {noi_growth.median():.1f}%")
    print(f"  Mean growth:   {noi_growth.mean():.1f}%")
    
    print(f"\n--- Revenue (Latest Year) ---")
    rev_valid = latest[latest['REVENUE_CURRENT'] > 0]
    print(f"  Properties with revenue: {len(rev_valid):,}")
    print(f"  Median revenue: ${rev_valid['REVENUE_CURRENT'].median():,.0f}")
    print(f"  Revenue per unit: ${rev_valid['REVENUE_PER_UNIT'].median():,.0f}")
    
    print(f"\n--- Operating Efficiency (Latest Year) ---")
    opex_valid = latest[latest['OPEX_RATIO'].between(10, 90)]
    print(f"  Median OpEx ratio: {opex_valid['OPEX_RATIO'].median():.1f}%")
    print(f"  Median NOI margin: {opex_valid['NOI_MARGIN'].median():.1f}%")
    
    print(f"\n--- DSCR (Latest Year) ---")
    dscr_valid = latest[latest['DSCR_CURRENT'].between(0, 10)]
    print(f"  Median DSCR: {dscr_valid['DSCR_CURRENT'].median():.2f}x")
    print(f"  % Below 1.0x: {(dscr_valid['DSCR_CURRENT'] < 1.0).mean()*100:.1f}%")
    
    print(f"\n--- Distress Signals (Latest Year) ---")
    print(f"  On watchlist:       {latest['IS_WATCHLIST'].sum():,} ({latest['IS_WATCHLIST'].mean()*100:.1f}%)")
    print(f"  Special servicing:  {latest['IS_SPECIAL_SERVICING'].sum():,} ({latest['IS_SPECIAL_SERVICING'].mean()*100:.1f}%)")
    print(f"  Delinquent:         {latest['IS_DELINQUENT'].sum():,} ({latest['IS_DELINQUENT'].mean()*100:.1f}%)")
    print(f"  Modified:           {latest['IS_MODIFIED'].sum():,} ({latest['IS_MODIFIED'].mean()*100:.1f}%)")
    print(f"  In foreclosure:     {latest['IS_FORECLOSURE'].sum():,} ({latest['IS_FORECLOSURE'].mean()*100:.1f}%)")
    print(f"  REO:                {latest['IS_REO'].sum():,} ({latest['IS_REO'].mean()*100:.1f}%)")
    
    print(f"\n--- Geographic Distribution (Latest Year, Top 15 States) ---")
    state_counts = latest.groupby('STATE').agg(
        count=('VACANCY_RATE', 'size'),
        median_vacancy=('VACANCY_RATE', 'median'),
        median_noi=('NOI_CURRENT', 'median'),
        pct_distressed=('DISTRESS_SCORE', lambda x: (x > 0).mean() * 100)
    ).sort_values('count', ascending=False).head(15)
    print(state_counts.to_string())
    
    print(f"\n--- Top 15 MSAs (Latest Year) ---")
    msa_counts = latest.groupby('MSA_NAME').agg(
        count=('VACANCY_RATE', 'size'),
        median_vacancy=('VACANCY_RATE', 'median'),
        median_noi=('NOI_CURRENT', 'median'),
        pct_distressed=('DISTRESS_SCORE', lambda x: (x > 0).mean() * 100)
    ).sort_values('count', ascending=False).head(15)
    print(msa_counts.to_string())
    
    return latest


def plot_vacancy_trends(df):
    """Plot self-storage vacancy trends over time."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Self-Storage CMBS Universe: Vacancy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Vacancy distribution by year
    yearly = df.groupby('REPORT_YEAR')['VACANCY_RATE'].agg(['median', 'mean', 
        lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    yearly.columns = ['Median', 'Mean', 'Q25', 'Q75']
    
    ax = axes[0, 0]
    ax.fill_between(yearly.index, yearly['Q25'], yearly['Q75'], alpha=0.3, color='steelblue')
    ax.plot(yearly.index, yearly['Median'], 'o-', color='steelblue', linewidth=2, label='Median')
    ax.plot(yearly.index, yearly['Mean'], 's--', color='orange', linewidth=1.5, label='Mean')
    ax.set_xlabel('Year')
    ax.set_ylabel('Vacancy Rate (%)')
    ax.set_title('Vacancy Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Current vacancy distribution
    latest_year = df['REPORT_YEAR'].max()
    latest = df[df['REPORT_YEAR'] == latest_year]
    
    ax = axes[0, 1]
    ax.hist(latest['VACANCY_RATE'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(latest['VACANCY_RATE'].median(), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {latest["VACANCY_RATE"].median():.1f}%')
    ax.set_xlabel('Vacancy Rate (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Vacancy Distribution ({latest_year})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Vacancy by distress status
    ax = axes[1, 0]
    distress_groups = {
        'Performing': latest[latest['DISTRESS_SCORE'] == 0]['VACANCY_RATE'],
        'Watchlist': latest[latest['IS_WATCHLIST'] == 1]['VACANCY_RATE'],
        'Special Svc': latest[latest['IS_SPECIAL_SERVICING'] == 1]['VACANCY_RATE'],
        'Delinquent': latest[latest['IS_DELINQUENT'] == 1]['VACANCY_RATE'],
    }
    data = [v.dropna() for v in distress_groups.values() if len(v.dropna()) > 0]
    labels = [k for k, v in distress_groups.items() if len(v.dropna()) > 0]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Vacancy Rate (%)')
    ax.set_title('Vacancy by Distress Status')
    ax.grid(True, alpha=0.3)
    
    # 4. Property count over time
    ax = axes[1, 1]
    yearly_count = df.groupby('REPORT_YEAR')['TREPPMASTERPROPERTYID'].nunique()
    ax.bar(yearly_count.index, yearly_count.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Properties')
    ax.set_title('Self-Storage Properties in CMBS Universe')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ss_vacancy_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'ss_vacancy_analysis.png'}")


def plot_noi_analysis(df):
    """Plot NOI and revenue trends for self-storage."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Self-Storage CMBS Universe: Revenue & NOI Analysis', fontsize=16, fontweight='bold')
    
    latest_year = df['REPORT_YEAR'].max()
    latest = df[df['REPORT_YEAR'] == latest_year]
    
    # 1. NOI growth distribution
    ax = axes[0, 0]
    noi_growth = latest['NOI_GROWTH_1Y'].dropna()
    noi_growth_clipped = noi_growth.clip(-50, 100)
    ax.hist(noi_growth_clipped, bins=50, color='forestgreen', edgecolor='white', alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(noi_growth.median(), color='red', linestyle='--', linewidth=2,
               label=f'Median: {noi_growth.median():.1f}%')
    ax.set_xlabel('NOI Growth YoY (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'NOI Growth Distribution ({latest_year})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. NOI over time
    yearly_noi = df[df['NOI_CURRENT'] > 0].groupby('REPORT_YEAR')['NOI_CURRENT'].agg(['median', 'mean'])
    ax = axes[0, 1]
    ax.plot(yearly_noi.index, yearly_noi['median'] / 1000, 'o-', color='forestgreen', linewidth=2, label='Median')
    ax.plot(yearly_noi.index, yearly_noi['mean'] / 1000, 's--', color='orange', linewidth=1.5, label='Mean')
    ax.set_xlabel('Year')
    ax.set_ylabel('NOI ($K)')
    ax.set_title('NOI Trends Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. DSCR distribution
    ax = axes[1, 0]
    dscr_valid = latest['DSCR_CURRENT'].dropna()
    dscr_clipped = dscr_valid[dscr_valid.between(0, 5)]
    ax.hist(dscr_clipped, bins=50, color='darkorange', edgecolor='white', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='DSCR = 1.0x')
    ax.axvline(dscr_clipped.median(), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {dscr_clipped.median():.2f}x')
    ax.set_xlabel('DSCR (NOI basis)')
    ax.set_ylabel('Count')
    ax.set_title(f'DSCR Distribution ({latest_year})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. OpEx ratio vs vacancy
    ax = axes[1, 1]
    opex_valid = latest[latest['OPEX_RATIO'].between(10, 90) & latest['VACANCY_RATE'].between(0, 50)]
    ax.scatter(opex_valid['VACANCY_RATE'], opex_valid['OPEX_RATIO'], alpha=0.3, s=15, color='purple')
    ax.set_xlabel('Vacancy Rate (%)')
    ax.set_ylabel('OpEx / Revenue (%)')
    ax.set_title('Operating Efficiency vs Vacancy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ss_noi_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'ss_noi_analysis.png'}")


def plot_distress_analysis(df):
    """Plot distress signal analysis for self-storage."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Self-Storage CMBS Universe: Distress Signals', fontsize=16, fontweight='bold')
    
    # 1. Distress rates over time
    yearly_distress = df.groupby('REPORT_YEAR').agg(
        watchlist_pct=('IS_WATCHLIST', 'mean'),
        ss_pct=('IS_SPECIAL_SERVICING', 'mean'),
        delinquent_pct=('IS_DELINQUENT', 'mean'),
        modified_pct=('IS_MODIFIED', 'mean'),
    ) * 100
    
    ax = axes[0, 0]
    ax.plot(yearly_distress.index, yearly_distress['watchlist_pct'], 'o-', label='Watchlist', linewidth=2)
    ax.plot(yearly_distress.index, yearly_distress['ss_pct'], 's-', label='Special Servicing', linewidth=2)
    ax.plot(yearly_distress.index, yearly_distress['delinquent_pct'], '^-', label='Delinquent', linewidth=2)
    ax.plot(yearly_distress.index, yearly_distress['modified_pct'], 'D-', label='Modified', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('% of Properties')
    ax.set_title('Distress Rates Over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Distress score distribution (latest year)
    latest_year = df['REPORT_YEAR'].max()
    latest = df[df['REPORT_YEAR'] == latest_year]
    
    ax = axes[0, 1]
    distress_dist = latest['DISTRESS_SCORE'].value_counts().sort_index()
    ax.bar(distress_dist.index, distress_dist.values, color='crimson', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Distress Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Distress Score Distribution ({latest_year})')
    ax.grid(True, alpha=0.3)
    
    # 3. NOI growth by distress status
    ax = axes[1, 0]
    for label, mask in [
        ('Performing', latest['DISTRESS_SCORE'] == 0),
        ('Watchlist', latest['IS_WATCHLIST'] == 1),
        ('Spec Svc', latest['IS_SPECIAL_SERVICING'] == 1),
    ]:
        noi_g = latest.loc[mask, 'NOI_GROWTH_1Y'].dropna().clip(-50, 100)
        if len(noi_g) > 5:
            ax.hist(noi_g, bins=30, alpha=0.4, label=f'{label} (n={len(noi_g)})')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('NOI Growth YoY (%)')
    ax.set_ylabel('Count')
    ax.set_title('NOI Growth by Distress Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. DSCR by distress status
    ax = axes[1, 1]
    groups = {
        'Performing': latest.loc[latest['DISTRESS_SCORE'] == 0, 'DSCR_CURRENT'].dropna(),
        'Watchlist': latest.loc[latest['IS_WATCHLIST'] == 1, 'DSCR_CURRENT'].dropna(),
        'Spec Svc': latest.loc[latest['IS_SPECIAL_SERVICING'] == 1, 'DSCR_CURRENT'].dropna(),
    }
    data = [v[v.between(0, 5)] for v in groups.values() if len(v) > 0]
    labels = [k for k, v in groups.items() if len(v) > 0]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('DSCR')
    ax.set_title('DSCR by Distress Status')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ss_distress_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'ss_distress_analysis.png'}")


def main():
    """Main execution: fetch, compute, explore, plot."""
    
    print("Fetching self-storage CMBS universe from Snowflake...")
    df = fetch_self_storage_universe()
    
    print(f"\nRaw data: {len(df):,} rows, {df.shape[1]} columns")
    
    # Save raw data
    df.to_csv(DATA_DIR / 'self_storage_universe.csv', index=False)
    print(f"Saved raw data: {DATA_DIR / 'self_storage_universe.csv'}")
    
    # Compute derived features
    df = compute_derived_features(df)
    
    # Print summary
    latest = print_universe_summary(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_vacancy_trends(df)
    plot_noi_analysis(df)
    plot_distress_analysis(df)
    
    # Save enriched data
    df.to_csv(DATA_DIR / 'self_storage_enriched.csv', index=False)
    print(f"\nSaved enriched data: {DATA_DIR / 'self_storage_enriched.csv'}")
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    df = main()
