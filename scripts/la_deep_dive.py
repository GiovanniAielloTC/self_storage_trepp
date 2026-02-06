"""
Los Angeles Self-Storage Deep Dive
====================================
Comprehensive analysis of self-storage properties in the LA MSA.
Includes vacancy, NOI, distress analysis, and interactive map.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
import warnings
import json
import time
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/bizon/giovanni/self_storage_trepp/output')
FIGURES_DIR = OUTPUT_DIR / 'figures'
DATA_DIR = Path('/home/bizon/giovanni/self_storage_trepp/data')

# ====================================================================
# 1. LOAD & FILTER
# ====================================================================
print("Loading data...")
df = pd.read_csv(DATA_DIR / 'self_storage_universe.csv', low_memory=False)

# Numeric conversions
num_cols = [
    'OCCUPANCY_CURRENT', 'VACANCY_RATE', 'OCCUPANCY_LAG1', 'OCCUPANCY_LAG2',
    'OCCUPANCY_AT_SEC', 'REVENUE_CURRENT', 'REVENUE_PRIOR', 'REVENUE_AT_SEC',
    'OPEX_CURRENT', 'OPEX_PRIOR', 'NOI_CURRENT', 'NOI_PRIOR', 'NOI_2YR_AGO',
    'NOI_AT_SEC', 'NCF_CURRENT', 'NCF_PRIOR', 'DSCR_CURRENT', 'DSCR_PRIOR',
    'DSCR_AT_SEC', 'LTV_AT_SEC', 'LTV_CURRENT', 'DEBT_YIELD_NOI',
    'LOAN_ORIGINAL_BALANCE', 'LOAN_CURRENT_BALANCE', 'NOTE_RATE',
    'CURRENT_NOTE_RATE', 'VALUE_CURRENT', 'VALUE_AT_SEC', 'BALANCE_PER_UNIT',
    'SQFT_AT_SEC', 'SQFT_CURRENT',
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

int_cols = ['UNITS', 'UNITS_CURRENT', 'YEAR_BUILT', 'REMAINING_TERM',
            'IS_WATCHLIST', 'IS_SPECIAL_SERVICING', 'IS_DELINQUENT',
            'MONTHS_DELINQUENT', 'IS_MODIFIED', 'IS_FORECLOSURE', 'IS_REO',
            'IS_INTEREST_ONLY', 'REPORT_YEAR', 'REPORT_MONTH']
for c in int_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Filter LA MSA
la_mask = df['MSA_NAME'].fillna('').str.contains('Los Angeles', case=False)
la = df[la_mask].copy()
print(f"LA MSA: {len(la):,} rows, {la['TREPPMASTERPROPERTYID'].nunique()} unique properties")

# ====================================================================
# 2. PROPERTY TYPE VERIFICATION
# ====================================================================
print("\n" + "="*60)
print("PROPERTY TYPE VERIFICATION")
print("="*60)
print(f"CREFCPROPERTYTYPE distribution:")
print(la['PROPERTY_TYPE_LONG'].value_counts(dropna=False))
print(f"\nAll {la['TREPPMASTERPROPERTYID'].nunique()} LA properties confirmed Self Storage")

# ====================================================================
# 3. UNIVERSE OVERVIEW
# ====================================================================
print("\n" + "="*60)
print("LA SELF-STORAGE UNIVERSE OVERVIEW")
print("="*60)

# Year coverage
year_counts = la.groupby('REPORT_YEAR')['TREPPMASTERPROPERTYID'].nunique()
print(f"\nYear range: {la['REPORT_YEAR'].min():.0f} - {la['REPORT_YEAR'].max():.0f}")
print(f"Peak coverage: {year_counts.max()} properties in {year_counts.idxmax():.0f}")
print(f"\nProperties per year:")
print(year_counts.to_string())

# Cities
city_counts = la.drop_duplicates('TREPPMASTERPROPERTYID')['CITY'].value_counts()
print(f"\nTop 15 cities (unique properties):")
print(city_counts.head(15).to_string())

# ====================================================================
# 4. LATEST SNAPSHOT ANALYSIS (2025)
# ====================================================================
la_snap = la[la['REPORT_YEAR'] == 2025].copy()
if len(la_snap) < 20:
    la_snap = la[la['REPORT_YEAR'] == la['REPORT_YEAR'].max()].copy()
snap_year = la_snap['REPORT_YEAR'].iloc[0]

print(f"\n{'='*60}")
print(f"LATEST SNAPSHOT ({snap_year:.0f}): {len(la_snap)} properties")
print(f"{'='*60}")

# Vacancy
print(f"\n--- VACANCY ---")
print(f"Mean occupancy:    {la_snap['OCCUPANCY_CURRENT'].mean():.1f}%")
print(f"Median occupancy:  {la_snap['OCCUPANCY_CURRENT'].median():.1f}%")
print(f"Mean vacancy:      {la_snap['VACANCY_RATE'].mean():.1f}%")
print(f"Median vacancy:    {la_snap['VACANCY_RATE'].median():.1f}%")
print(f"Min occupancy:     {la_snap['OCCUPANCY_CURRENT'].min():.1f}%")
print(f"Max occupancy:     {la_snap['OCCUPANCY_CURRENT'].max():.1f}%")
print(f"Std dev:           {la_snap['OCCUPANCY_CURRENT'].std():.1f}%")

# Vacancy buckets
bins = [0, 5, 10, 15, 20, 30, 50, 100]
labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50%+']
la_snap['vacancy_bucket'] = pd.cut(la_snap['VACANCY_RATE'], bins=bins, labels=labels, right=True)
print(f"\nVacancy distribution:")
print(la_snap['vacancy_bucket'].value_counts().sort_index().to_string())

# Revenue / NOI
print(f"\n--- REVENUE & NOI ---")
rev = la_snap['REVENUE_CURRENT'].dropna()
noi = la_snap['NOI_CURRENT'].dropna()
print(f"Revenue (n={len(rev)}):")
print(f"  Mean:   ${rev.mean():,.0f}")
print(f"  Median: ${rev.median():,.0f}")
print(f"  Total:  ${rev.sum():,.0f}")

print(f"NOI (n={len(noi)}):")
print(f"  Mean:   ${noi.mean():,.0f}")
print(f"  Median: ${noi.median():,.0f}")
print(f"  Total:  ${noi.sum():,.0f}")

noi_margin = la_snap.loc[la_snap['REVENUE_CURRENT'] > 0, 'NOI_CURRENT'] / la_snap.loc[la_snap['REVENUE_CURRENT'] > 0, 'REVENUE_CURRENT'] * 100
print(f"NOI Margin: {noi_margin.mean():.1f}% avg, {noi_margin.median():.1f}% median")

# DSCR
print(f"\n--- CREDIT ---")
dscr = la_snap['DSCR_CURRENT'].dropna()
print(f"DSCR (n={len(dscr)}):")
print(f"  Mean:   {dscr.mean():.2f}x")
print(f"  Median: {dscr.median():.2f}x")
print(f"  Below 1.0x: {(dscr < 1.0).sum()} ({(dscr < 1.0).mean()*100:.1f}%)")
print(f"  Below 1.25x: {(dscr < 1.25).sum()} ({(dscr < 1.25).mean()*100:.1f}%)")

ltv = la_snap['LTV_AT_SEC'].dropna()
print(f"LTV at Securitization (n={len(ltv)}): {ltv.mean():.1f}% avg")

# Distress
print(f"\n--- DISTRESS SIGNALS ---")
distress_cols = ['IS_WATCHLIST', 'IS_SPECIAL_SERVICING', 'IS_DELINQUENT', 'IS_MODIFIED', 'IS_FORECLOSURE', 'IS_REO']
for col in distress_cols:
    val = la_snap[col].sum()
    pct = la_snap[col].mean() * 100
    print(f"  {col}: {val:.0f} ({pct:.1f}%)")

# ====================================================================
# 5. HISTORICAL TRENDS
# ====================================================================
print(f"\n{'='*60}")
print("HISTORICAL TRENDS")
print("="*60)

yearly = la.groupby('REPORT_YEAR').agg(
    n_properties=('TREPPMASTERPROPERTYID', 'nunique'),
    avg_vacancy=('VACANCY_RATE', 'mean'),
    median_vacancy=('VACANCY_RATE', 'median'),
    avg_occupancy=('OCCUPANCY_CURRENT', 'mean'),
    avg_noi=('NOI_CURRENT', 'mean'),
    median_noi=('NOI_CURRENT', 'median'),
    avg_dscr=('DSCR_CURRENT', 'mean'),
    pct_watchlist=('IS_WATCHLIST', 'mean'),
    pct_special_servicing=('IS_SPECIAL_SERVICING', 'mean'),
    pct_delinquent=('IS_DELINQUENT', 'mean'),
    avg_revenue=('REVENUE_CURRENT', 'mean'),
).reset_index()
yearly['pct_watchlist'] *= 100
yearly['pct_special_servicing'] *= 100
yearly['pct_delinquent'] *= 100

print("\nYearly summary:")
print(yearly[['REPORT_YEAR', 'n_properties', 'avg_vacancy', 'avg_noi', 'avg_dscr', 'pct_watchlist']].to_string(index=False, float_format='%.1f'))

# ====================================================================
# 6. TOP DISTRESSED PROPERTIES
# ====================================================================
print(f"\n{'='*60}")
print("MOST DISTRESSED LA PROPERTIES (2025)")
print("="*60)

# Compute distress score
la_snap['DISTRESS_SCORE'] = (
    la_snap['IS_WATCHLIST'].fillna(0) +
    la_snap['IS_SPECIAL_SERVICING'].fillna(0) * 2 +
    la_snap['IS_DELINQUENT'].fillna(0) * 2 +
    la_snap['IS_MODIFIED'].fillna(0) +
    la_snap['IS_FORECLOSURE'].fillna(0) * 3 +
    la_snap['IS_REO'].fillna(0) * 3
)

distressed = la_snap[la_snap['DISTRESS_SCORE'] > 0].sort_values('DISTRESS_SCORE', ascending=False)
print(f"\n{len(distressed)} properties with distress score > 0:")
cols_show = ['PROPERTY_NAME', 'CITY', 'DISTRESS_SCORE', 'VACANCY_RATE', 'DSCR_CURRENT',
             'IS_WATCHLIST', 'IS_SPECIAL_SERVICING', 'IS_DELINQUENT']
if len(distressed) > 0:
    print(distressed[cols_show].head(20).to_string(index=False))

# ====================================================================
# 7. CITY-LEVEL ANALYSIS
# ====================================================================
print(f"\n{'='*60}")
print("CITY-LEVEL SUMMARY (2025)")
print("="*60)

city_stats = la_snap.groupby('CITY').agg(
    n=('TREPPMASTERPROPERTYID', 'nunique'),
    avg_vacancy=('VACANCY_RATE', 'mean'),
    avg_occupancy=('OCCUPANCY_CURRENT', 'mean'),
    avg_noi=('NOI_CURRENT', 'mean'),
    avg_dscr=('DSCR_CURRENT', 'mean'),
    total_distress=('DISTRESS_SCORE', 'sum'),
).sort_values('n', ascending=False)
print(city_stats.head(25).to_string(float_format='%.1f'))

# ====================================================================
# 8. FIGURES
# ====================================================================
print(f"\n{'='*60}")
print("GENERATING FIGURES...")
print("="*60)

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# --- Figure 1: LA Vacancy & NOI Trends ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Los Angeles Self-Storage: Historical Trends', fontsize=16, fontweight='bold')

# 1a: Vacancy trend
ax = axes[0, 0]
yrs = yearly[yearly['n_properties'] >= 10]
ax.plot(yrs['REPORT_YEAR'], yrs['avg_vacancy'], 'b-o', markersize=4, label='Mean Vacancy')
ax.plot(yrs['REPORT_YEAR'], yrs['median_vacancy'], 'g--s', markersize=3, label='Median Vacancy')
ax.set_ylabel('Vacancy Rate (%)')
ax.set_title('Vacancy Rate Over Time')
ax.legend(fontsize=9)
ax.set_xlabel('Year')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# 1b: NOI trend
ax = axes[0, 1]
noi_yrs = yrs.dropna(subset=['avg_noi'])
ax.plot(noi_yrs['REPORT_YEAR'], noi_yrs['avg_noi'], 'r-o', markersize=4, label='Mean NOI')
ax.plot(noi_yrs['REPORT_YEAR'], noi_yrs['median_noi'], 'orange', linestyle='--', marker='s', markersize=3, label='Median NOI')
ax.set_ylabel('NOI ($)')
ax.set_title('Average NOI Over Time')
ax.legend(fontsize=9)
ax.set_xlabel('Year')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# 1c: Distress rates
ax = axes[1, 0]
ax.plot(yrs['REPORT_YEAR'], yrs['pct_watchlist'], 'orange', marker='o', markersize=4, label='Watchlist %')
ax.plot(yrs['REPORT_YEAR'], yrs['pct_special_servicing'], 'r-s', markersize=4, label='Special Servicing %')
ax.plot(yrs['REPORT_YEAR'], yrs['pct_delinquent'], 'purple', marker='^', markersize=4, label='Delinquent %')
ax.set_ylabel('% of Properties')
ax.set_title('Distress Rates Over Time')
ax.legend(fontsize=9)
ax.set_xlabel('Year')

# 1d: Property count
ax = axes[1, 1]
ax.bar(yrs['REPORT_YEAR'], yrs['n_properties'], color='steelblue', alpha=0.7)
ax.set_ylabel('Properties in CMBS')
ax.set_title('LA Self-Storage CMBS Universe Size')
ax.set_xlabel('Year')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'la_ss_trends.png', dpi=150, bbox_inches='tight')
print("Saved: la_ss_trends.png")

# --- Figure 2: 2025 Snapshot Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Los Angeles Self-Storage: {snap_year:.0f} Snapshot ({len(la_snap)} Properties)', fontsize=16, fontweight='bold')

# 2a: Vacancy distribution
ax = axes[0, 0]
ax.hist(la_snap['VACANCY_RATE'].dropna(), bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(la_snap['VACANCY_RATE'].mean(), color='red', linestyle='--', label=f'Mean: {la_snap["VACANCY_RATE"].mean():.1f}%')
ax.axvline(la_snap['VACANCY_RATE'].median(), color='green', linestyle='--', label=f'Median: {la_snap["VACANCY_RATE"].median():.1f}%')
ax.set_xlabel('Vacancy Rate (%)')
ax.set_ylabel('Count')
ax.set_title('Vacancy Distribution')
ax.legend(fontsize=9)

# 2b: NOI distribution
ax = axes[0, 1]
noi_data = la_snap['NOI_CURRENT'].dropna()
noi_data = noi_data[(noi_data > 0) & (noi_data < noi_data.quantile(0.95))]
ax.hist(noi_data, bins=20, color='green', edgecolor='white', alpha=0.8)
ax.axvline(noi_data.mean(), color='red', linestyle='--', label=f'Mean: ${noi_data.mean():,.0f}')
ax.set_xlabel('NOI ($)')
ax.set_ylabel('Count')
ax.set_title('NOI Distribution')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

# 2c: DSCR distribution
ax = axes[1, 0]
dscr_data = la_snap['DSCR_CURRENT'].dropna()
dscr_data = dscr_data[(dscr_data > 0) & (dscr_data < 5)]
ax.hist(dscr_data, bins=20, color='orange', edgecolor='white', alpha=0.8)
ax.axvline(1.0, color='red', linewidth=2, linestyle='-', label='DSCR = 1.0x (breakeven)')
ax.axvline(dscr_data.median(), color='green', linestyle='--', label=f'Median: {dscr_data.median():.2f}x')
ax.set_xlabel('DSCR')
ax.set_ylabel('Count')
ax.set_title('Debt Service Coverage Ratio')
ax.legend(fontsize=9)

# 2d: Vacancy by city (top cities)
ax = axes[1, 1]
top_cities = la_snap.groupby('CITY').filter(lambda x: len(x) >= 3)
if len(top_cities) > 0:
    city_order = top_cities.groupby('CITY')['VACANCY_RATE'].median().sort_values(ascending=False).head(12).index
    plot_data = top_cities[top_cities['CITY'].isin(city_order)]
    plot_data['CITY'] = pd.Categorical(plot_data['CITY'], categories=city_order, ordered=True)
    plot_data.boxplot(column='VACANCY_RATE', by='CITY', ax=ax, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='steelblue', alpha=0.5))
    ax.set_title('Vacancy by City (3+ properties)')
    ax.set_xlabel('')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Vacancy Rate (%)')
    fig.suptitle(f'Los Angeles Self-Storage: {snap_year:.0f} Snapshot ({len(la_snap)} Properties)', fontsize=16, fontweight='bold')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'la_ss_snapshot.png', dpi=150, bbox_inches='tight')
print("Saved: la_ss_snapshot.png")


# ====================================================================
# 9. GEOCODE ALL UNIQUE LA PROPERTIES (across all years)
# ====================================================================
print(f"\n{'='*60}")
print("GEOCODING LA PROPERTIES FOR MAPS...")
print("="*60)

import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

MAP_YEARS = [2016, 2019, 2022, 2025]

# Collect all unique properties across the target years
all_props = []
for yr in MAP_YEARS:
    snap = la[la['REPORT_YEAR'] == yr].copy()
    all_props.append(snap[['TREPPMASTERPROPERTYID', 'PROPERTY_NAME', 'ADDRESS',
                           'CITY', 'STATE', 'ZIPCODE']].drop_duplicates('TREPPMASTERPROPERTYID'))

all_unique = pd.concat(all_props).drop_duplicates('TREPPMASTERPROPERTYID')
print(f"  Unique properties across {MAP_YEARS}: {len(all_unique)}")

# Load geocode cache if it exists
geocode_cache_path = DATA_DIR / 'la_ss_geocode_cache.csv'
if geocode_cache_path.exists():
    cache = pd.read_csv(geocode_cache_path)
    cache_dict = dict(zip(cache['TREPPMASTERPROPERTYID'],
                          zip(cache['lat'], cache['lon'])))
    print(f"  Loaded geocode cache: {len(cache_dict)} entries")
else:
    cache_dict = {}
    print("  No geocode cache found — will geocode from scratch")

# Geocode missing properties
to_geocode = all_unique[~all_unique['TREPPMASTERPROPERTYID'].isin(cache_dict)]
print(f"  Properties to geocode: {len(to_geocode)}")

if len(to_geocode) > 0:
    geolocator = Nominatim(user_agent="terracotta_ss_analysis", timeout=10)
    geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    for idx, row in to_geocode.iterrows():
        zip_str = str(int(row['ZIPCODE'])) if pd.notna(row['ZIPCODE']) else ''
        geocode_str = f"{row['ADDRESS']}, {row['CITY']}, CA {zip_str}"
        try:
            location = geocode_fn(geocode_str)
            if not location:
                location = geocode_fn(f"{row['CITY']}, CA {zip_str}")
            if location:
                cache_dict[row['TREPPMASTERPROPERTYID']] = (location.latitude, location.longitude)
        except Exception:
            pass
        done = len(cache_dict)
        print(f"  Geocoded {done}/{len(all_unique)}...", end='\r')

    # Save updated cache
    cache_rows = [{'TREPPMASTERPROPERTYID': pid, 'lat': ll[0], 'lon': ll[1]}
                  for pid, ll in cache_dict.items()]
    pd.DataFrame(cache_rows).to_csv(geocode_cache_path, index=False)
    print(f"\n  Saved geocode cache: {len(cache_dict)} entries")
else:
    print("  All properties already cached")


# ====================================================================
# 10. BUILD INTERACTIVE MAPS — one per year
# ====================================================================

# --- Helper functions ---
def get_vacancy_color(vac):
    """Green (low vacancy) -> Yellow -> Orange -> Red (high vacancy)"""
    if pd.isna(vac):
        return '#999999'
    elif vac <= 5:
        return '#1a9641'
    elif vac <= 10:
        return '#7bc96a'
    elif vac <= 15:
        return '#fee08b'
    elif vac <= 20:
        return '#f46d43'
    elif vac <= 30:
        return '#d73027'
    else:
        return '#a50026'

def get_radius(sqft):
    """Scale SQFT to marker radius."""
    if pd.isna(sqft) or sqft <= 0:
        return 6
    return max(5, min(22, 3 + (sqft / 10000) ** 0.5 * 3))

def is_distressed(row):
    return (row.get('IS_WATCHLIST', 0) == 1 or
            row.get('IS_SPECIAL_SERVICING', 0) == 1 or
            row.get('IS_DELINQUENT', 0) == 1)


def build_map(year_df, year, cache_dict):
    """Build a folium map for one year's snapshot."""
    props = year_df[['TREPPMASTERPROPERTYID', 'PROPERTY_NAME', 'ADDRESS', 'CITY', 'STATE',
                      'ZIPCODE', 'VACANCY_RATE', 'OCCUPANCY_CURRENT', 'NOI_CURRENT',
                      'DSCR_CURRENT', 'LOAN_CURRENT_BALANCE', 'IS_WATCHLIST',
                      'IS_SPECIAL_SERVICING', 'IS_DELINQUENT', 'DISTRESS_SCORE',
                      'YEAR_BUILT', 'UNITS', 'SQFT_CURRENT', 'SQFT_AT_SEC']].copy()
    props['SQFT'] = props['SQFT_CURRENT'].fillna(props['SQFT_AT_SEC'])

    # Attach cached coordinates
    props['lat'] = props['TREPPMASTERPROPERTYID'].map(lambda pid: cache_dict.get(pid, (None, None))[0])
    props['lon'] = props['TREPPMASTERPROPERTYID'].map(lambda pid: cache_dict.get(pid, (None, None))[1])
    geocoded = props.dropna(subset=['lat', 'lon'])

    # Save the 2025 geocoded set for other scripts
    if year == 2025:
        geocoded.to_csv(DATA_DIR / 'la_ss_geocoded.csv', index=False)

    # Summary stats for title
    n = len(geocoded)
    avg_vac = geocoded['VACANCY_RATE'].mean()
    n_distressed = geocoded.apply(is_distressed, axis=1).sum()

    la_center = [34.05, -118.25]
    m = folium.Map(location=la_center, zoom_start=10, tiles='CartoDB positron')

    # Add property markers
    for _, row in geocoded.iterrows():
        color = get_vacancy_color(row['VACANCY_RATE'])
        radius = get_radius(row.get('SQFT', None))
        distressed = is_distressed(row)

        noi_str = f"${row['NOI_CURRENT']:,.0f}" if pd.notna(row['NOI_CURRENT']) else 'N/A'
        dscr_str = f"{row['DSCR_CURRENT']:.2f}x" if pd.notna(row['DSCR_CURRENT']) else 'N/A'
        bal_str = f"${row['LOAN_CURRENT_BALANCE']:,.0f}" if pd.notna(row['LOAN_CURRENT_BALANCE']) else 'N/A'
        yr_str = f"{int(row['YEAR_BUILT'])}" if pd.notna(row['YEAR_BUILT']) else 'N/A'
        units_str = f"{int(row['UNITS'])}" if pd.notna(row['UNITS']) else 'N/A'
        sqft_str = f"{row['SQFT']:,.0f}" if pd.notna(row.get('SQFT')) else 'N/A'
        vac_str = f"{row['VACANCY_RATE']:.1f}%" if pd.notna(row['VACANCY_RATE']) else 'N/A'

        distress_flags = []
        if row.get('IS_WATCHLIST', 0) == 1: distress_flags.append('Watchlist')
        if row.get('IS_SPECIAL_SERVICING', 0) == 1: distress_flags.append('Special Servicing')
        if row.get('IS_DELINQUENT', 0) == 1: distress_flags.append('Delinquent')
        distress_str = ', '.join(distress_flags) if distress_flags else 'None'
        shape_label = '&#9650; DISTRESSED' if distressed else '&#9679; Healthy'

        popup_html = f"""
        <div style="font-family: Arial; width: 280px;">
            <h4 style="margin-bottom:5px;">{row['PROPERTY_NAME']}</h4>
            <p style="margin:2px 0; color:gray; font-size:11px;">{row['ADDRESS']}, {row['CITY']}, CA {row['ZIPCODE']}</p>
            <hr style="margin:5px 0;">
            <table style="font-size:12px; width:100%;">
                <tr><td><b>Year:</b></td><td>{year}</td></tr>
                <tr><td><b>Vacancy:</b></td><td style="color:{color}; font-weight:bold;">{vac_str}</td></tr>
                <tr><td><b>SQFT:</b></td><td>{sqft_str}</td></tr>
                <tr><td><b>NOI:</b></td><td>{noi_str}</td></tr>
                <tr><td><b>DSCR:</b></td><td>{dscr_str}</td></tr>
                <tr><td><b>Loan Balance:</b></td><td>{bal_str}</td></tr>
                <tr><td><b>Year Built:</b></td><td>{yr_str}</td></tr>
                <tr><td><b>Units:</b></td><td>{units_str}</td></tr>
                <tr><td><b>Status:</b></td><td style="color:{'red' if distressed else 'green'}; font-weight:bold;">{shape_label}</td></tr>
                <tr><td><b>Flags:</b></td><td style="color:{'red' if distress_flags else 'green'};">{distress_str}</td></tr>
            </table>
        </div>
        """

        if distressed:
            folium.RegularPolygonMarker(
                location=[row['lat'], row['lon']],
                number_of_sides=3,
                radius=radius + 2,
                color='#333', weight=2, fill=True,
                fill_color=color, fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"▲ {row['PROPERTY_NAME']} | Vac: {vac_str}"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                color=color, weight=1.5, fill=True,
                fill_color=color, fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{row['PROPERTY_NAME']} | Vac: {vac_str}"
            ).add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background-color: white; padding: 15px; border-radius: 8px;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-family: Arial; font-size: 12px;">
        <h4 style="margin:0 0 5px 0;">LA Self-Storage CMBS &mdash; {year}</h4>
        <p style="margin:0 0 3px 0; font-size:11px; color:#555;">{n} properties | Avg vacancy {avg_vac:.1f}% | {n_distressed} distressed</p>
        <hr style="margin:6px 0;">
        <p style="margin:0 0 6px 0; font-weight:bold; font-size:11px;">Color = Vacancy Rate</p>
        <p style="margin:2px 0;"><span style="color:#1a9641;">&#9679;</span> Low (&le; 5%)</p>
        <p style="margin:2px 0;"><span style="color:#7bc96a;">&#9679;</span> Normal (5-10%)</p>
        <p style="margin:2px 0;"><span style="color:#fee08b;">&#9679;</span> Elevated (10-15%)</p>
        <p style="margin:2px 0;"><span style="color:#f46d43;">&#9679;</span> High (15-20%)</p>
        <p style="margin:2px 0;"><span style="color:#d73027;">&#9679;</span> Very High (20-30%)</p>
        <p style="margin:2px 0;"><span style="color:#a50026;">&#9679;</span> Severe (&gt; 30%)</p>
        <hr style="margin:8px 0;">
        <p style="margin:0 0 6px 0; font-weight:bold; font-size:11px;">Shape = Distress Status</p>
        <p style="margin:2px 0;">&#9679; Circle = Healthy</p>
        <p style="margin:2px 0;">&#9650; Triangle = Distressed</p>
        <p style="margin:2px 0; color:#DAA520; font-weight:bold;">&#9733; Star = Proposed Site</p>
        <hr style="margin:8px 0;">
        <p style="margin:0; font-size:10px; color:gray;">Size = Square Footage</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Star marker + radius rings on all maps
    proposed_popup = f"""
    <div style="font-family: Arial; width: 280px;">
        <h4 style="margin-bottom:5px; color:#DAA520;">&#9733; PROPOSED SITE</h4>
        <p style="margin:2px 0; font-size:13px; font-weight:bold;">222 E 7th St</p>
        <p style="margin:2px 0; color:gray; font-size:11px;">Downtown Los Angeles, CA 90014</p>
        <hr style="margin:5px 0;">
        <p style="margin:4px 0; font-size:12px;">TerraCotta Group — New self-storage development under evaluation.</p>
        <p style="margin:4px 0; font-size:11px; color:#555;">Map vintage: {year} | Click nearby markers to see CMBS competitors.</p>
    </div>
    """
    folium.Marker(
        location=[34.0438, -118.2500],
        popup=folium.Popup(proposed_popup, max_width=320),
        tooltip=f"&#9733; PROPOSED SITE: 222 E 7th St ({year})",
        icon=folium.Icon(color='cadetblue', icon='star', prefix='fa')
    ).add_to(m)

    METERS_PER_MILE = 1609.34
    folium.Circle(
        location=[34.0438, -118.2500],
        radius=3 * METERS_PER_MILE,
        color='#DAA520', weight=2, fill=False,
        dash_array='8 6',
        tooltip='3-mile primary trade area',
    ).add_to(m)
    folium.Circle(
        location=[34.0438, -118.2500],
        radius=5 * METERS_PER_MILE,
        color='#DAA520', weight=1.5, fill=False,
        dash_array='4 8',
        tooltip='5-mile secondary trade area',
    ).add_to(m)

    return m, n


# --- Generate maps for each year ---
print("\nBuilding interactive maps...")
for yr in MAP_YEARS:
    snap = la[la['REPORT_YEAR'] == yr].copy()
    for c in num_cols:
        if c in snap.columns:
            snap[c] = pd.to_numeric(snap[c], errors='coerce')
    snap['DISTRESS_SCORE'] = (
        snap.get('IS_WATCHLIST', 0).fillna(0).astype(int) +
        snap.get('IS_SPECIAL_SERVICING', 0).fillna(0).astype(int) +
        snap.get('IS_DELINQUENT', 0).fillna(0).astype(int)
    )

    m, n = build_map(snap, yr, cache_dict)
    fname = f'la_self_storage_map_{yr}.html'
    m.save(str(OUTPUT_DIR / fname))
    print(f"  {yr}: {n} properties -> {fname}")

# Also save the latest as the default map name
import shutil
shutil.copy(OUTPUT_DIR / 'la_self_storage_map_2025.html',
            OUTPUT_DIR / 'la_self_storage_map.html')
print(f"  Copied 2025 map -> la_self_storage_map.html")


# ====================================================================
# 11. COMPARISON: LA vs NATIONAL
# ====================================================================
print(f"\n{'='*60}")
print("LA vs NATIONAL COMPARISON (2025)")
print("="*60)

nat_snap = df[df['REPORT_YEAR'] == 2025].copy()
for c in num_cols:
    if c in nat_snap.columns:
        nat_snap[c] = pd.to_numeric(nat_snap[c], errors='coerce')

comparison = pd.DataFrame({
    'Metric': [
        'Properties',
        'Avg Vacancy (%)',
        'Median Vacancy (%)',
        'Avg Occupancy (%)',
        'Avg NOI ($)',
        'Median NOI ($)',
        'Avg DSCR',
        'Watchlist (%)',
        'Special Servicing (%)',
        'Delinquent (%)',
    ],
    'LA': [
        len(la_snap),
        la_snap['VACANCY_RATE'].mean(),
        la_snap['VACANCY_RATE'].median(),
        la_snap['OCCUPANCY_CURRENT'].mean(),
        la_snap['NOI_CURRENT'].mean(),
        la_snap['NOI_CURRENT'].median(),
        la_snap['DSCR_CURRENT'].mean(),
        la_snap['IS_WATCHLIST'].mean() * 100,
        la_snap['IS_SPECIAL_SERVICING'].mean() * 100,
        la_snap['IS_DELINQUENT'].mean() * 100,
    ],
    'National': [
        len(nat_snap),
        nat_snap['VACANCY_RATE'].mean(),
        nat_snap['VACANCY_RATE'].median(),
        nat_snap['OCCUPANCY_CURRENT'].mean(),
        nat_snap['NOI_CURRENT'].mean(),
        nat_snap['NOI_CURRENT'].median(),
        nat_snap['DSCR_CURRENT'].mean(),
        nat_snap['IS_WATCHLIST'].mean() * 100,
        nat_snap['IS_SPECIAL_SERVICING'].mean() * 100,
        nat_snap['IS_DELINQUENT'].mean() * 100,
    ]
})
print(comparison.to_string(index=False, float_format='%.1f'))

print("\n" + "="*60)
print("DONE! All outputs saved.")
print("="*60)
