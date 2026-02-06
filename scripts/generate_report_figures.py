"""
Generate all figures for the Self-Storage CMBS Report
=====================================================
Produces publication-quality charts for the LA-focused report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Setup ──────────────────────────────────────────────────────────
DATA = Path('/home/bizon/giovanni/self_storage_trepp/data')
FIG  = Path('/home/bizon/giovanni/self_storage_trepp/output/figures')
FIG.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})

TC_BLUE   = '#1B3A5C'
TC_GREEN  = '#1a9641'
TC_ORANGE = '#f46d43'
TC_RED    = '#d73027'
TC_GRAY   = '#888888'

# ── Load data ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA / 'self_storage_universe.csv', low_memory=False)
num_cols = [
    'OCCUPANCY_CURRENT', 'VACANCY_RATE', 'OCCUPANCY_LAG1',
    'REVENUE_CURRENT', 'REVENUE_PRIOR', 'REVENUE_AT_SEC',
    'OPEX_CURRENT', 'OPEX_PRIOR', 'OPEX_AT_SEC',
    'NOI_CURRENT', 'NOI_PRIOR', 'NOI_AT_SEC',
    'NCF_CURRENT', 'DSCR_CURRENT', 'DSCR_PRIOR',
    'LTV_AT_SEC', 'LTV_CURRENT',
    'LOAN_CURRENT_BALANCE', 'SQFT_CURRENT', 'SQFT_AT_SEC',
    'IS_WATCHLIST', 'IS_SPECIAL_SERVICING', 'IS_DELINQUENT',
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

la  = df[df['MSA_NAME'].fillna('').str.contains('Los Angeles', case=False)]
nat = df.copy()

# Derived metrics
for d in [la, nat]:
    d['OPEX_RATIO']  = np.where(d['REVENUE_CURRENT']>0, d['OPEX_CURRENT']/d['REVENUE_CURRENT']*100, np.nan)
    d['NOI_MARGIN']  = np.where(d['REVENUE_CURRENT']>0, d['NOI_CURRENT']/d['REVENUE_CURRENT']*100, np.nan)

# Snapshots
la_snap  = la[la['REPORT_YEAR']==2025].copy()
nat_snap = nat[nat['REPORT_YEAR']==2025].copy()

for d in [la_snap, nat_snap]:
    d['OPEX_RATIO']   = np.where(d['REVENUE_CURRENT']>0, d['OPEX_CURRENT']/d['REVENUE_CURRENT']*100, np.nan)
    d['NOI_MARGIN']   = np.where(d['REVENUE_CURRENT']>0, d['NOI_CURRENT']/d['REVENUE_CURRENT']*100, np.nan)
    d['OPEX_GROWTH']  = np.where(d['OPEX_PRIOR']>0, (d['OPEX_CURRENT']-d['OPEX_PRIOR'])/d['OPEX_PRIOR']*100, np.nan)
    d['REV_GROWTH']   = np.where(d['REVENUE_PRIOR']>0, (d['REVENUE_CURRENT']-d['REVENUE_PRIOR'])/d['REVENUE_PRIOR']*100, np.nan)
    d['OPEX_VS_SEC']  = np.where(d['OPEX_AT_SEC']>0, (d['OPEX_CURRENT']-d['OPEX_AT_SEC'])/d['OPEX_AT_SEC']*100, np.nan)
    d['REV_PER_SQFT'] = np.where(d['SQFT_CURRENT']>0, d['REVENUE_CURRENT']/d['SQFT_CURRENT'], np.nan)
    d['NOI_PER_SQFT'] = np.where(d['SQFT_CURRENT']>0, d['NOI_CURRENT']/d['SQFT_CURRENT'], np.nan)

# Yearly aggregation
def yearly_agg(data, label=''):
    y = data.groupby('REPORT_YEAR').agg(
        n=('TREPPMASTERPROPERTYID', 'nunique'),
        avg_vac=('VACANCY_RATE', 'mean'),
        med_vac=('VACANCY_RATE', 'median'),
        avg_noi=('NOI_CURRENT', 'mean'),
        med_noi=('NOI_CURRENT', 'median'),
        avg_dscr=('DSCR_CURRENT', 'mean'),
        med_dscr=('DSCR_CURRENT', 'median'),
        pct_wl=('IS_WATCHLIST', 'mean'),
        avg_opex_ratio=('OPEX_RATIO', 'mean'),
        avg_noi_margin=('NOI_MARGIN', 'mean'),
    ).reset_index()
    y['pct_wl'] *= 100
    return y[y['n'] >= 10]

la_yr  = yearly_agg(la, 'LA')
nat_yr = yearly_agg(nat, 'NAT')

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: National Overview (2x2)
# ═══════════════════════════════════════════════════════════════════
print("Figure 1: National overview...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('U.S. Self-Storage CMBS: National Overview', fontsize=16, fontweight='bold', y=0.98)

# 1a: Universe size
ax = axes[0, 0]
ax.bar(nat_yr['REPORT_YEAR'], nat_yr['n'], color=TC_BLUE, alpha=0.8)
ax.set_ylabel('Active Properties')
ax.set_title('CMBS Self-Storage Universe')
ax.set_xlabel('Year')

# 1b: Vacancy trend
ax = axes[0, 1]
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['avg_vac'], 'o-', color=TC_RED, markersize=4, label='Mean')
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['med_vac'], 's--', color=TC_ORANGE, markersize=3, label='Median')
ax.set_ylabel('Vacancy Rate (%)')
ax.set_title('National Vacancy Trend')
ax.legend(fontsize=9)
ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5)

# 1c: NOI trend
ax = axes[1, 0]
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['med_noi']/1e6, 'o-', color=TC_GREEN, markersize=4)
ax.set_ylabel('Median NOI ($M)')
ax.set_title('National Median NOI')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'${x:.1f}M'))

# 1d: Watchlist %
ax = axes[1, 1]
ax.fill_between(nat_yr['REPORT_YEAR'], nat_yr['pct_wl'], alpha=0.3, color=TC_ORANGE)
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['pct_wl'], 'o-', color=TC_ORANGE, markersize=4)
ax.set_ylabel('% on Watchlist')
ax.set_title('National Watchlist Rate')
ax.set_xlabel('Year')

plt.tight_layout()
fig.savefig(FIG / 'report_01_national_overview.png', dpi=150, bbox_inches='tight')
print(f"  Saved: report_01_national_overview.png")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: LA vs National Comparison (2x2)
# ═══════════════════════════════════════════════════════════════════
print("Figure 2: LA vs National...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Los Angeles vs. National: Self-Storage CMBS', fontsize=16, fontweight='bold', y=0.98)

# 2a: Vacancy comparison
ax = axes[0, 0]
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['avg_vac'], 'o-', color=TC_GRAY, markersize=4, label='National')
ax.plot(la_yr['REPORT_YEAR'], la_yr['avg_vac'], 's-', color=TC_BLUE, markersize=5, label='Los Angeles', linewidth=2)
ax.set_ylabel('Avg Vacancy (%)')
ax.set_title('Vacancy Rate: LA vs National')
ax.legend(fontsize=9)
ax.axhline(y=10, color='gray', linestyle=':', alpha=0.4)

# 2b: DSCR comparison
ax = axes[0, 1]
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['med_dscr'], 'o-', color=TC_GRAY, markersize=4, label='National')
ax.plot(la_yr['REPORT_YEAR'], la_yr['med_dscr'], 's-', color=TC_GREEN, markersize=5, label='Los Angeles', linewidth=2)
ax.set_ylabel('Median DSCR')
ax.set_title('DSCR: LA vs National')
ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Breakeven')
ax.legend(fontsize=9)

# 2c: NOI comparison
ax = axes[1, 0]
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['med_noi']/1e6, 'o-', color=TC_GRAY, markersize=4, label='National')
ax.plot(la_yr['REPORT_YEAR'], la_yr['med_noi']/1e6, 's-', color=TC_GREEN, markersize=5, label='Los Angeles', linewidth=2)
ax.set_ylabel('Median NOI ($M)')
ax.set_title('NOI: LA vs National')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'${x:.1f}M'))

# 2d: Watchlist comparison
ax = axes[1, 1]
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['pct_wl'], 'o-', color=TC_GRAY, markersize=4, label='National')
ax.plot(la_yr['REPORT_YEAR'], la_yr['pct_wl'], 's-', color=TC_ORANGE, markersize=5, label='Los Angeles', linewidth=2)
ax.set_ylabel('% on Watchlist')
ax.set_title('Watchlist Rate: LA vs National')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG / 'report_02_la_vs_national.png', dpi=150, bbox_inches='tight')
print(f"  Saved: report_02_la_vs_national.png")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: LA 2025 Snapshot - Vacancy & NOI distributions
# ═══════════════════════════════════════════════════════════════════
print("Figure 3: LA snapshot distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Los Angeles Self-Storage: 2025 Snapshot (101 Properties)', fontsize=16, fontweight='bold', y=0.98)

# 3a: Vacancy distribution
ax = axes[0, 0]
ax.hist(la_snap['VACANCY_RATE'].dropna(), bins=20, color=TC_BLUE, edgecolor='white', alpha=0.8)
ax.axvline(la_snap['VACANCY_RATE'].mean(), color=TC_RED, linestyle='--', linewidth=1.5,
           label=f'Mean: {la_snap["VACANCY_RATE"].mean():.1f}%')
ax.axvline(la_snap['VACANCY_RATE'].median(), color=TC_GREEN, linestyle='--', linewidth=1.5,
           label=f'Median: {la_snap["VACANCY_RATE"].median():.1f}%')
ax.set_xlabel('Vacancy Rate (%)')
ax.set_ylabel('Count')
ax.set_title('Vacancy Distribution')
ax.legend(fontsize=9)

# 3b: DSCR distribution
ax = axes[0, 1]
dscr_d = la_snap['DSCR_CURRENT'].dropna()
dscr_d = dscr_d[(dscr_d > 0) & (dscr_d < 6)]
ax.hist(dscr_d, bins=20, color=TC_GREEN, edgecolor='white', alpha=0.8)
ax.axvline(1.0, color=TC_RED, linewidth=2, linestyle='-', label='DSCR=1.0x (breakeven)')
ax.axvline(dscr_d.median(), color=TC_BLUE, linestyle='--', label=f'Median: {dscr_d.median():.2f}x')
ax.set_xlabel('DSCR')
ax.set_ylabel('Count')
ax.set_title('DSCR Distribution')
ax.legend(fontsize=9)

# 3c: NOI distribution
ax = axes[1, 0]
noi_d = la_snap['NOI_CURRENT'].dropna()
noi_d = noi_d[(noi_d > 0) & (noi_d < noi_d.quantile(0.95))]
ax.hist(noi_d / 1e6, bins=20, color=TC_GREEN, edgecolor='white', alpha=0.8)
ax.axvline(noi_d.mean()/1e6, color=TC_RED, linestyle='--', label=f'Mean: ${noi_d.mean()/1e6:.1f}M')
ax.set_xlabel('NOI ($M)')
ax.set_ylabel('Count')
ax.set_title('NOI Distribution')
ax.legend(fontsize=9)

# 3d: Vacancy by top cities
ax = axes[1, 1]
top_cities = la_snap.groupby('CITY').filter(lambda x: len(x) >= 3)
if len(top_cities) > 0:
    city_med = top_cities.groupby('CITY')['VACANCY_RATE'].median().sort_values(ascending=True)
    colors = [TC_GREEN if v <= 10 else (TC_ORANGE if v <= 15 else TC_RED) for v in city_med.values]
    city_med.plot.barh(ax=ax, color=colors)
    ax.set_xlabel('Median Vacancy (%)')
    ax.set_title('Vacancy by City (3+ properties)')
    ax.axvline(10, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
fig.savefig(FIG / 'report_03_la_snapshot.png', dpi=150, bbox_inches='tight')
print(f"  Saved: report_03_la_snapshot.png")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: OPEX & Efficiency Metrics
# ═══════════════════════════════════════════════════════════════════
print("Figure 4: OPEX & efficiency...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Los Angeles Self-Storage: Operating Efficiency (2025)', fontsize=16, fontweight='bold', y=0.98)

# 4a: OpEx Ratio distribution, LA vs National overlay
ax = axes[0, 0]
bins_h = np.arange(0, 85, 5)
ax.hist(nat_snap['OPEX_RATIO'].dropna(), bins=bins_h, color=TC_GRAY, alpha=0.4, label=f'National (med: {nat_snap["OPEX_RATIO"].median():.0f}%)', density=True)
ax.hist(la_snap['OPEX_RATIO'].dropna(), bins=bins_h, color=TC_BLUE, alpha=0.6, label=f'LA (med: {la_snap["OPEX_RATIO"].median():.0f}%)', density=True)
ax.set_xlabel('OpEx / Revenue (%)')
ax.set_ylabel('Density')
ax.set_title('OpEx Ratio: LA vs National')
ax.legend(fontsize=9)

# 4b: NOI Margin distribution
ax = axes[0, 1]
ax.hist(nat_snap['NOI_MARGIN'].dropna(), bins=bins_h, color=TC_GRAY, alpha=0.4, label=f'National (med: {nat_snap["NOI_MARGIN"].median():.0f}%)', density=True)
ax.hist(la_snap['NOI_MARGIN'].dropna(), bins=bins_h, color=TC_GREEN, alpha=0.6, label=f'LA (med: {la_snap["NOI_MARGIN"].median():.0f}%)', density=True)
ax.set_xlabel('NOI / Revenue (%)')
ax.set_ylabel('Density')
ax.set_title('NOI Margin: LA vs National')
ax.legend(fontsize=9)

# 4c: Historical OpEx Ratio trend
ax = axes[1, 0]
ax.plot(nat_yr['REPORT_YEAR'], nat_yr['avg_opex_ratio'], 'o-', color=TC_GRAY, markersize=4, label='National')
ax.plot(la_yr['REPORT_YEAR'], la_yr['avg_opex_ratio'], 's-', color=TC_BLUE, markersize=5, label='LA', linewidth=2)
ax.set_ylabel('Avg OpEx Ratio (%)')
ax.set_title('OpEx Ratio Over Time')
ax.legend(fontsize=9)
ax.set_xlabel('Year')

# 4d: Vacancy vs NOI Margin scatter (LA 2025)
ax = axes[1, 1]
valid = la_snap.dropna(subset=['VACANCY_RATE', 'NOI_MARGIN'])
wl = valid['IS_WATCHLIST'] == 1
ax.scatter(valid.loc[~wl, 'VACANCY_RATE'], valid.loc[~wl, 'NOI_MARGIN'],
           c=TC_BLUE, alpha=0.6, s=50, label='Healthy', edgecolors='white', linewidth=0.5)
ax.scatter(valid.loc[wl, 'VACANCY_RATE'], valid.loc[wl, 'NOI_MARGIN'],
           c=TC_RED, alpha=0.8, s=80, marker='^', label='Watchlist', edgecolors='white', linewidth=0.5)
ax.set_xlabel('Vacancy Rate (%)')
ax.set_ylabel('NOI Margin (%)')
ax.set_title('Vacancy vs. NOI Margin')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG / 'report_04_opex_efficiency.png', dpi=150, bbox_inches='tight')
print(f"  Saved: report_04_opex_efficiency.png")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Static Map (vacancy color, SQFT size, triangle distress)
# ═══════════════════════════════════════════════════════════════════
print("Figure 5: Static map...")
try:
    import contextily as cx

    geo = pd.read_csv(DATA / 'la_ss_geocoded.csv')
    for c in ['VACANCY_RATE', 'SQFT', 'SQFT_CURRENT', 'SQFT_AT_SEC', 'IS_WATCHLIST', 'IS_SPECIAL_SERVICING', 'IS_DELINQUENT']:
        if c in geo.columns:
            geo[c] = pd.to_numeric(geo[c], errors='coerce')

    if 'SQFT' not in geo.columns:
        geo['SQFT'] = geo['SQFT_CURRENT'].fillna(geo.get('SQFT_AT_SEC', np.nan))

    geo['is_distressed'] = (geo['IS_WATCHLIST'].fillna(0) == 1) | \
                           (geo.get('IS_SPECIAL_SERVICING', pd.Series(0)).fillna(0) == 1) | \
                           (geo.get('IS_DELINQUENT', pd.Series(0)).fillna(0) == 1)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Vacancy color
    def vac_color(v):
        if v <= 5: return '#1a9641'
        elif v <= 10: return '#7bc96a'
        elif v <= 15: return '#fee08b'
        elif v <= 20: return '#f46d43'
        elif v <= 30: return '#d73027'
        else: return '#a50026'

    geo['color'] = geo['VACANCY_RATE'].apply(vac_color)
    geo['size'] = geo['SQFT'].apply(lambda s: max(30, min(350, s/500)) if pd.notna(s) and s > 0 else 40)

    healthy = geo[~geo['is_distressed']]
    distressed = geo[geo['is_distressed']]

    # Circles for healthy
    ax.scatter(healthy['lon'], healthy['lat'], s=healthy['size'], c=healthy['color'],
               alpha=0.7, edgecolors='#333', linewidth=0.5, marker='o', zorder=5, label='Healthy')

    # Triangles for distressed
    if len(distressed) > 0:
        ax.scatter(distressed['lon'], distressed['lat'], s=distressed['size']*1.5,
                   c=distressed['color'], alpha=0.85, edgecolors='black', linewidth=1.5,
                   marker='^', zorder=6, label='Distressed (▲)')

    # Basemap
    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron, zoom=11)

    ax.set_xlim(geo['lon'].min()-0.12, geo['lon'].max()+0.12)
    ax.set_ylim(geo['lat'].min()-0.08, geo['lat'].max()+0.08)
    ax.set_axis_off()
    ax.set_title('LA Self-Storage CMBS: 101 Properties (2025)\nColor = Vacancy | Size = SQFT | ▲ = Distressed',
                 fontsize=14, fontweight='bold', pad=15)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a9641', markersize=10, label='Vac ≤ 5%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7bc96a', markersize=10, label='Vac 5-10%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fee08b', markersize=10, label='Vac 10-15%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f46d43', markersize=10, label='Vac 15-20%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', markersize=10, label='Vac 20-30%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#a50026', markersize=10, label='Vac > 30%'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black',
               markersize=12, label='▲ Distressed'),
    ]
    ax.legend(handles=legend_els, loc='lower left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIG / 'report_05_la_map.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: report_05_la_map.png")
except Exception as e:
    print(f"  Map error: {e}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: LA Historical Deep Dive
# ═══════════════════════════════════════════════════════════════════
print("Figure 6: LA historical...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Los Angeles Self-Storage: Historical Trends', fontsize=16, fontweight='bold', y=0.98)

# 6a: Property count over time
ax = axes[0, 0]
ax.bar(la_yr['REPORT_YEAR'], la_yr['n'], color=TC_BLUE, alpha=0.8)
ax.set_ylabel('Active Properties')
ax.set_title('LA CMBS Self-Storage Universe')

# 6b: Vacancy with cycle context
ax = axes[0, 1]
ax.fill_between(la_yr['REPORT_YEAR'], la_yr['avg_vac'], alpha=0.2, color=TC_RED)
ax.plot(la_yr['REPORT_YEAR'], la_yr['avg_vac'], 'o-', color=TC_RED, markersize=4, linewidth=2)
ax.axhspan(0, 10, alpha=0.05, color='green')
ax.axhline(10, color='gray', linestyle=':', alpha=0.5)
ax.set_ylabel('Avg Vacancy (%)')
ax.set_title('LA Vacancy History')
# COVID & GFC labels
for yr, lbl in [(2009, 'GFC'), (2020, 'COVID')]:
    if yr in la_yr['REPORT_YEAR'].values:
        v = la_yr.loc[la_yr['REPORT_YEAR']==yr, 'avg_vac'].iloc[0]
        ax.annotate(lbl, xy=(yr, v), xytext=(yr+1, v+2),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray')

# 6c: NOI margin trend
ax = axes[1, 0]
ax.plot(la_yr['REPORT_YEAR'], la_yr['avg_noi_margin'], 'o-', color=TC_GREEN, markersize=4, linewidth=2)
ax.set_ylabel('Avg NOI Margin (%)')
ax.set_title('LA NOI Margin Over Time')
ax.set_xlabel('Year')

# 6d: DSCR trend
ax = axes[1, 1]
ax.plot(la_yr['REPORT_YEAR'], la_yr['med_dscr'], 's-', color=TC_BLUE, markersize=5, linewidth=2)
ax.axhline(1.0, color=TC_RED, linestyle=':', label='Breakeven')
ax.set_ylabel('Median DSCR')
ax.set_title('LA DSCR Over Time')
ax.set_xlabel('Year')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG / 'report_06_la_historical.png', dpi=150, bbox_inches='tight')
print(f"  Saved: report_06_la_historical.png")

print("\n✅ All report figures generated!")
