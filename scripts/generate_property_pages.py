"""
Generate 8 Interactive Property Profile Pages
==============================================
Each page shows full time-series trends for a single self-storage property:
vacancy, NOI, OPEX, revenue, DSCR, loan balance, OPEX ratio, NOI margin.

Selection criteria:
  2 × Start Low → End High vacancy
  2 × Start High → End Low vacancy
  2 × Always Low vacancy
  2 × Always High vacancy
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('/home/bizon/giovanni/self_storage_trepp/output')
DATA_DIR = Path('/home/bizon/giovanni/self_storage_trepp/data')

# ─── Selected properties ────────────────────────────────────────────────
PROPERTIES = [
    # (name, category, emoji, color)
    ("Lockaway SS - Hollywood",        "Low → High Vacancy",  "📈", "#e74c3c"),
    ("Cypress Mini Storage",           "Low → High Vacancy",  "📈", "#e74c3c"),
    ("Storage Center of Valencia",     "High → Low Vacancy",  "📉", "#27ae60"),
    ("Storage Express - Palmdale, CA", "High → Low Vacancy",  "📉", "#27ae60"),
    ("BV Hollywood Storage",           "Always Low Vacancy",  "🟢", "#2ecc71"),
    ("Storage Etc. - Torrance, CA",    "Always Low Vacancy",  "🟢", "#2ecc71"),
    ("SoCal Torrance",                 "Always High Vacancy", "🔴", "#c0392b"),
    ("Storbox Self Storage",           "Always High Vacancy", "🔴", "#c0392b"),
]


def build_property_page(prop_df, prop_name, category, emoji, cat_color):
    """Build a single-page interactive HTML dashboard for one property."""
    
    prop_df = prop_df.sort_values("REPORT_YEAR").copy()
    
    # Extract data
    city = prop_df["CITY"].iloc[0]
    state = prop_df["STATE"].iloc[0]
    address = prop_df["ADDRESS"].iloc[0] if "ADDRESS" in prop_df.columns else ""
    zipcode = prop_df["ZIPCODE"].iloc[0] if "ZIPCODE" in prop_df.columns else ""
    year_built = prop_df["YEAR_BUILT"].iloc[0] if "YEAR_BUILT" in prop_df.columns else None
    
    years = prop_df["REPORT_YEAR"].astype(int).tolist()
    
    def safe_list(col):
        if col in prop_df.columns:
            return [round(float(v), 2) if pd.notna(v) else None for v in prop_df[col]]
        return [None] * len(years)
    
    vacancy = safe_list("VACANCY_RATE")
    occupancy = safe_list("OCCUPANCY_CURRENT")
    noi = safe_list("NOI_CURRENT")
    revenue = safe_list("REVENUE_CURRENT")
    opex = safe_list("OPEX_CURRENT")
    dscr = safe_list("DSCR_CURRENT")
    loan = safe_list("LOAN_CURRENT_BALANCE")
    
    # Compute OPEX ratio and NOI margin
    opex_ratio = []
    noi_margin = []
    for i in range(len(years)):
        r = revenue[i]
        o = opex[i]
        n = noi[i]
        if r and r > 0 and o is not None:
            opex_ratio.append(round(o / r * 100, 1))
        else:
            opex_ratio.append(None)
        if r and r > 0 and n is not None:
            noi_margin.append(round(n / r * 100, 1))
        else:
            noi_margin.append(None)
    
    # Summary stats
    latest = prop_df.iloc[-1]
    earliest = prop_df.iloc[0]
    
    def fmt_money(v):
        if pd.isna(v) or v is None: return "N/A"
        return f"${v:,.0f}"
    
    def fmt_pct(v):
        if pd.isna(v) or v is None: return "N/A"
        return f"{v:.1f}%"
    
    yr_built_str = str(int(year_built)) if pd.notna(year_built) else "N/A"
    
    # JSON-safe null handling
    def js_array(arr):
        return str(arr).replace("None", "null")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{prop_name} — Property Profile</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; color: #333; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white; padding: 30px 20px; text-align: center;
        }}
        .header h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
        .header .address {{ color: #aaa; font-size: 0.95em; }}
        .header .category {{
            display: inline-block; margin-top: 10px; padding: 4px 14px;
            border-radius: 20px; font-size: 0.85em; font-weight: 600;
            background: {cat_color}; color: white;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
        .kpi-row {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px; margin-bottom: 24px;
        }}
        .kpi {{
            background: white; border-radius: 10px; padding: 16px; text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .kpi .label {{ font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
        .kpi .value {{ font-size: 1.5em; font-weight: 700; margin-top: 4px; }}
        .kpi .sub {{ font-size: 0.75em; color: #999; margin-top: 2px; }}
        .chart-grid {{
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 16px; margin-bottom: 20px;
        }}
        .chart-card {{
            background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .chart-card h3 {{
            font-size: 0.95em; color: #555; margin-bottom: 12px;
            border-bottom: 2px solid #eee; padding-bottom: 6px;
        }}
        canvas {{ max-height: 280px; }}
        .data-table {{
            background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px;
            overflow-x: auto;
        }}
        .data-table h3 {{ font-size: 0.95em; color: #555; margin-bottom: 12px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
        th {{ background: #f1f3f5; padding: 8px 10px; text-align: right; font-weight: 600; color: #555; }}
        th:first-child {{ text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #f0f0f0; text-align: right; }}
        td:first-child {{ text-align: left; font-weight: 600; }}
        .footer {{
            text-align: center; padding: 20px; color: #aaa; font-size: 0.8em;
            border-top: 1px solid #eee; margin-top: 20px;
        }}
        .footer a {{ color: #888; text-decoration: none; }}
        .nav {{ text-align: center; margin-bottom: 16px; }}
        .nav a {{
            display: inline-block; padding: 6px 16px; margin: 4px;
            border-radius: 20px; background: #e9ecef; color: #495057;
            text-decoration: none; font-size: 0.85em;
        }}
        .nav a:hover {{ background: #dee2e6; }}
        @media (max-width: 700px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .kpi-row {{ grid-template-columns: repeat(3, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{emoji} {prop_name}</h1>
        <div class="address">{address}, {city}, {state} {zipcode} &nbsp;|&nbsp; Built: {yr_built_str}</div>
        <div class="category">{category}</div>
    </div>
    <div class="container">
        <div class="nav">
            <a href="index.html">← All Maps</a>
            <a href="property_profiles.html">All 8 Profiles</a>
        </div>

        <div class="kpi-row">
            <div class="kpi">
                <div class="label">Latest Vacancy</div>
                <div class="value" style="color: {'#e74c3c' if (latest.get('VACANCY_RATE') or 0) > 15 else '#27ae60'};">{fmt_pct(latest.get('VACANCY_RATE'))}</div>
                <div class="sub">{int(latest['REPORT_YEAR'])}</div>
            </div>
            <div class="kpi">
                <div class="label">Latest NOI</div>
                <div class="value">{fmt_money(latest.get('NOI_CURRENT'))}</div>
                <div class="sub">{int(latest['REPORT_YEAR'])}</div>
            </div>
            <div class="kpi">
                <div class="label">Latest DSCR</div>
                <div class="value" style="color: {'#e74c3c' if (latest.get('DSCR_CURRENT') or 0) < 1.25 else '#27ae60'};">{f"{latest['DSCR_CURRENT']:.2f}x" if pd.notna(latest.get('DSCR_CURRENT')) else 'N/A'}</div>
                <div class="sub">{int(latest['REPORT_YEAR'])}</div>
            </div>
            <div class="kpi">
                <div class="label">Loan Balance</div>
                <div class="value">{fmt_money(latest.get('LOAN_CURRENT_BALANCE'))}</div>
                <div class="sub">{int(latest['REPORT_YEAR'])}</div>
            </div>
            <div class="kpi">
                <div class="label">Data Span</div>
                <div class="value" style="font-size:1.1em;">{int(earliest['REPORT_YEAR'])}–{int(latest['REPORT_YEAR'])}</div>
                <div class="sub">{len(years)} years</div>
            </div>
            <div class="kpi">
                <div class="label">Watchlist</div>
                <div class="value" style="color: {'#e74c3c' if latest.get('IS_WATCHLIST') == 1 else '#27ae60'};">{'⚠️ Yes' if latest.get('IS_WATCHLIST') == 1 else '✅ No'}</div>
                <div class="sub">{int(latest['REPORT_YEAR'])}</div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-card">
                <h3>📊 Vacancy Rate (%)</h3>
                <canvas id="chartVacancy"></canvas>
            </div>
            <div class="chart-card">
                <h3>💰 NOI & Revenue ($)</h3>
                <canvas id="chartNOI"></canvas>
            </div>
            <div class="chart-card">
                <h3>🏗️ OPEX & OPEX Ratio</h3>
                <canvas id="chartOPEX"></canvas>
            </div>
            <div class="chart-card">
                <h3>🏦 Loan Balance ($)</h3>
                <canvas id="chartLoan"></canvas>
            </div>
            <div class="chart-card">
                <h3>📈 DSCR (Debt Service Coverage)</h3>
                <canvas id="chartDSCR"></canvas>
            </div>
            <div class="chart-card">
                <h3>📊 NOI Margin (%)</h3>
                <canvas id="chartMargin"></canvas>
            </div>
        </div>

        <div class="data-table">
            <h3>📋 Full Data Table</h3>
            <table>
                <thead>
                    <tr>
                        <th>Year</th><th>Vacancy</th><th>NOI</th><th>Revenue</th>
                        <th>OPEX</th><th>OPEX Ratio</th><th>NOI Margin</th>
                        <th>DSCR</th><th>Loan Bal.</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for i, yr in enumerate(years):
        v = fmt_pct(vacancy[i]) if vacancy[i] is not None else "—"
        n = fmt_money(noi[i]) if noi[i] is not None else "—"
        r = fmt_money(revenue[i]) if revenue[i] is not None else "—"
        o = fmt_money(opex[i]) if opex[i] is not None else "—"
        oratio = f"{opex_ratio[i]:.1f}%" if opex_ratio[i] is not None else "—"
        nm = f"{noi_margin[i]:.1f}%" if noi_margin[i] is not None else "—"
        d = f"{dscr[i]:.2f}x" if dscr[i] is not None else "—"
        lb = fmt_money(loan[i]) if loan[i] is not None else "—"
        html += f"                    <tr><td>{yr}</td><td>{v}</td><td>{n}</td><td>{r}</td><td>{o}</td><td>{oratio}</td><td>{nm}</td><td>{d}</td><td>{lb}</td></tr>\n"
    
    html += f"""                </tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        <p>TerraCotta Group — CRE Research | Data: Trepp CMBS via Snowflake</p>
        <p><a href="index.html">← Back to Maps</a> &nbsp;|&nbsp; <a href="property_profiles.html">All Profiles</a></p>
    </div>

    <script>
    const years = {years};
    const vacancy = {js_array(vacancy)};
    const noi = {js_array(noi)};
    const revenue = {js_array(revenue)};
    const opex = {js_array(opex)};
    const opexRatio = {js_array(opex_ratio)};
    const noiMargin = {js_array(noi_margin)};
    const dscr = {js_array(dscr)};
    const loan = {js_array(loan)};

    const commonOpts = {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{ legend: {{ display: true, position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
        scales: {{ x: {{ ticks: {{ font: {{ size: 11 }} }} }}, y: {{ ticks: {{ font: {{ size: 11 }} }} }} }},
        spanGaps: true,
    }};

    // Vacancy
    new Chart(document.getElementById('chartVacancy'), {{
        type: 'line',
        data: {{
            labels: years,
            datasets: [{{
                label: 'Vacancy %',
                data: vacancy,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231,76,60,0.1)',
                fill: true, tension: 0.3, pointRadius: 4, borderWidth: 2,
            }}]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ min: 0, ticks: {{ callback: v => v+'%' }} }} }} }}
    }});

    // NOI & Revenue
    new Chart(document.getElementById('chartNOI'), {{
        type: 'bar',
        data: {{
            labels: years,
            datasets: [
                {{ label: 'Revenue', data: revenue, backgroundColor: 'rgba(52,152,219,0.6)', borderRadius: 3 }},
                {{ label: 'NOI', data: noi, backgroundColor: 'rgba(46,204,113,0.7)', borderRadius: 3 }},
            ]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ ticks: {{ callback: v => '$'+Math.round(v/1000)+'K' }} }} }} }}
    }});

    // OPEX + OPEX Ratio
    new Chart(document.getElementById('chartOPEX'), {{
        type: 'bar',
        data: {{
            labels: years,
            datasets: [
                {{ label: 'OPEX ($)', data: opex, backgroundColor: 'rgba(230,126,34,0.6)', borderRadius: 3, yAxisID: 'y' }},
                {{ label: 'OPEX Ratio (%)', data: opexRatio, type: 'line', borderColor: '#8e44ad', pointRadius: 4, borderWidth: 2, yAxisID: 'y1' }},
            ]
        }},
        options: {{
            ...commonOpts,
            scales: {{
                x: commonOpts.scales.x,
                y: {{ position: 'left', ticks: {{ callback: v => '$'+Math.round(v/1000)+'K' }} }},
                y1: {{ position: 'right', grid: {{ drawOnChartArea: false }}, ticks: {{ callback: v => v+'%' }}, min: 0, max: 100 }},
            }}
        }}
    }});

    // Loan Balance
    new Chart(document.getElementById('chartLoan'), {{
        type: 'line',
        data: {{
            labels: years,
            datasets: [{{
                label: 'Loan Balance',
                data: loan,
                borderColor: '#2c3e50',
                backgroundColor: 'rgba(44,62,80,0.08)',
                fill: true, tension: 0.2, pointRadius: 4, borderWidth: 2,
            }}]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ ticks: {{ callback: v => '$'+(v/1e6).toFixed(1)+'M' }} }} }} }}
    }});

    // DSCR
    new Chart(document.getElementById('chartDSCR'), {{
        type: 'line',
        data: {{
            labels: years,
            datasets: [
                {{ label: 'DSCR', data: dscr, borderColor: '#2980b9', pointRadius: 4, borderWidth: 2, tension: 0.3 }},
                {{ label: '1.0x Threshold', data: years.map(() => 1.0), borderColor: '#e74c3c', borderDash: [6,4], borderWidth: 1.5, pointRadius: 0 }},
                {{ label: '1.25x Threshold', data: years.map(() => 1.25), borderColor: '#f39c12', borderDash: [6,4], borderWidth: 1.5, pointRadius: 0 }},
            ]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ min: 0, ticks: {{ callback: v => v.toFixed(1)+'x' }} }} }} }}
    }});

    // NOI Margin
    new Chart(document.getElementById('chartMargin'), {{
        type: 'line',
        data: {{
            labels: years,
            datasets: [{{
                label: 'NOI Margin %',
                data: noiMargin,
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39,174,96,0.1)',
                fill: true, tension: 0.3, pointRadius: 4, borderWidth: 2,
            }}]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ min: 0, max: 100, ticks: {{ callback: v => v+'%' }} }} }} }}
    }});
    </script>
</body>
</html>"""
    
    return html


def build_index_page():
    """Build a landing page linking to all 8 property profiles."""
    cards = ""
    for name, cat, emoji, color in PROPERTIES:
        safe = name.lower().replace(" ", "_").replace(",", "").replace(".", "").replace("-", "_").replace("(", "").replace(")", "")
        cards += f"""
            <a class="card" href="profile_{safe}.html">
                <div class="card-body">
                    <div class="badge" style="background:{color};">{cat}</div>
                    <div class="name">{emoji} {name}</div>
                </div>
            </a>"""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Property Profiles — LA Self-Storage Case Studies</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; color: #333; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white; padding: 40px 20px; text-align: center;
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 8px; }}
        .header p {{ font-size: 1.05em; color: #aaa; }}
        .container {{ max-width: 900px; margin: 30px auto; padding: 0 20px; }}
        .nav {{ text-align: center; margin-bottom: 20px; }}
        .nav a {{
            display: inline-block; padding: 6px 16px; border-radius: 20px;
            background: #e9ecef; color: #495057; text-decoration: none; font-size: 0.85em;
        }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
        .card {{
            background: white; border-radius: 10px; overflow: hidden;
            box-shadow: 0 1px 5px rgba(0,0,0,0.1); transition: transform 0.2s;
            text-decoration: none; color: inherit; display: block;
        }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 14px rgba(0,0,0,0.15); }}
        .card-body {{ padding: 20px; }}
        .badge {{
            display: inline-block; padding: 3px 10px; border-radius: 16px;
            font-size: 0.75em; font-weight: 600; color: white; margin-bottom: 8px;
        }}
        .name {{ font-size: 1.1em; font-weight: 600; }}
        .footer {{ text-align: center; padding: 30px; color: #999; font-size: 0.85em; margin-top: 30px; }}
        @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LA Self-Storage — 8 Property Profiles</h1>
        <p>Vacancy trajectories through the cycle: Low→High, High→Low, Always Low, Always High</p>
    </div>
    <div class="container">
        <div class="nav"><a href="index.html">← Back to Maps</a></div>
        <div class="grid">{cards}
        </div>
    </div>
    <div class="footer">TerraCotta Group — CRE Research | Data: Trepp CMBS</div>
</body>
</html>"""


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / 'self_storage_universe.csv', low_memory=False)
    la = df[df['MSA_NAME'].fillna('').str.contains('Los Angeles', case=False)].copy()
    
    num_cols = ['VACANCY_RATE', 'OCCUPANCY_CURRENT', 'NOI_CURRENT', 'REVENUE_CURRENT',
                'OPEX_CURRENT', 'DSCR_CURRENT', 'LOAN_CURRENT_BALANCE', 'REPORT_YEAR',
                'REPORT_MONTH', 'IS_WATCHLIST', 'IS_SPECIAL_SERVICING', 'IS_DELINQUENT',
                'YEAR_BUILT', 'UNITS', 'SQFT_CURRENT']
    for c in num_cols:
        if c in la.columns:
            la[c] = pd.to_numeric(la[c], errors='coerce')
    
    profiles_dir = OUTPUT_DIR
    
    for name, cat, emoji, color in PROPERTIES:
        print(f"\n  Building profile: {name}")
        prop = la[la['PROPERTY_NAME'] == name].copy()
        
        if len(prop) == 0:
            print(f"    ⚠ NOT FOUND — skipping")
            continue
        
        # Dedup: one row per year (latest month)
        prop = prop.sort_values(['REPORT_YEAR', 'REPORT_MONTH'])
        prop = prop.drop_duplicates(subset=['TREPPMASTERPROPERTYID', 'REPORT_YEAR'], keep='last')
        prop = prop.sort_values('REPORT_YEAR')
        
        print(f"    {len(prop)} years of data ({int(prop['REPORT_YEAR'].min())}–{int(prop['REPORT_YEAR'].max())})")
        
        html = build_property_page(prop, name, cat, emoji, color)
        
        safe = name.lower().replace(" ", "_").replace(",", "").replace(".", "").replace("-", "_").replace("(", "").replace(")", "")
        path = profiles_dir / f"profile_{safe}.html"
        with open(path, 'w') as f:
            f.write(html)
        print(f"    Saved: {path.name}")
    
    # Build index
    idx_html = build_index_page()
    idx_path = profiles_dir / "property_profiles.html"
    with open(idx_path, 'w') as f:
        f.write(idx_html)
    print(f"\n  Saved profiles index: {idx_path.name}")
    
    print("\n✅ All 8 property profile pages generated.")


if __name__ == "__main__":
    main()
