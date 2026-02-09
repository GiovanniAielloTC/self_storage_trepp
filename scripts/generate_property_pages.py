"""
Generate 8 Interactive Property Profile Pages — Monthly Resolution
==================================================================
Uses raw monthly Trepp CMBS data so that:
  - Loan balance is a smooth monthly amortisation curve
  - NOI / OPEX / Revenue / DSCR / Vacancy are plotted at their ACTUAL
    filing cadence (quarterly, semi-annual, or annual) — NOT padded to
    every month.

Selection criteria:
  2 x Start Low -> End High vacancy
  2 x Start High -> End Low vacancy
  2 x Always Low vacancy
  2 x Always High vacancy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

OUTPUT_DIR = Path('/home/bizon/giovanni/self_storage_trepp/output')
DATA_DIR = Path('/home/bizon/giovanni/self_storage_trepp/data')

# --- Selected properties -------------------------------------------------
PROPERTIES = [
    # (name, category, emoji, color)
    ("Lockaway SS - Hollywood",        "Low -> High Vacancy",  "\U0001f4c8", "#e74c3c"),
    ("Cypress Mini Storage",           "Low -> High Vacancy",  "\U0001f4c8", "#e74c3c"),
    ("Storage Center of Valencia",     "High -> Low Vacancy",  "\U0001f4c9", "#27ae60"),
    ("Storage Express - Palmdale, CA", "High -> Low Vacancy",  "\U0001f4c9", "#27ae60"),
    ("BV Hollywood Storage",           "Always Low Vacancy",   "\U0001f7e2", "#2ecc71"),
    ("Storage Etc. - Torrance, CA",    "Always Low Vacancy",   "\U0001f7e2", "#2ecc71"),
    ("SoCal Torrance",                 "Always High Vacancy",  "\U0001f534", "#c0392b"),
    ("Storbox Self Storage",           "Always High Vacancy",  "\U0001f534", "#c0392b"),
]


def _filing_cadence(series, date_idx):
    """Detect filing cadence string from a series of values + date indices."""
    vals = series.dropna()
    if len(vals) < 3:
        return "Unknown"
    changed = vals.diff().ne(0)
    idx = date_idx.loc[changed.index][changed]
    if len(idx) < 2:
        return "Unknown"
    gaps = idx.diff().dropna()
    avg = gaps.mean()
    if avg > 9:
        return "Annual"
    elif avg > 4.5:
        return "Semi-Annual"
    else:
        return "Quarterly"


def _dedupe_opst(sub, col):
    """Return only rows where an operating-statement metric actually changed."""
    s = sub[['date_label', col]].dropna(subset=[col]).copy()
    if len(s) == 0:
        return [], []
    s['changed'] = s[col].diff().ne(0)
    s = s[s['changed']]
    return s['date_label'].tolist(), [round(float(v), 2) for v in s[col]]


def build_property_page(monthly_df, meta, prop_name, category, emoji, cat_color, filing_freq):
    """Build a single-page interactive HTML dashboard for one property."""

    sub = monthly_df.sort_values(['REPORT_YEAR', 'REPORT_MONTH']).copy()
    sub['date_label'] = sub.apply(lambda r: f"{int(r.REPORT_YEAR)}-{int(r.REPORT_MONTH):02d}", axis=1)

    # -- Static metadata from yearly universe file --
    city = meta.get('CITY', '')
    state = meta.get('STATE', '')
    address = meta.get('ADDRESS', '')
    zipcode = meta.get('ZIPCODE', '')
    year_built = meta.get('YEAR_BUILT')
    yr_built_str = str(int(year_built)) if pd.notna(year_built) else "N/A"

    # -- Monthly loan balance (continuous) --
    all_labels = sub['date_label'].tolist()
    loan_vals_raw = [round(float(v), 2) if pd.notna(v) else None for v in sub['LOAN_CURRENT_BALANCE']]

    # Trim trailing $0 months (loan paid off / defeased)
    # Find the last month with a non-zero balance
    last_nonzero_idx = None
    for i in range(len(loan_vals_raw) - 1, -1, -1):
        if loan_vals_raw[i] is not None and loan_vals_raw[i] > 0:
            last_nonzero_idx = i
            break

    loan_paid_off = False
    loan_payoff_date = None
    if last_nonzero_idx is not None and last_nonzero_idx < len(loan_vals_raw) - 1:
        # There are $0 months after the last non-zero → loan paid off
        loan_paid_off = True
        payoff_idx = last_nonzero_idx + 1
        loan_payoff_date = all_labels[payoff_idx]
        # Keep up to and including the first $0 month to show the payoff
        loan_labels = all_labels[:payoff_idx + 1]
        loan_vals = loan_vals_raw[:payoff_idx + 1]
    elif last_nonzero_idx is None:
        # All zeros or null — could be IO or never funded; show all
        loan_labels = all_labels
        loan_vals = loan_vals_raw
    else:
        loan_labels = all_labels
        loan_vals = loan_vals_raw

    # -- Operating statement metrics: keep only change-points --
    noi_labels,  noi_vals  = _dedupe_opst(sub, 'NOI_CURRENT')
    rev_labels,  rev_vals  = _dedupe_opst(sub, 'REVENUE_CURRENT')
    opex_labels, opex_vals = _dedupe_opst(sub, 'OPEX_CURRENT')
    dscr_labels, dscr_vals = _dedupe_opst(sub, 'DSCR_CURRENT')
    vac_labels,  vac_vals  = _dedupe_opst(sub, 'VACANCY_RATE')

    # -- OPEX ratio & NOI margin at change-points --
    opex_ratio_labels, opex_ratio_vals = [], []
    noi_margin_labels, noi_margin_vals = [], []
    merged = sub[['date_label', 'NOI_CURRENT', 'REVENUE_CURRENT', 'OPEX_CURRENT']].dropna(
        subset=['NOI_CURRENT', 'REVENUE_CURRENT', 'OPEX_CURRENT']).copy()
    if len(merged) > 0:
        merged['changed'] = (merged['NOI_CURRENT'].diff().ne(0) |
                             merged['REVENUE_CURRENT'].diff().ne(0) |
                             merged['OPEX_CURRENT'].diff().ne(0))
        merged = merged[merged['changed']]
        for _, row in merged.iterrows():
            rev = row['REVENUE_CURRENT']
            if rev and rev > 0:
                opex_ratio_labels.append(row['date_label'])
                opex_ratio_vals.append(round(row['OPEX_CURRENT'] / rev * 100, 1))
                noi_margin_labels.append(row['date_label'])
                noi_margin_vals.append(round(row['NOI_CURRENT'] / rev * 100, 1))

    # -- Latest KPI values --
    latest_noi = noi_vals[-1] if noi_vals else None
    latest_noi_date = noi_labels[-1] if noi_labels else None
    latest_vac = vac_vals[-1] if vac_vals else None
    latest_vac_date = vac_labels[-1] if vac_labels else None
    latest_dscr = dscr_vals[-1] if dscr_vals else None
    latest_dscr_date = dscr_labels[-1] if dscr_labels else None
    # For loan KPI, show last non-zero balance (not the $0 payoff)
    latest_loan, latest_loan_date = None, None
    for v, lbl in zip(reversed(loan_vals_raw), reversed(all_labels)):
        if v is not None and v > 0:
            latest_loan, latest_loan_date = v, lbl
            break
    # If loan fully paid off, override
    if loan_paid_off:
        latest_loan = 0
        latest_loan_date = loan_payoff_date

    yr_min = int(sub['REPORT_YEAR'].min())
    yr_max = int(sub['REPORT_YEAR'].max())
    n_months = len(sub)
    is_watchlist = meta.get('IS_WATCHLIST', 0)

    def fmt_money(v):
        if v is None: return "N/A"
        if isinstance(v, float) and np.isnan(v): return "N/A"
        return f"${v:,.0f}"

    def fmt_pct(v):
        if v is None: return "N/A"
        if isinstance(v, float) and np.isnan(v): return "N/A"
        return f"{v:.1f}%"

    # -- Data table (at operating-statement change-points) --
    table_dates = sorted(set(noi_labels) | set(vac_labels) | set(opex_labels))
    if not table_dates:
        table_dates = all_labels[::12]

    noi_dict = dict(zip(noi_labels, noi_vals))
    rev_dict = dict(zip(rev_labels, rev_vals))
    opex_dict = dict(zip(opex_labels, opex_vals))
    dscr_dict = dict(zip(dscr_labels, dscr_vals))
    vac_dict = dict(zip(vac_labels, vac_vals))
    loan_dict = dict(zip(all_labels, loan_vals))
    opex_ratio_dict = dict(zip(opex_ratio_labels, opex_ratio_vals))
    noi_margin_dict = dict(zip(noi_margin_labels, noi_margin_vals))

    def js(arr):
        return json.dumps(arr)

    vac_color = '#e74c3c' if (latest_vac or 0) > 15 else '#27ae60'
    dscr_color = '#e74c3c' if (latest_dscr or 0) < 1.25 else '#27ae60'
    dscr_str = f'{latest_dscr:.2f}x' if latest_dscr else 'N/A'
    wl_color = '#e74c3c' if is_watchlist == 1 else '#27ae60'
    wl_txt = 'Yes' if is_watchlist == 1 else 'No'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{prop_name} — Property Profile (Monthly)</title>
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
        .header .freq {{
            display: inline-block; margin-top: 6px; padding: 3px 12px;
            border-radius: 16px; font-size: 0.78em; font-weight: 500;
            background: rgba(255,255,255,0.15); color: #ddd;
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
        .chart-card .note {{
            font-size: 0.72em; color: #aaa; margin-top: 4px; font-style: italic;
        }}
        canvas {{ max-height: 300px; }}
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
        <div class="freq">Operating Statements Filed: {filing_freq} &nbsp;|&nbsp; Loan: Monthly</div>
    </div>
    <div class="container">
        <div class="nav">
            <a href="index.html">&larr; All Maps</a>
            <a href="property_profiles.html">All 8 Profiles</a>
        </div>

        <div class="kpi-row">
            <div class="kpi">
                <div class="label">Latest Vacancy</div>
                <div class="value" style="color: {vac_color};">{fmt_pct(latest_vac)}</div>
                <div class="sub">{latest_vac_date or '—'}</div>
            </div>
            <div class="kpi">
                <div class="label">Latest NOI</div>
                <div class="value">{fmt_money(latest_noi)}</div>
                <div class="sub">{latest_noi_date or '—'}</div>
            </div>
            <div class="kpi">
                <div class="label">Latest DSCR</div>
                <div class="value" style="color: {dscr_color};">{dscr_str}</div>
                <div class="sub">{latest_dscr_date or '—'}</div>
            </div>
            <div class="kpi">
                <div class="label">Loan Balance</div>
                <div class="value">{'Paid Off' if loan_paid_off else fmt_money(latest_loan)}</div>
                <div class="sub">{('Defeased ' + loan_payoff_date) if loan_paid_off else (latest_loan_date or '—')}</div>
            </div>
            <div class="kpi">
                <div class="label">Data Span</div>
                <div class="value" style="font-size:1.1em;">{yr_min}&ndash;{yr_max}</div>
                <div class="sub">{n_months} monthly obs</div>
            </div>
            <div class="kpi">
                <div class="label">Watchlist</div>
                <div class="value" style="color: {wl_color};">{wl_txt}</div>
                <div class="sub">&nbsp;</div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-card">
                <h3>Vacancy Rate (%) — from Loan Table</h3>
                <canvas id="chartVacancy"></canvas>
                <div class="note">Source: MOSTRECENTPHYSICALOCCUPANCY (loan remittance tape, updated with each operating statement)</div>
            </div>
            <div class="chart-card">
                <h3>NOI &amp; Revenue ($) — from Operating Statement</h3>
                <canvas id="chartNOI"></canvas>
                <div class="note">{filing_freq} operating statements ({len(noi_vals)} filings)</div>
            </div>
            <div class="chart-card">
                <h3>OPEX &amp; OPEX Ratio — from Operating Statement</h3>
                <canvas id="chartOPEX"></canvas>
                <div class="note">{filing_freq} operating statements ({len(opex_vals)} filings)</div>
            </div>
            <div class="chart-card">
                <h3>Loan Balance — Monthly Amortisation ($)</h3>
                <canvas id="chartLoan"></canvas>
                <div class="note">{'Loan paid off / defeased ' + loan_payoff_date + '. Post-payoff $0 months trimmed.' if loan_paid_off else 'Every monthly remittance tape observation'}</div>
            </div>
            <div class="chart-card">
                <h3>DSCR (Debt Service Coverage) \u2014 from Loan Table</h3>
                <canvas id="chartDSCR"></canvas>
                <div class="note">Source: MOSTRECENTDSCR_NOI ({len(dscr_vals)} filings)</div>
            </div>
            <div class="chart-card">
                <h3>NOI Margin (%) \u2014 from Operating Statement</h3>
                <canvas id="chartMargin"></canvas>
                <div class="note">{filing_freq} operating statements ({len(noi_vals)} filings)</div>
            </div>
        </div>

        <div class="data-table">
            <h3>Operating Statement Data (Filed {filing_freq})</h3>
            <table>
                <thead>
                    <tr>
                        <th>Date</th><th>Vacancy</th><th>NOI</th><th>Revenue</th>
                        <th>OPEX</th><th>OPEX Ratio</th><th>NOI Margin</th>
                        <th>DSCR</th><th>Loan Bal.</th>
                    </tr>
                </thead>
                <tbody>
"""

    for d in table_dates:
        v = fmt_pct(vac_dict[d]) if d in vac_dict else "—"
        n = fmt_money(noi_dict[d]) if d in noi_dict else "—"
        r = fmt_money(rev_dict[d]) if d in rev_dict else "—"
        o = fmt_money(opex_dict[d]) if d in opex_dict else "—"
        oratio = f"{opex_ratio_dict[d]:.1f}%" if d in opex_ratio_dict else "—"
        nm = f"{noi_margin_dict[d]:.1f}%" if d in noi_margin_dict else "—"
        dc = f"{dscr_dict[d]:.2f}x" if d in dscr_dict else "—"
        lb = fmt_money(loan_dict.get(d)) if loan_dict.get(d) is not None else "—"
        html += f"                    <tr><td>{d}</td><td>{v}</td><td>{n}</td><td>{r}</td><td>{o}</td><td>{oratio}</td><td>{nm}</td><td>{dc}</td><td>{lb}</td></tr>\n"

    html += f"""                </tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        <p>TerraCotta Group — CRE Research | Data: Trepp CMBS via Snowflake (monthly resolution)</p>
        <p><a href="index.html">&larr; Back to Maps</a> &nbsp;|&nbsp; <a href="property_profiles.html">All Profiles</a></p>
    </div>

    <script>
    const loanLabels = {js(loan_labels)};
    const loanData   = {js(loan_vals)};
    const vacLabels  = {js(vac_labels)};
    const vacData    = {js(vac_vals)};
    const noiLabels  = {js(noi_labels)};
    const noiData    = {js(noi_vals)};
    const revLabels  = {js(rev_labels)};
    const revData    = {js(rev_vals)};
    const opexLabels = {js(opex_labels)};
    const opexData   = {js(opex_vals)};
    const dscrLabels = {js(dscr_labels)};
    const dscrData   = {js(dscr_vals)};
    const opexRatioLabels = {js(opex_ratio_labels)};
    const opexRatioData   = {js(opex_ratio_vals)};
    const noiMarginLabels = {js(noi_margin_labels)};
    const noiMarginData   = {js(noi_margin_vals)};

    const commonOpts = {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
            legend: {{ display: true, position: 'top', labels: {{ font: {{ size: 11 }} }} }},
            tooltip: {{ mode: 'index', intersect: false }}
        }},
        scales: {{
            x: {{ ticks: {{ font: {{ size: 10 }}, maxRotation: 45, autoSkip: true, maxTicksLimit: 18 }} }},
            y: {{ ticks: {{ font: {{ size: 11 }} }} }}
        }},
        spanGaps: true,
    }};

    new Chart(document.getElementById('chartVacancy'), {{
        type: 'line',
        data: {{
            labels: vacLabels,
            datasets: [{{
                label: 'Vacancy %',
                data: vacData,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231,76,60,0.1)',
                fill: true, tension: 0.3, pointRadius: 5, borderWidth: 2,
            }}]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ min: 0, ticks: {{ callback: v => v+'%' }} }} }} }}
    }});

    new Chart(document.getElementById('chartNOI'), {{
        type: 'bar',
        data: {{
            labels: noiLabels,
            datasets: [
                {{ label: 'Revenue', data: revData, backgroundColor: 'rgba(52,152,219,0.6)', borderRadius: 3 }},
                {{ label: 'NOI', data: noiData, backgroundColor: 'rgba(46,204,113,0.7)', borderRadius: 3 }},
            ]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ ticks: {{ callback: v => '$'+Math.round(v/1000)+'K' }} }} }} }}
    }});

    new Chart(document.getElementById('chartOPEX'), {{
        type: 'bar',
        data: {{
            labels: opexLabels,
            datasets: [
                {{ label: 'OPEX ($)', data: opexData, backgroundColor: 'rgba(230,126,34,0.6)', borderRadius: 3, yAxisID: 'y' }},
                {{ label: 'OPEX Ratio (%)', data: opexRatioData, type: 'line', borderColor: '#8e44ad',
                   pointRadius: 4, borderWidth: 2, yAxisID: 'y1' }},
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

    new Chart(document.getElementById('chartLoan'), {{
        type: 'line',
        data: {{
            labels: loanLabels,
            datasets: [{{
                label: 'Loan Balance',
                data: loanData,
                borderColor: '#2c3e50',
                backgroundColor: 'rgba(44,62,80,0.08)',
                fill: true, tension: 0.1, pointRadius: 0, borderWidth: 2,
            }}]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ ticks: {{ callback: v => '$'+(v/1e6).toFixed(1)+'M' }} }} }} }}
    }});

    new Chart(document.getElementById('chartDSCR'), {{
        type: 'line',
        data: {{
            labels: dscrLabels,
            datasets: [
                {{ label: 'DSCR', data: dscrData, borderColor: '#2980b9', pointRadius: 5, borderWidth: 2, tension: 0.3 }},
                {{ label: '1.0x', data: dscrLabels.map(() => 1.0), borderColor: '#e74c3c', borderDash: [6,4], borderWidth: 1.5, pointRadius: 0 }},
                {{ label: '1.25x', data: dscrLabels.map(() => 1.25), borderColor: '#f39c12', borderDash: [6,4], borderWidth: 1.5, pointRadius: 0 }},
            ]
        }},
        options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ min: 0, ticks: {{ callback: v => v.toFixed(1)+'x' }} }} }} }}
    }});

    new Chart(document.getElementById('chartMargin'), {{
        type: 'line',
        data: {{
            labels: noiMarginLabels,
            datasets: [{{
                label: 'NOI Margin %',
                data: noiMarginData,
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39,174,96,0.1)',
                fill: true, tension: 0.3, pointRadius: 5, borderWidth: 2,
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
        <p>Monthly resolution: loan amortisation curves + operating statements at actual filing cadence</p>
    </div>
    <div class="container">
        <div class="nav"><a href="index.html">&larr; Back to Maps</a></div>
        <div class="grid">{cards}
        </div>
    </div>
    <div class="footer">TerraCotta Group — CRE Research | Data: Trepp CMBS (monthly remittance tape)</div>
</body>
</html>"""


def main():
    print("Loading monthly raw data...")
    monthly = pd.read_csv(DATA_DIR / 'profiles_monthly_raw.csv')
    print(f"  {len(monthly)} monthly rows for {monthly['PROPERTY_NAME'].nunique()} properties")

    print("Loading yearly universe for metadata...")
    universe = pd.read_csv(DATA_DIR / 'self_storage_universe.csv', low_memory=False)
    la = universe[universe['MSA_NAME'].fillna('').str.contains('Los Angeles', case=False)].copy()

    for name, cat, emoji, color in PROPERTIES:
        print(f"\n  Building profile: {name}")

        prop_monthly = monthly[monthly['PROPERTY_NAME'] == name].copy()
        if len(prop_monthly) == 0:
            print(f"    WARNING: NOT FOUND in monthly data — skipping")
            continue

        # Get static metadata from yearly file
        prop_yearly = la[la['PROPERTY_NAME'] == name]
        if len(prop_yearly) > 0:
            latest_yr = prop_yearly.sort_values('REPORT_YEAR').iloc[-1]
            meta = latest_yr.to_dict()
        else:
            meta = {}

        # Detect filing frequency
        sub = prop_monthly.sort_values(['REPORT_YEAR', 'REPORT_MONTH']).copy()
        sub['date_idx'] = sub['REPORT_YEAR'] * 12 + sub['REPORT_MONTH']
        freq = _filing_cadence(sub['NOI_CURRENT'], sub['date_idx'])
        print(f"    {len(prop_monthly)} monthly rows | Filing: {freq}")

        html = build_property_page(prop_monthly, meta, name, cat, emoji, color, freq)

        safe = name.lower().replace(" ", "_").replace(",", "").replace(".", "").replace("-", "_").replace("(", "").replace(")", "")
        path = OUTPUT_DIR / f"profile_{safe}.html"
        with open(path, 'w') as f:
            f.write(html)
        print(f"    Saved: {path.name}")

    # Build index
    idx_html = build_index_page()
    idx_path = OUTPUT_DIR / "property_profiles.html"
    with open(idx_path, 'w') as f:
        f.write(idx_html)
    print(f"\n  Saved profiles index: {idx_path.name}")

    print("\nAll 8 property profile pages generated (monthly resolution).")


if __name__ == "__main__":
    main()
