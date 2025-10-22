#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import base64, io, json, subprocess, os, datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

BASES = [Path("outputs_from_expected"), Path("outputs")]

def _find_base() -> Path:
    for b in BASES:
        if (b).exists():
            return b
    raise FileNotFoundError("No outputs folder found (looked for outputs_from_expected/ and outputs/)")

def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def main() -> None:
    base = _find_base()
    report_csv = base / "report_all.csv"
    if not report_csv.exists():
        raise FileNotFoundError(f"{report_csv} not found. Run `make report` first.")
    df = pd.read_csv(report_csv, parse_dates=["date"]).set_index("date").sort_index()
    
    # metadata
    commit = (subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout or "").strip()
    model_uri = os.getenv("MLFLOW_MODEL_URI","") or (Path("artifacts/structured_model_uri.txt").read_text().strip() if (Path("artifacts/structured_model_uri.txt")).exists() else "")
    exp_meta = {}
    if (Path("expected/metadata.json")).exists():
        exp_meta = json.loads(Path("expected/metadata.json").read_text())
    exp_ts = df.index.max().strftime("%Y-%m-%d") if len(df) else "NA"
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Cum P&L curve
    cum = (1.0 + df["ret"].fillna(0.0)).cumprod()
    fig = plt.figure()
    cum.plot()
    cum_b64 = _b64_png(fig)

    # Rolling Sharpe (60d)
    r60 = df["ret"].rolling(60)
    sharpe60 = (r60.mean() / (r60.std() + 1e-12)) * (252**0.5)
    fig2 = plt.figure()
    sharpe60.plot()
    sharpe_b64 = _b64_png(fig2)

    # Exposures diagnostics
    fig3 = plt.figure()
    df[["net","beta","sector_l2"]].plot(ax=plt.gca())
    diag_b64 = _b64_png(fig3)

    # Simple HTML
    html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>MAIE Report</title>
<style>
body{{font-family:ui-sans-serif,system-ui,Arial;margin:24px}} h1{{margin:0 0 8px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:16px 0}}
img{{max-width:100%;height:auto;border-radius:8px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #eee;padding:6px 8px;text-align:right}}
th{{background:#fafafa;text-align:left}}
</style></head><body>
<h1>MAIE — PM Report</h1>
<p>Rows: {len(df):,} &nbsp; Start: {df.index.min().date()} &nbsp; End: {df.index.max().date()}</p>

<div class="card"><h2>Cumulative P&L</h2>
<img src="data:image/png;base64,{cum_b64}"/></div>

<div class="card"><h2>Rolling 60-day Sharpe</h2>
<img src="data:image/png;base64,{sharpe_b64}"/></div>

<div class="card"><h2>Exposure Diagnostics (daily)</h2>
<img src="data:image/png;base64,{diag_b64}"/></div>

<div class="card"><h2>Daily Cutout (head)</h2>
{df.head(15).to_html(classes='tbl', float_format=lambda x: f"{x:,.6f}")}
</div>

</body></html>
"""
    # Add metadata to HTML
    html = html.replace("</body></html>", f"""
    <div class="card"><h3>Run Metadata</h3>
    <ul>
      <li><b>Build time:</b> {exp_meta.get('build_seconds','NA')} s</li>
      <li><b>Expected rows×cols:</b> {exp_meta.get('shape','NA')}</li>
      <li><b>Expected span:</b> {exp_meta.get('start','NA')} → {exp_meta.get('end','NA')}</li>
      <li><b>Commit:</b> {commit}</li>
      <li><b>Model URI:</b> {model_uri}</li>
      <li><b>Generated:</b> {now}</li>
    </ul>
    </div>
    </body></html>
    """)
    
    out = base / "report.html"
    out.write_text(html)
    print(f"Wrote {out.resolve()}")

if __name__ == "__main__":
    main()
