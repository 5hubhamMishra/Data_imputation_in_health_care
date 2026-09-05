"""Export docs/supervisor_progress_summary.md to a supervisor-ready PDF.

Master prompt section 42 ("Automatic Supervisor Report Export"): a clean,
supervisor-ready PDF at reports/RP_Progress_Report.pdf, readable without
inspecting source code, containing only verified current progress plus a
couple of the most important existing result tables/figures.

This script does not compute or interpret anything new. It renders the
already-written, already-validated docs/supervisor_progress_summary.md as
HTML and appends two existing result tables (read directly from their CSV
files) and two existing figures (embedded as already-generated PNGs), then
converts the HTML to PDF with xhtml2pdf.

Run: python scripts/generate_progress_report_pdf.py
"""
from __future__ import annotations

import base64
import csv
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_MD = ROOT / "docs" / "supervisor_progress_summary.md"
OUTPUT_PDF = ROOT / "reports" / "RP_Progress_Report.pdf"

# The most important report-ready table/figure pairs to include, per section 42's
# "include selected report-ready figures/tables when available".
TABLES = [
    ("Four-way comparison (E0 / E1-E3 / E5 / E6)", ROOT / "results" / "tables" / "four_way_comparison.csv"),
    ("GA feature selection frequency (5 runs)", ROOT / "results" / "tables" / "ga_feature_frequency.csv"),
]
FIGURES = [
    ("Four-way comparison", ROOT / "results" / "figures" / "four_way_comparison.png"),
    ("GA feature selection frequency", ROOT / "results" / "figures" / "ga_feature_frequency.png"),
]

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.4; }
h1 { font-size: 18pt; }
h2 { font-size: 13pt; margin-top: 20px; border-bottom: 1pt solid #333; padding-bottom: 3px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 8pt; }
th, td { border: 0.5pt solid #999; padding: 3px 5px; text-align: left; }
th { background: #eee; }
img { max-width: 100%; margin: 8px 0; }
code { background: #eee; padding: 1px 3px; }
"""


def _dashes_to_hyphens(text: str) -> str:
    # xhtml2pdf's default font renders em/en dashes as replacement characters.
    return text.replace("—", " - ").replace("–", "-")


def _csv_to_html_table(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    thead = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f"<table><tr>{thead}</tr>{trs}</table>"


def _image_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_html() -> str:
    summary_text = _dashes_to_hyphens(SUMMARY_MD.read_text(encoding="utf-8"))
    body_html = markdown.markdown(summary_text)

    tables_html = "<h2>Key Result Tables</h2>"
    for title, path in TABLES:
        tables_html += f"<h3>{title}</h3>" + _csv_to_html_table(path)

    figures_html = "<h2>Key Figures</h2>"
    for title, path in FIGURES:
        figures_html += f"<p><b>{title}</b></p><img src='{_image_to_data_uri(path)}'/>"

    return f"<html><head><style>{CSS}</style></head><body>{body_html}{tables_html}{figures_html}</body></html>"


def main() -> None:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    with OUTPUT_PDF.open("wb") as out:
        result = pisa.CreatePDF(html, dest=out)
    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s)")
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
