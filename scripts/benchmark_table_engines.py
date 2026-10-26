"""
Comprehensive benchmark comparing PyMuPDF, pdfplumber, and Docling
on representative tables from HDFC Bank Annual Report FY25-26.

Evaluates:
- Column alignment
- Row alignment
- Headers (multi-line / hierarchical)
- Multi-line cells
- Merged cells
- Numbers and formatting
- Dashes (— / – / -) and nil markers
- Tables without visible borders (borderless)
- Page-spanning tables
- Footnotes
- Latency & dependency footprint
"""

import json
import time
from pathlib import Path

import pdfplumber
import pymupdf
from docling.document_converter import DocumentConverter

PDF_PATH = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")
SCRATCH_DIR = Path("scratch")
SCRATCH_DIR.mkdir(exist_ok=True)

# Test cases representing distinct diligence table typologies
TEST_CASES = [
    {
        "id": "balance_sheet",
        "name": "Standalone Balance Sheet",
        "page": 378,
        "description": "Standard financial statement with sparse horizontal lines, multi-line row labels, schedule references, large integers.",
        "key_checks": [
            "Capital line item with Schedule 1 and FY26/FY25 amounts",
            "Multi-line row 'Employees stock options / units outstanding'",
            "Schedule column isolation (1, 2, 3, etc.)",
            "Separation of table from bottom auditor/director signature block",
        ],
    },
    {
        "id": "profit_loss",
        "name": "Standalone Profit and Loss Account",
        "page": 379,
        "description": "Multi-column income statement with hierarchical revenue/expenditure sections, schedules, and earnings per share.",
        "key_checks": [
            "Interest earned vs Interest expended",
            "Multi-line cell 'Payments to and provisions for employees'",
            "Schedule column (13, 14, 15, 16)",
            "Earnings per equity share (Basic and Diluted)",
        ],
    },
    {
        "id": "board_attendance",
        "name": "Board of Directors Meeting Attendance",
        "page": 343,
        "description": "Governance disclosure table with director names, meeting attendance ratios, AGM attendance, and explanatory footnotes.",
        "key_checks": [
            "Director names (multi-word)",
            "Attendance numbers (e.g. 21 of 21, 20 of 21)",
            "Exclusion of running header 'Statutory Reports and Financial Statements'",
            "Footnote handling (* Held during tenure)",
        ],
    },
    {
        "id": "capital_adequacy",
        "name": "Capital Adequacy & Basel III Ratios",
        "page": 402,
        "description": "Regulatory disclosure with sparse borders, percentages, ratios, and sub-headers.",
        "key_checks": [
            "Common Equity Tier 1 ratio",
            "Tier 1 capital ratio",
            "Percentages formatting (e.g. 16.5%)",
            "Handling of borderless lines",
        ],
    },
    {
        "id": "segment_reporting",
        "name": "Segment Reporting (Business Segments)",
        "page": 468,
        "description": "Multi-column business segment table (Treasury, Retail, Wholesale, Other) with nil markers and totals.",
        "key_checks": [
            "Multi-column headers across segments",
            "Preservation of en/em-dashes for nil/unallocated items",
            "Segment revenue and segment results row alignment",
        ],
    },
]


def extract_single_page_pdf(page_num: int) -> Path:
    """Extract 1-page PDF fixture for fast, isolated evaluation."""
    target_path = SCRATCH_DIR / f"test_page_{page_num}.pdf"
    if not target_path.exists():
        src_doc = pymupdf.open(PDF_PATH)
        out_doc = pymupdf.open()
        out_doc.insert_pdf(src_doc, from_page=page_num - 1, to_page=page_num - 1)
        out_doc.save(target_path)
        out_doc.close()
        src_doc.close()
    return target_path


def run_pymupdf(pdf_file: Path) -> dict:
    start_t = time.perf_counter()
    doc = pymupdf.open(pdf_file)
    page = doc[0]

    # Try both lines and text strategies
    tables_lines = page.find_tables(strategy="lines")
    tables_text = page.find_tables(strategy="text")

    res = {
        "engine": "PyMuPDF find_tables()",
        "time_ms": round((time.perf_counter() - start_t) * 1000, 2),
        "lines_count": len(tables_lines.tables),
        "text_count": len(tables_text.tables),
        "lines_tables": [
            {
                "shape": (t.row_count, t.col_count),
                "bbox": list(t.bbox),
                "headers": t.header.names if t.header else [],
                "sample_rows": t.extract()[:6],
                "last_rows": t.extract()[-3:],
            }
            for t in tables_lines.tables
        ],
        "text_tables": [
            {
                "shape": (t.row_count, t.col_count),
                "bbox": list(t.bbox),
                "headers": t.header.names if t.header else [],
                "sample_rows": t.extract()[:6],
                "last_rows": t.extract()[-3:],
            }
            for t in tables_text.tables
        ],
    }
    doc.close()
    return res


def run_pdfplumber(pdf_file: Path) -> dict:
    start_t = time.perf_counter()
    with pdfplumber.open(pdf_file) as doc:
        page = doc.pages[0]

        # 1. Default (lines)
        tabs_default = page.extract_tables()
        # 2. Text strategy
        tabs_text = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
        # 3. Explicit / text vertical + lines horizontal
        tabs_hybrid = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "lines"})

    return {
        "engine": "pdfplumber",
        "time_ms": round((time.perf_counter() - start_t) * 1000, 2),
        "default_count": len(tabs_default),
        "text_count": len(tabs_text),
        "hybrid_count": len(tabs_hybrid),
        "default_tables": [
            {
                "shape": (len(t), len(t[0]) if t else 0),
                "sample_rows": t[:6],
                "last_rows": t[-3:],
            }
            for t in tabs_default
        ],
        "text_tables": [
            {
                "shape": (len(t), len(t[0]) if t else 0),
                "sample_rows": t[:6],
                "last_rows": t[-3:],
            }
            for t in tabs_text
        ],
    }


def run_docling(pdf_file: Path) -> dict:
    start_t = time.perf_counter()
    converter = DocumentConverter()
    res = converter.convert(pdf_file)
    doc = res.document

    tables_out = []
    for t in doc.tables:
        df = t.export_to_dataframe()
        markdown = t.export_to_markdown()
        tables_out.append({
            "shape": df.shape,
            "columns": [str(c) for c in df.columns.tolist()],
            "sample_rows": df.head(6).values.tolist(),
            "last_rows": df.tail(3).values.tolist(),
            "markdown_snippet": "\n".join(markdown.splitlines()[:12]),
        })

    return {
        "engine": "Docling",
        "time_ms": round((time.perf_counter() - start_t) * 1000, 2),
        "table_count": len(doc.tables),
        "tables": tables_out,
    }


def main():
    benchmark_results = {}

    for tc in TEST_CASES:
        p_num = tc["page"]
        tc_id = tc["id"]
        print("\n================================================================================")
        print(f"BENCHMARKING: {tc['name']} (Page {p_num})")
        print(f"Description: {tc['description']}")
        print("================================================================================")

        page_pdf = extract_single_page_pdf(p_num)

        print("  Running PyMuPDF...")
        res_mupdf = run_pymupdf(page_pdf)
        print(f"    -> Done in {res_mupdf['time_ms']} ms")

        print("  Running pdfplumber...")
        res_plumber = run_pdfplumber(page_pdf)
        print(f"    -> Done in {res_plumber['time_ms']} ms")

        print("  Running Docling...")
        res_docling = run_docling(page_pdf)
        print(f"    -> Done in {res_docling['time_ms']} ms")

        benchmark_results[tc_id] = {
            "test_case": tc,
            "pymupdf": res_mupdf,
            "pdfplumber": res_plumber,
            "docling": res_docling,
        }

    out_file = SCRATCH_DIR / "table_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full benchmark results to {out_file}")


if __name__ == "__main__":
    main()
