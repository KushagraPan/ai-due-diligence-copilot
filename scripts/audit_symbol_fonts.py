"""
Audit script for symbol fonts and CP1252 smart punctuation in raw PDFs.

Scans all PDFs in data/raw/ using PyMuPDF page.get_text("dict"), analyzes font
names, and verifies against confirmed SYMBOL_CODEPOINT_MAP in parser.py.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

from due_diligence.ingestion import (
    get_symbol_font_category,
    is_symbol_font,
    load_symbol_mappings,
    strip_subset_prefix,
)

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        import logging

        logging.getLogger(__name__).debug("Could not reconfigure stdout encoding")


def run_audit(raw_dir: Path = Path("data/raw")):
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {raw_dir}")
        return

    symbol_map, _ = load_symbol_mappings()
    symbol_counts = Counter()  # (font_name, codepoint, char, is_mapped) -> count
    symbol_contexts = defaultdict(list)
    unmapped_symbol_counts = Counter()

    cp1252_counts = Counter()
    cp1252_contexts = defaultdict(list)

    total_pages = 0

    for pdf_path in pdf_files:
        print(f"Auditing {pdf_path.name}...")
        doc = pymupdf.open(pdf_path)
        total_pages += len(doc)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block.get("lines", []):
                    line_spans = line.get("spans", [])
                    line_text = "".join(s.get("text", "") for s in line_spans).strip()

                    for span in line_spans:
                        font = span.get("font", "UnknownFont")
                        clean_font = strip_subset_prefix(font)
                        text = span.get("text", "")
                        if not text:
                            continue

                        if is_symbol_font(font):
                            category = get_symbol_font_category(font)
                            for char in text:
                                cp = ord(char)
                                is_mapped = (category, cp) in symbol_map
                                key = (clean_font, cp, char, is_mapped)
                                symbol_counts[key] += 1
                                if not is_mapped:
                                    unmapped_symbol_counts[(clean_font, cp, char)] += 1
                                if len(symbol_contexts[(font, cp)]) < 3:
                                    context_sample = f"p.{page_num} [{line_text[:70]}]"
                                    symbol_contexts[(font, cp)].append(context_sample)
                        else:
                            for char in text:
                                cp = ord(char)
                                if (0x80 <= cp <= 0x9F) or cp == 0xFFFD:
                                    key = (font, cp, char)
                                    cp1252_counts[key] += 1
                                    if len(cp1252_contexts[(font, cp)]) < 3:
                                        context_sample = f"p.{page_num} [{line_text[:70]}]"
                                        cp1252_contexts[(font, cp)].append(context_sample)

        doc.close()

    print("\n" + "=" * 90)
    print(f"AUDIT SUMMARY: Scanned {len(pdf_files)} PDF(s), {total_pages} total pages.")
    print("=" * 90)

    print("\n" + "-" * 90)
    print("1. SYMBOL FONT CODEPOINTS STATUS ACROSS CORPUS:")
    print("-" * 90)
    print(
        f"{'Font Name':<28} {'Codepoint':<14} {'Char':<6} {'Count':<8} {'Mapped?':<9} Sample Context"
    )
    print("-" * 90)
    for (font, cp, char, is_mapped), count in symbol_counts.most_common():
        hex_cp = f"0x{cp:04X} ({cp})"
        repr_char = repr(char)
        status_str = "YES" if is_mapped else "NO (UNMAPPED)"
        samples = "; ".join(symbol_contexts[(font, cp)][:2])
        print(f"{font:<28} {hex_cp:<14} {repr_char:<6} {count:<8} {status_str:<9} {samples}")

    print("\n" + "-" * 90)
    print("2. UNMAPPED SYMBOL CODEPOINTS:")
    print("-" * 90)
    if not unmapped_symbol_counts:
        print("ZERO unmapped symbol-font codepoints remaining. All 100% covered by parser mapping!")
    else:
        for (font, cp, char), count in unmapped_symbol_counts.most_common():
            print(f"UNMAPPED: {font} 0x{cp:04X} {char!r} ({count} occurrences)")

    print("\n" + "-" * 90)
    print("3. NON-SYMBOL FONT CP1252 / LEAKED BYTES (0x80-0x9F, 0xFFFD):")
    print("-" * 90)
    if not cp1252_counts:
        print("None found in raw text spans. (Handled via TEXT_PUNCTUATION_MAP in parser).")
    else:
        for (font, cp, char), count in cp1252_counts.most_common():
            print(f"Found: {font} 0x{cp:04X} ({count} occurrences)")


if __name__ == "__main__":
    run_audit()
