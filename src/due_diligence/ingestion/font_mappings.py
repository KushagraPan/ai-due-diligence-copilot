"""
Symbol font mappings loader and font-name normalization utilities.

1. Purpose: Manages externalized configuration for symbol font glyph remapping
   and CP1252/Unicode smart punctuation normalization.
2. Pipeline Role: Supplies font-scoped translation tables to the PDF text extraction
   layer, ensuring consistent, audit-ready text before semantic chunking.
3. Expected Inputs: Optional config file path (defaults to config/symbol_font_mappings.yaml)
   and font strings from PDF text spans.
4. Outputs: Cached lookup dictionaries ((font_category, codepoint) -> replacement string,
   codepoint -> replacement string), normalized font names, and unmapped placeholders.
5. Dependencies: PyYAML (`yaml`), standard library (`re`, `pathlib`, `functools`, `logging`).
6. Outside Scope: PDF stream reading, layout assembly, chunking, and vector indexing.

NOTE ON MAPPING PROVENANCE & LIVING SPECIFICATION:
The initial mappings in `config/symbol_font_mappings.yaml` were derived from a single
corporate filing (HDFC Bank FY25-26 Annual Report, audited and confirmed on 2026-09-25).
This is a living, extensible configuration, not a universal one. As new documents from
different issuers and design software are ingested, new (font, codepoint) pairs will
surface and must follow the same audit-then-visually-confirm protocol before addition.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default relative location of the externalized YAML configuration
DEFAULT_CONFIG_PATH = Path("config/symbol_font_mappings.yaml")

# Strict regex patterns for canonical symbol font families
# Ensures Wingdings, Wingdings2, Wingdings3, Webdings, and Symbol resolve to distinct categories
# and prevents unrecognized variants (e.g. Wingdings4, CustomWingdings) from matching
_SYMBOL_FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "wingdings3": re.compile(
        r"^wingdings[-_\s]?3(?:[-_\s]?(?:mt|regular|bold|italic|roman|psmt))*$",
        re.IGNORECASE,
    ),
    "wingdings2": re.compile(
        r"^wingdings[-_\s]?2(?:[-_\s]?(?:mt|regular|bold|italic|roman|psmt))*$",
        re.IGNORECASE,
    ),
    "wingdings": re.compile(
        r"^wingdings(?:[-_\s]?(?:mt|regular|bold|italic|roman|psmt))*$",
        re.IGNORECASE,
    ),
    "webdings": re.compile(
        r"^webdings(?:[-_\s]?(?:mt|regular|bold|italic|roman|psmt))*$",
        re.IGNORECASE,
    ),
    "symbol": re.compile(
        r"^symbol(?:[-_\s]?(?:mt|regular|bold|italic|roman|psmt))*$",
        re.IGNORECASE,
    ),
}

# Broad pattern to identify symbol/icon fonts so unmapped variants do not leak raw text
_BROAD_SYMBOL_FONT_PATTERN = re.compile(
    r"(?:wingdings|webdings|zapfdingbats|dingbats|marlett|\bsymbol(?:mt)?\b)",
    re.IGNORECASE,
)

# Pattern matching standard 6-uppercase-letter PDF subset prefix (e.g. "ABCDEF+Wingdings-Regular")
_SUBSET_PREFIX_PATTERN = re.compile(r"^[A-Z]{6}\+")


def strip_subset_prefix(font_name: str) -> str:
    """
    Remove random 6-character uppercase PDF subset prefix if present.

    Example:
        'ABCDEF+Wingdings-Regular' -> 'Wingdings-Regular'
        'Wingdings-Regular' -> 'Wingdings-Regular'
    """
    return _SUBSET_PREFIX_PATTERN.sub("", font_name)


def is_symbol_font(font_name: str) -> bool:
    """
    Check if font name (with or without subset prefix) matches known symbol font families.

    Returns True for any symbol font (including unmapped variants like Wingdings4 or Dingbats)
    to ensure non-standard character codes are intercepted and never leaked raw into text.
    """
    clean_font = strip_subset_prefix(font_name)
    return bool(_BROAD_SYMBOL_FONT_PATTERN.search(clean_font))


def get_symbol_font_category(font_name: str) -> str:
    """
    Classify font into a strictly defined canonical symbol category.

    Matches only validated font families (wingdings, wingdings2, wingdings3, webdings, symbol).
    Unrecognized or unmapped families (e.g. 'Wingdings4', 'WingdingsPro', 'CustomSymbol')
    strictly return 'unknown', directing their glyphs to the unmapped placeholder path.
    """
    clean_font = strip_subset_prefix(font_name)
    for category, pattern in _SYMBOL_FAMILY_PATTERNS.items():
        if pattern.match(clean_font):
            return category
    return "unknown"


def format_unmapped_placeholder(font_name: str, codepoint: int) -> str:
    """
    Format a space-padded placeholder token for unmapped symbol font codepoints.

    Ensures that unmapped glyphs do not merge adjacent words while remaining clearly
    identifiable for downstream auditing.
    """
    clean_font = strip_subset_prefix(font_name)
    return f" [UNMAPPED:{clean_font}:0x{codepoint:04X}] "


@lru_cache(maxsize=1)
def load_symbol_mappings(
    config_path: str | Path | None = None,
) -> tuple[dict[tuple[str, int], str], dict[int, str]]:
    """
    Load and cache symbol font and smart punctuation translation tables from YAML.

    Args:
        config_path: Optional path to YAML config file. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        tuple containing:
        - symbol_codepoint_map: dict[(font_category, codepoint_int), replacement_string]
        - text_punctuation_map: dict[codepoint_int, replacement_string]
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_file():
        logger.warning(
            "Symbol font mapping config not found at '%s'. Using fallback empty maps.", path
        )
        return {}, {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Failed to load symbol font mappings from '%s': %s", path, exc)
        return {}, {}

    symbol_map: dict[tuple[str, int], str] = {}
    symbol_fonts_section = data.get("symbol_fonts", {})

    for font_cat, codepoints in symbol_fonts_section.items():
        cat_lower = font_cat.lower()
        for hex_key, entry in codepoints.items():
            try:
                cp_int = int(hex_key, 16) if isinstance(hex_key, str) else int(hex_key)
                glyph = entry.get("glyph", "")
                symbol_map[(cat_lower, cp_int)] = glyph
            except (ValueError, TypeError) as err:
                logger.warning(
                    "Invalid codepoint key '%s' in category '%s': %s", hex_key, font_cat, err
                )

    punctuation_map: dict[int, str] = {}
    punct_section = data.get("text_punctuation", {})

    for hex_key, entry in punct_section.items():
        try:
            cp_int = int(hex_key, 16) if isinstance(hex_key, str) else int(hex_key)
            glyph = entry.get("glyph", "")
            punctuation_map[cp_int] = glyph
        except (ValueError, TypeError) as err:
            logger.warning("Invalid text punctuation key '%s': %s", hex_key, err)

    return symbol_map, punctuation_map


def clear_symbol_mappings_cache() -> None:
    """Clear the cached mappings loader (used primarily in test suites)."""
    load_symbol_mappings.cache_clear()
