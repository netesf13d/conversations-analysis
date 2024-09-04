# -*- coding: utf-8 -*-
"""
Utilities
"""

import re
import unicodedata
from pathlib import Path
from string import punctuation


# =============================================================================
# Types
# =============================================================================

PathObj = Path | str

# =============================================================================
# Constants
# =============================================================================

CRLF_REGEX = re.compile("[\\\n\\\r]")
PUNCTUATION_REGEX = re.compile(f"[{re.escape(punctuation)}]")


# =============================================================================
# Functions
# =============================================================================

def strip_diacritics(txt: str)-> str:
    """
    Remove all diacritics from words and characters forbidden in filenames.
    """
    norm_txt = unicodedata.normalize('NFD', txt)
    shaved = ''.join(c for c in norm_txt
                     if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)
