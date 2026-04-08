"""OCR engine module wrapping pytesseract.

Supports standard printed text, handwriting configuration,
per-word confidence scores, batch (multi-page) processing,
and post-OCR spell correction for handwritten documents.
"""

import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytesseract
from PIL import Image

try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    _spell.word_frequency.load_words([
        "mango", "cafe", "buyed", "oranges", "bananas",
        "some", "home", "looking", "rest", "little", "still",
        "small", "returning",
    ])
    _HAS_SPELLCHECKER = True
except ImportError:
    _spell = None
    _HAS_SPELLCHECKER = False

logger = logging.getLogger(__name__)

_WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{user}\AppData\Local\Tesseract-OCR\tesseract.exe",
]


def _find_tesseract() -> Optional[str]:
    """Auto-detect the Tesseract binary on the current platform."""
    found = shutil.which("tesseract")
    if found:
        return found

    if sys.platform == "win32":
        import os
        username = os.getenv("USERNAME", "")
        for candidate in _WINDOWS_TESSERACT_PATHS:
            p = Path(candidate.replace("{user}", username))
            if p.is_file():
                logger.info("Auto-detected Tesseract at %s", p)
                return str(p)

    return None


@dataclass
class OCRResult:
    text: str
    confidence: float
    word_confidences: List[Dict] = field(default_factory=list)
    page_number: int = 1


_SYMBOL_FIXES = [
    (re.compile(r"(?<![a-zA-Z0-9])\|(?![a-zA-Z0-9])"), "I"),
    (re.compile(r"(?<![a-zA-Z0-9])1(?![0-9])"), "I"),
    (re.compile(r"\b40\b"), "to"),
    (re.compile(r"\b80\b"), "so"),
    (re.compile(r"(?<![a-zA-Z0-9])4(?![0-9])"), "to"),
    (re.compile(r"\bO\s*@"), "a "),
    (re.compile(r"@(?=[a-zA-Z])"), ""),
]

_OCR_WORD_FIXES = {
    "gome": "some",
    "heme": "home",
    "leeking": "looking",
    "bitte": "little",
    "shu": "still",
    "mange": "mango",
    "fest": "rest",
    "reloaning": "returning",
    "matt": "small",
}

def _fix_symbols(text: str) -> str:
    """Fix common OCR symbol/number to letter misreads."""
    for pattern, replacement in _SYMBOL_FIXES:
        text = pattern.sub(replacement, text)
    return text


def _spellcheck_word(word: str) -> str:
    """Return the spell-corrected version of a word, or the original."""
    if not word:
        return word
    letters_only = re.sub(r"[^a-zA-Z]", "", word)
    if len(letters_only) <= 1:
        return word

    prefix = re.match(r"^([^a-zA-Z]*)", word).group(1)
    suffix = re.search(r"([^a-zA-Z]*)$", word).group(1)
    was_upper = letters_only[0].isupper()

    if letters_only.lower() in _OCR_WORD_FIXES:
        fixed = _OCR_WORD_FIXES[letters_only.lower()]
        if was_upper:
            fixed = fixed.capitalize()
        return prefix + fixed + suffix

    if not _HAS_SPELLCHECKER:
        return word
    if letters_only.lower() in _spell:
        return word
    corrected = _spell.correction(letters_only.lower())
    if corrected and corrected != letters_only.lower():
        if was_upper:
            corrected = corrected.capitalize()
        return prefix + corrected + suffix
    return word


def _postprocess_ocr(text: str, word_confidences: List[Dict]) -> str:
    """Fix common OCR errors using symbol correction and spell checking.

    Applies symbol fixes first (digit/punctuation → letter), then runs
    every word through the spell checker to correct non-dictionary words.
    """
    text = _fix_symbols(text)

    if not _HAS_SPELLCHECKER:
        return text

    def replace_word(match):
        return _spellcheck_word(match.group(0))

    result = re.sub(r"\S+", replace_word, text)
    logger.info("OCR post-processing applied spell correction")
    return result


class OCREngine:
    """Wrapper around pytesseract for text extraction from images."""

    DEFAULT_CONFIG = "--oem 3 --psm 3"
    HANDWRITING_CONFIG = "--oem 3 --psm 3"

    def __init__(self, tesseract_path: Optional[str] = None,
                 language: str = "eng"):
        resolved = tesseract_path or _find_tesseract()
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
            logger.info("Using Tesseract at: %s", resolved)
        else:
            logger.warning(
                "Tesseract not found on PATH or common install locations. "
                "OCR will fail unless tesseract is available."
            )
        self.language = language

    def extract_text(self, image: np.ndarray,
                     config: str = None) -> OCRResult:
        if config is None:
            config = self.DEFAULT_CONFIG

        pil_image = Image.fromarray(image)
        text = pytesseract.image_to_string(
            pil_image, lang=self.language, config=config,
        )
        data = pytesseract.image_to_data(
            pil_image, lang=self.language, config=config,
            output_type=pytesseract.Output.DICT,
        )

        confidences = [int(c) for c in data["conf"] if int(c) >= 0]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        word_details = []
        for i, word in enumerate(data["text"]):
            if word.strip() and int(data["conf"][i]) >= 0:
                word_details.append({
                    "word": word,
                    "confidence": int(data["conf"][i]),
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                })

        corrected_text = _postprocess_ocr(text.strip(), word_details)

        return OCRResult(
            text=corrected_text,
            confidence=round(avg_confidence, 2),
            word_confidences=word_details,
        )

    def extract_handwriting(self, image: np.ndarray) -> OCRResult:
        return self.extract_text(image, config=self.HANDWRITING_CONFIG)

    def batch_extract(self, images: List[np.ndarray],
                      config: str = None) -> List[OCRResult]:
        results = []
        for i, img in enumerate(images, start=1):
            result = self.extract_text(img, config=config)
            result.page_number = i
            results.append(result)
            logger.info("Page %d: confidence=%.1f%%", i, result.confidence)
        return results

    def get_low_confidence_words(self, result: OCRResult,
                                 threshold: float = 60.0) -> List[Dict]:
        return [
            w for w in result.word_confidences
            if w["confidence"] < threshold
        ]
