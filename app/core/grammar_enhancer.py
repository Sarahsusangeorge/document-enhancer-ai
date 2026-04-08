"""Grammar enhancement module using LanguageTool with custom rules.

Provides automated grammar and spelling correction with before/after
tracking for every correction applied.  Includes OCR text pre-cleaning
and regex-based custom rules for patterns that LanguageTool frequently
misses (possessive apostrophes, subject-verb agreement, etc.).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """A single correction detected by LanguageTool or a custom rule."""
    original: str
    corrected: str
    rule_id: str
    message: str
    offset: int
    length: int
    category: str = ""
    context: str = ""


@dataclass
class GrammarResult:
    original_text: str
    corrected_text: str
    corrections: List[Correction] = field(default_factory=list)
    total_errors: int = 0
    categories: Dict[str, int] = field(default_factory=dict)

    @property
    def correction_count(self) -> int:
        return len(self.corrections)


def _clean_ocr_text(text: str) -> str:
    """Normalize common OCR artefacts before grammar checking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z])", ". ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


# ── Custom regex rules ────────────────────────────────────────────────
# Each rule: (compiled_pattern, replacement, rule_id, message, category)
# Rules are applied independently; each scans the latest version of the
# text so earlier substitutions are visible to later rules.

_CUSTOM_RULE_DEFS = [
    (
        r"\b(students|universities|teachers|schools|systems|people|children"
        r"|institutions|members|companies|users)\s+has\b",
        r"\1 have",
        "CUSTOM_SVA_PLURAL_HAS",
        "Subject-verb agreement: plural subject should use 'have' instead of 'has'.",
        "Grammar / Subject-Verb Agreement",
        re.IGNORECASE,
    ),
    (
        r"\b(Students|Teachers|Children|Users|People|Workers|Customers|Members)"
        r"\s+(personal|private|individual|own|academic|daily|social)\b",
        r"\1' \2",
        "CUSTOM_POSSESSIVE_APOSTROPHE",
        "Missing possessive apostrophe.",
        "Punctuation / Possessives",
        0,
    ),
    (
        r"(?<=\S)  +(?=\S)",
        " ",
        "CUSTOM_DOUBLE_SPACE",
        "Multiple consecutive spaces should be a single space.",
        "Typographical",
        0,
    ),
    (
        r"(?<=[a-z])\.(?=[A-Z])",
        ". ",
        "CUSTOM_MISSING_SPACE_PERIOD",
        "Missing space after period.",
        "Typographical",
        0,
    ),
]

_COMPILED_RULES: list = []


def _get_compiled_rules():
    if _COMPILED_RULES:
        return _COMPILED_RULES
    for pat_str, repl, rid, msg, cat, flags in _CUSTOM_RULE_DEFS:
        _COMPILED_RULES.append((re.compile(pat_str, flags), repl, rid, msg, cat))
    return _COMPILED_RULES


def _java_available() -> bool:
    """Return True if a ``java`` executable is on PATH."""
    import shutil
    return shutil.which("java") is not None


class GrammarEnhancer:
    """Enhances text grammar and spelling using LanguageTool + custom rules."""

    def __init__(self, language: str = "en-US"):
        self.language = language
        self._tool = None
        self._tool_failed = False

    def _get_tool(self):
        """Lazily start LanguageTool.  Returns None if unavailable."""
        if self._tool is not None:
            return self._tool
        if self._tool_failed:
            return None

        if not _java_available():
            logger.warning(
                "Java not found on PATH – LanguageTool unavailable, "
                "falling back to custom rules only."
            )
            self._tool_failed = True
            return None

        try:
            import language_tool_python
            self._tool = language_tool_python.LanguageTool(self.language)
            return self._tool
        except Exception as e:
            logger.warning("LanguageTool init failed: %s", e)
            self._tool_failed = True
            return None

    def _run_custom_rules(self, text: str) -> Tuple[str, List[Correction]]:
        """Apply each regex rule independently on the current text."""
        rules = _get_compiled_rules()
        all_corrections: List[Correction] = []
        current = text

        for pattern, repl, rule_id, message, category in rules:
            matches = list(pattern.finditer(current))
            if not matches:
                continue

            rule_corrections: List[Correction] = []
            for m in matches:
                orig = m.group()
                fixed = m.expand(repl)
                if orig == fixed:
                    continue

                ctx_start = max(0, m.start() - 30)
                ctx_end = min(len(current), m.end() + 30)
                rule_corrections.append(Correction(
                    original=orig,
                    corrected=fixed,
                    rule_id=rule_id,
                    message=message,
                    offset=m.start(),
                    length=len(orig),
                    category=category,
                    context=current[ctx_start:ctx_end],
                ))

            if rule_corrections:
                current = pattern.sub(repl, current)
                all_corrections.extend(rule_corrections)

        return current, all_corrections

    def _singular_verb_fix(self, text: str) -> Tuple[str, List[Correction]]:
        """Fix 'a student struggle' -> 'a student struggles'."""
        pattern = re.compile(
            r"\b(a\s+(?:student|teacher|child|person|user|member|system"
            r"|institution))\s+"
            r"(struggle|learn|need|want|work|try|fail|start|begin|create"
            r"|handle|manage|provide|require|use)\b",
            re.IGNORECASE,
        )
        corrections: List[Correction] = []
        parts: list = []
        last_end = 0

        for m in pattern.finditer(text):
            verb = m.group(2)
            if verb.endswith("s"):
                continue
            corrected_verb = verb + "s"
            full_original = m.group()
            full_corrected = m.group(1) + " " + corrected_verb

            ctx_start = max(0, m.start() - 30)
            ctx_end = min(len(text), m.end() + 30)
            corrections.append(Correction(
                original=full_original,
                corrected=full_corrected,
                rule_id="CUSTOM_SVA_SINGULAR_A",
                message="Subject-verb agreement: singular subject needs "
                        "verb ending in '-s'.",
                offset=m.start(),
                length=len(full_original),
                category="Grammar / Subject-Verb Agreement",
                context=text[ctx_start:ctx_end],
            ))
            parts.append(text[last_end:m.start()])
            parts.append(full_corrected)
            last_end = m.end()

        if not corrections:
            return text, []

        parts.append(text[last_end:])
        return "".join(parts), corrections

    def check(self, text: str) -> GrammarResult:
        """Run LanguageTool on the text and return results."""
        tool = self._get_tool()
        if tool is None:
            return GrammarResult(
                original_text=text, corrected_text=text,
            )
        matches = tool.check(text)

        corrections: List[Correction] = []
        for match in matches:
            replacement = match.replacements[0] if match.replacements else ""
            err_len = getattr(match, "error_length", None) or getattr(match, "errorLength", 0)
            rule_id = getattr(match, "rule_id", None) or getattr(match, "ruleId", "")
            original_fragment = text[match.offset:match.offset + err_len]

            context_start = max(0, match.offset - 30)
            context_end = min(len(text), match.offset + err_len + 30)

            corrections.append(Correction(
                original=original_fragment,
                corrected=replacement,
                rule_id=rule_id,
                message=match.message,
                offset=match.offset,
                length=err_len,
                category=getattr(match, "category", ""),
                context=text[context_start:context_end],
            ))

        corrected_text = self._apply_corrections(text, corrections)

        return GrammarResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=corrections,
            total_errors=len(matches),
        )

    def _apply_corrections(self, text: str,
                           corrections: List[Correction]) -> str:
        sorted_corrections = sorted(
            corrections, key=lambda c: c.offset, reverse=True,
        )
        result = text
        for corr in sorted_corrections:
            if corr.corrected:
                result = (
                    result[:corr.offset]
                    + corr.corrected
                    + result[corr.offset + corr.length:]
                )
        return result

    def _categorize(self, corrections: List[Correction]) -> Dict[str, int]:
        cats: Dict[str, int] = {}
        for c in corrections:
            key = c.category or "Other"
            cats[key] = cats.get(key, 0) + 1
        return cats

    def enhance(self, text: str, max_passes: int = 3) -> GrammarResult:
        """Run grammar checking with multiple passes for iterative improvement.

        Pipeline: OCR clean-up -> LanguageTool (multi-pass) -> custom rules.
        """
        cleaned = _clean_ocr_text(text)
        all_corrections: List[Correction] = []
        current_text = cleaned

        try:
            for pass_num in range(max_passes):
                result = self.check(current_text)
                if not result.corrections:
                    break
                all_corrections.extend(result.corrections)
                current_text = result.corrected_text
                logger.info(
                    "Grammar pass %d: %d corrections",
                    pass_num + 1, len(result.corrections),
                )
        except Exception as e:
            logger.warning("LanguageTool check failed, using custom rules only: %s", e)

        current_text, custom_corrections = self._run_custom_rules(current_text)
        current_text, sva_corrections = self._singular_verb_fix(current_text)

        already_fixed = {(c.offset, c.original) for c in all_corrections}
        for c in custom_corrections + sva_corrections:
            if (c.offset, c.original) not in already_fixed:
                all_corrections.append(c)

        return GrammarResult(
            original_text=text,
            corrected_text=current_text,
            corrections=all_corrections,
            total_errors=len(all_corrections),
            categories=self._categorize(all_corrections),
        )

    def close(self) -> None:
        if self._tool is not None:
            self._tool.close()
            self._tool = None
