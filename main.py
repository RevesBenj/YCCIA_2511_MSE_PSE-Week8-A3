"""
Benjelyn Reves Patiag
Week 8 - Activity 3
AI CV ATS Analyzer (single-file version)

What this script do (simple):
- Read CV + Job Description from PDF / DOCX / TXT
- Compute simple ATS keyword match score
- Ask LLM (Gemini or OpenAI) for improvement suggestions (optional)

Notes:
- API keys are NOT hardcoded (use environment variables)
- Optional libs (pdfplumber / python-docx / google-genai / openai) are imported safely
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional, Protocol


# =====================================================
# 0) CONFIGURATION
# =====================================================

LLM_PROVIDER: Final[str] = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# Gemini
GEMINI_API_KEY: Final[str] = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: Final[str] = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

# OpenAI
OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL: Final[str] = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

SUPPORTED_EXTS: Final[set[str]] = {".pdf", ".docx", ".txt"}

STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "and",
        "or",
        "the",
        "a",
        "an",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "as",
        "at",
        "by",
        "is",
        "are",
        "be",
        "this",
        "that",
    }
)

TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z\+\#\.\-]{1,}")


# =====================================================
# 1) TEXT EXTRACTION (OOP)
# =====================================================


class TextExtractor:
    """Base extractor interface."""

    def extract(self, file_path: Path) -> str:
        """Read text from the given file path."""
        raise NotImplementedError


class PDFTextExtractor(TextExtractor):
    """Extracts text from PDF files."""

    def extract(self, file_path: Path) -> str:
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Missing dependency: pdfplumber. Install with: pip install pdfplumber"
            ) from exc

        text_parts: list[str] = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                text_parts.append((page.extract_text() or "").strip())
        return "\n".join(p for p in text_parts if p).strip()


class DOCXTextExtractor(TextExtractor):
    """Extracts text from DOCX files."""

    def extract(self, file_path: Path) -> str:
        try:
            import docx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Missing dependency: python-docx. Install with: pip install python-docx"
            ) from exc

        document = docx.Document(str(file_path))
        return "\n".join(p.text for p in document.paragraphs).strip()


class TXTTextExtractor(TextExtractor):
    """Extracts text from TXT files."""

    def extract(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()


class ExtractorFactory:
    """Factory to choose the correct extractor."""

    @staticmethod
    def get(file_path: Path) -> TextExtractor:
        """Return correct extractor based on file extension."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return PDFTextExtractor()
        if ext == ".docx":
            return DOCXTextExtractor()
        if ext == ".txt":
            return TXTTextExtractor()
        raise ValueError(f"Unsupported file format: {ext}. Use PDF, DOCX, or TXT.")


# =====================================================
# 2) ATS SCORING (OOP)
# =====================================================


def tokenize(text: str) -> List[str]:
    """Convert text into keyword tokens (simple ATS-style)."""
    words = TOKEN_PATTERN.findall(text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 2]


@dataclass(frozen=True, slots=True)
class ATSResult:
    """Result of ATS scoring."""

    score_percent: int
    matched_keywords: List[str]
    missing_keywords: List[str]


class ATSScorer:
    """Simple ATS scoring using keyword overlap."""

    def score(self, cv_text: str, jd_text: str, top_n: int = 30) -> ATSResult:
        """Score a CV vs Job Description using keyword overlap."""
        if top_n <= 0:
            raise ValueError("top_n must be > 0")

        cv_tokens = set(tokenize(cv_text))
        jd_tokens = tokenize(jd_text)

        if not jd_tokens:
            return ATSResult(score_percent=0, matched_keywords=[], missing_keywords=[])

        freq = Counter(jd_tokens)
        top_keywords = [w for w, _ in freq.most_common(top_n)]

        matched = [k for k in top_keywords if k in cv_tokens]
        missing = [k for k in top_keywords if k not in cv_tokens]

        score = int((len(matched) / len(top_keywords)) * 100) if top_keywords else 0
        return ATSResult(score_percent=score, matched_keywords=matched, missing_keywords=missing)


# =====================================================
# 3) LLM CLIENTS (OOP)
# =====================================================


class LLMClient(Protocol):
    """LLM client protocol."""

    def generate(self, prompt: str) -> str:
        """Generate text response from prompt."""
        raise NotImplementedError


class GeminiClient:
    """Gemini LLM client wrapper (google-genai)."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in your environment.")
        self._model = model

        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Missing dependency: google-genai. Install with: pip install google-genai"
            ) from exc

        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return (getattr(response, "text", "") or "").strip()


class OpenAIClient:
    """OpenAI LLM client wrapper (openai python sdk)."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Set it in your environment.")
        self._model = model

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Missing dependency: openai. Install with: pip install openai") from exc

        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are an expert ATS resume reviewer."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class LLMFactory:
    """Factory to create chosen LLM client."""

    @staticmethod
    def create() -> Optional[LLMClient]:
        """Create LLM client. Return None if API key is missing."""
        if LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                return None
            return OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

        if not GEMINI_API_KEY:
            return None
        return GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)


# =====================================================
# 4) MAIN APPLICATION (OOP)
# =====================================================


class CVAnalyzerApp:
    """Main CLI app to analyze CV vs Job Description."""

    def __init__(self) -> None:
        self._scorer = ATSScorer()
        self._llm = LLMFactory.create()

    @staticmethod
    def _read_text(path_str: str) -> str:
        """Read text from a file path string."""
        path = Path(path_str.strip().strip('"')).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file type: {path.suffix}. Use PDF/DOCX/TXT.")

        extractor = ExtractorFactory.get(path)
        return extractor.extract(path)

    @staticmethod
    def _build_prompt(cv_text: str, jd_text: str, ats: ATSResult) -> str:
        """Build prompt for LLM."""
        missing_preview = ", ".join(ats.missing_keywords[:15])
        return (
            "You are an ATS expert and professional recruiter.\n\n"
            "Analyze the CV against the Job Description and provide improvements.\n\n"
            "Format your answer as:\n"
            "1) ATS Summary (2–3 lines)\n"
            "2) Missing Keywords (bullet list)\n"
            "3) Skills Section Improvements\n"
            "4) Experience Bullet Improvements (use STAR)\n"
            "5) Two rewritten achievement bullet examples\n\n"
            f"ATS Score: {ats.score_percent}%\n"
            f"Missing Keywords: {missing_preview}\n\n"
            "JOB DESCRIPTION:\n"
            f"{jd_text}\n\n"
            "CV:\n"
            f"{cv_text}\n"
        ).strip()

    def run(self) -> None:
        """Run CLI flow."""
        print("=== AI CV Analyzer (CV vs Job Description) ===")

        cv_path = input("Enter CV file path (PDF/DOCX/TXT): ").strip()
        jd_path = input("Enter Job Description file path (PDF/DOCX/TXT): ").strip()

        cv_text = self._read_text(cv_path)
        jd_text = self._read_text(jd_path)

        ats = self._scorer.score(cv_text, jd_text)

        print("\n--- ATS Quick Result ---")
        print(f"ATS Match Score: {ats.score_percent}%")
        print("Matched keywords:", ", ".join(ats.matched_keywords[:10]) or "None")
        print("Missing keywords:", ", ".join(ats.missing_keywords[:10]) or "None")

        print("\n--- AI Recommendations ---")
        if self._llm is None:
            print("LLM is disabled (missing API key). Set GEMINI_API_KEY or OPENAI_API_KEY.")
            return

        prompt = self._build_prompt(cv_text, jd_text, ats)
        advice = self._llm.generate(prompt)
        print(advice)


# =====================================================
# 5) ENTRY POINT
# =====================================================


def main() -> int:
    """Program entry point. Return 0 if ok else 1."""
    try:
        CVAnalyzerApp().run()
        return 0
    except (FileNotFoundError, ValueError, ImportError, OSError) as exc:
        print("\nERROR:", exc)
        print("\nCheck:")
        print("- File paths are correct")
        print("- Required libraries installed")
        print("- API keys set in environment variables (if using LLM)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
