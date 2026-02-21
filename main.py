"""
Benjelyn Reves Patiag
Week 8 - Activity 3
AI CV ATS Analyzer (Single File Version)

Features:
- CV vs Job Description comparison
- ATS keyword matching + AI recommendations
- Hardcoded API key for demo
"""


import os
import re
from dataclasses import dataclass
from typing import List, Protocol


# =====================================================
# 0) CONFIGURATION (FOR CLASS DEMO ONLY)
# =====================================================

LLM_PROVIDER = "gemini"   # "gemini" or "openai"

# ---- Gemini ----
GEMINI_API_KEY = "your_gemini_api_key_here"  # Replace with your actual Gemini API key
GEMINI_MODEL = "gemini-2.0-flash"

# ---- OpenAI (optional) ----
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4.1-mini"


# =====================================================
# 1) TEXT EXTRACTION (OOP)
# =====================================================


class TextExtractor:
    """Base extractor class (interface style)."""

    def extract(self, file_path: str) -> str:
        """Extract text content from the given file path."""
        raise NotImplementedError


class PDFTextExtractor(TextExtractor):
    """Extract text from PDF files."""

    def extract(self, file_path: str) -> str:
        """Read PDF and return extracted text."""
        # disable=import-outside-toplevel
        import pdfplumber

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip()


class DOCXTextExtractor(TextExtractor):
    """Extract text from DOCX files."""

    def extract(self, file_path: str) -> str:
        """Read DOCX and return extracted text."""
        # disable=import-outside-toplevel
        import docx

        d = docx.Document(file_path)
        return "\n".join(p.text for p in d.paragraphs).strip()


class TXTTextExtractor(TextExtractor):
    """Extract text from TXT files."""

    def extract(self, file_path: str) -> str:
        """Read TXT and return extracted text."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()


class ExtractorFactory:
    """Factory for choosing the correct extractor."""

    @staticmethod
    def get(file_path: str) -> TextExtractor:
        """Return the right extractor based on file extension."""
        ext = os.path.splitext(file_path.lower())[1]
        if ext == ".pdf":
            return PDFTextExtractor()
        if ext == ".docx":
            return DOCXTextExtractor()
        if ext == ".txt":
            return TXTTextExtractor()
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")


# =====================================================
# 2) ATS SCORING (OOP)
# =====================================================

STOPWORDS = {
    "and", "or", "the", "a", "an", "to", "for", "of", "in", "on",
    "with", "as", "at", "by", "is", "are", "be", "this", "that"
}


def tokenize(text: str) -> List[str]:
    """Tokenize text for ATS keyword matching."""
    words = re.findall(r"[A-Za-z][A-Za-z\+\#\.\-]{1,}", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 2]


@dataclass
class ATSResult:
    """ATS scoring result container."""

    score_percent: int
    matched_keywords: List[str]
    missing_keywords: List[str]


class ATSScorer:
    """
    Simple ATS scoring:
    - Extract top keywords from Job Description
    - Measure overlap with CV
    """

    def score(self, cv_text: str, jd_text: str, top_n: int = 30) -> ATSResult:
        """Compute ATS keyword overlap score and return ATSResult."""
        cv_tokens = set(tokenize(cv_text))
        jd_tokens = tokenize(jd_text)

        freq: dict[str, int] = {}
        for w in jd_tokens:
            freq[w] = freq.get(w, 0) + 1

        top_keywords = sorted(freq, key=freq.get, reverse=True)[:top_n]
        matched = [k for k in top_keywords if k in cv_tokens]
        missing = [k for k in top_keywords if k not in cv_tokens]

        score = 0 if not top_keywords else int((len(matched) / len(top_keywords)) * 100)

        return ATSResult(score, matched, missing)


# =====================================================
# 3) LLM CLIENTS (OOP)
# =====================================================


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate(self, prompt: str) -> str:
        """Generate response text based on prompt."""
        ...


class GeminiClient:
    """Gemini client implementation."""

    def __init__(self) -> None:
        # disable=import-outside-toplevel
        from google import genai

        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key not set in main.py")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL

    def generate(self, prompt: str) -> str:
        """Call Gemini model and return text response."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return (response.text or "").strip()


class OpenAIClient:
    """OpenAI client implementation."""

    def __init__(self) -> None:
        # disable=import-outside-toplevel
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not set in main.py")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL

    def generate(self, prompt: str) -> str:
        """Call OpenAI chat completion and return text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert ATS resume reviewer."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class LLMFactory:
    """Factory to build a chosen LLM client."""

    @staticmethod
    def create() -> LLMClient:
        """Create the LLM client based on LLM_PROVIDER."""
        if LLM_PROVIDER.lower() == "openai":
            return OpenAIClient()
        return GeminiClient()


# =====================================================
# 4) MAIN APPLICATION (OOP)
# =====================================================


class CVAnalyzerApp:
    """Main application class for CV vs JD analysis."""

    def __init__(self) -> None:
        self.scorer = ATSScorer()
        self.llm = LLMFactory.create()

    def read_text(self, path: str) -> str:
        """Read and extract text from file path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        extractor = ExtractorFactory.get(path)
        return extractor.extract(path)

    def build_prompt(self, cv_text: str, jd_text: str, ats: ATSResult) -> str:
        """Build LLM prompt using CV, JD, and ATS result."""
        return f"""
You are an ATS expert and professional recruiter.

Analyze the CV against the Job Description and provide improvements.

Format your answer as:
1) ATS Summary (2–3 lines)
2) Missing Keywords (bullet list)
3) Skills Section Improvements
4) Experience Bullet Improvements (use STAR)
5) Two rewritten achievement bullet examples

ATS Score: {ats.score_percent}%
Missing Keywords: {", ".join(ats.missing_keywords[:15])}

JOB DESCRIPTION:
{jd_text}

CV:
{cv_text}
""".strip()

    def run(self) -> None:
        """Run the CLI app."""
        print("=== AI CV Analyzer (CV vs Job Description) ===")

        cv_path = input("Enter CV file path (PDF/DOCX/TXT): ").strip().strip('"')
        jd_path = input("Enter Job Description file path (PDF/DOCX/TXT): ").strip().strip('"')

        cv_text = self.read_text(cv_path)
        jd_text = self.read_text(jd_path)

        ats = self.scorer.score(cv_text, jd_text)

        print("\n--- ATS Quick Result ---")
        print(f"ATS Match Score: {ats.score_percent}%")
        print("Matched keywords:", ", ".join(ats.matched_keywords[:10]) or "None")
        print("Missing keywords:", ", ".join(ats.missing_keywords[:10]) or "None")

        print("\n--- AI Recommendations ---")
        prompt = self.build_prompt(cv_text, jd_text, ats)
        advice = self.llm.generate(prompt)
        print(advice)


# =====================================================
# 5) ENTRY POINT
# =====================================================


def main() -> None:
    """Entry point wrapper."""
    try:
        CVAnalyzerApp().run()
    except Exception as e:  # disable=broad-except
        print("\nERROR:", e)
        print("\nCheck:")
        print("- File paths are correct")
        print("- Required libraries installed")
        print("- API key pasted correctly in main.py")


if __name__ == "__main__":
    main()
