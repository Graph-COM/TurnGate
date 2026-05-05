"""
Gemini-based JSON format corrector.

Uses Gemini API (via BlackBoxModel) to fix malformed JSON output
from weaker LLMs (e.g., 8B models). No GPU needed — API only.
"""

import logging


class GeminiJSONCorrector:
    """Thin wrapper around BlackBoxModel that provides correct_json_format() via API (no GPU)."""

    def __init__(self, model):
        self._model = model
        self.logger = logging.getLogger(self.__class__.__name__)

    def correct_json_format(
        self, raw_output: str, expected_schema_description: str
    ) -> str:
        prompt = (
            "You are a JSON format corrector. Your ONLY job is to fix JSON formatting issues "
            "in the given text. You must STRICTLY preserve all original content, meaning, and wording. "
            "Do NOT add, remove, rephrase, or enhance any information. Do NOT use your own knowledge. "
            "Only fix: missing/extra commas, unescaped characters, broken quotes, missing braces/brackets, "
            "and other syntax issues. Output ONLY the corrected JSON, nothing else.\n\n"
            f"Expected JSON schema:\n{expected_schema_description}\n\n"
            f"Raw output to fix:\n{raw_output}\n\n"
            "Output the corrected JSON only, no explanation:"
        )
        try:
            return self._model.generate(prompt)
        except Exception as e:
            self.logger.warning(f"[GeminiJSONCorrector] API call failed: {e}")
            return raw_output


def create_gemini_corrector(config: dict) -> "GeminiJSONCorrector":
    """
    Create a GeminiJSONCorrector from a config dict.

    Expected config keys:
        provider: "gemini"
        name: "gemini-2.5-flash"
        api_key: str
        max_tokens: int (default 512)
        temperature: float (default 0.0)
    """
    from model.model_loader import BlackBoxModel

    corrector_model = BlackBoxModel(config)
    return GeminiJSONCorrector(corrector_model)
