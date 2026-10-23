"""Every call to Gemma 4 goes through this file.

Gemma 4 is reached through the Gemini API (the key comes from Google AI Studio).
Features used here, all on the same `generate_content` endpoint:
  * system instructions   -> the "coach, never write the patch" rules
  * function calling      -> Gemma decides which repo files to read
  * thinking mode         -> thinking_level "high" (or "minimal" with --fast)
  * image understanding   -> optional screenshot of an error or failing test
  * long context          -> README + CONTRIBUTING + issue thread + file tree + code
"""
from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

DEFAULT_MODEL = "gemma-4-26b-a4b-it"  # fast Mixture-of-Experts model
MODEL_ALIASES = {
    "26b": "gemma-4-26b-a4b-it",
    "moe": "gemma-4-26b-a4b-it",
    "31b": "gemma-4-31b-it",
    "dense": "gemma-4-31b-it",
}
PLACEHOLDER_KEYS = {"", "paste_your_key_here", "your-key-here", "your_api_key_here"}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


class GemmaError(Exception):
    """Friendly, user-facing error."""


def resolve_model(name: str | None) -> str:
    name = (name or os.getenv("GEMMA_MODEL") or DEFAULT_MODEL).strip()
    return MODEL_ALIASES.get(name.lower(), name)


def get_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if key.lower() in PLACEHOLDER_KEYS:
        raise GemmaError(
            "No API key found.\n"
            "  1. Create a key at https://aistudio.google.com/apikey\n"
            "  2. Copy .env.example to a new file named .env\n"
            "  3. Replace PASTE_YOUR_KEY_HERE with your key and save."
        )
    return key


class GemmaClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        thinking: str | None = "high",  # "high", "minimal" or None
        max_tool_steps: int = 6,
        client=None,  # injectable for tests
        sleep=time.sleep,
    ):
        self.model = resolve_model(model)
        self.thinking = thinking
        self.max_tool_steps = max_tool_steps
        self.client = client or genai.Client(api_key=api_key or get_api_key())
        self.calls = 0  # successful Gemma responses (shown at the end of each run)
        self.notes: list[str] = []
        self._sleep = sleep

    # ----------------------------------------------------------------- config
    def _config(self, system: str | None, tools: list[dict] | None, thinking: str | None):
        kwargs: dict = {}
        if system:
            kwargs["system_instruction"] = system
        if tools:
            kwargs["tools"] = [types.Tool(function_declarations=tools)]
            kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        level = thinking if thinking is not None else self.thinking
        if level:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            except Exception:  # very old SDK
                self.notes.append("thinking_level not supported by this SDK version; skipped.")
        return types.GenerateContentConfig(**kwargs)

    # ------------------------------------------------------------ raw request
    def _generate(self, contents, config):
        delays = [6, 20, 45]
        attempt = 0
        while True:
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                self.calls += 1
                return resp
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                msg = str(e)
                if code == 400 and config.thinking_config is not None and "think" in msg.lower():
                    config.thinking_config = None
                    self.notes.append("Thinking mode was rejected by the API; continued without it.")
                    continue
                if code in (429, 500, 503) and attempt < len(delays):
                    self._sleep(delays[attempt])
                    attempt += 1
                    continue
                raise GemmaError(self._friendly(code, msg)) from e

    def _friendly(self, code, msg: str) -> str:
        low = msg.lower()
        if code in (401, 403) or "api key" in low:
            return "Google rejected your API key. Check GEMINI_API_KEY in your .env file."
        if code == 404:
            return (
                f"Model '{self.model}' is not available to your key. Try --model 31b, "
                "or check the model list at https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api"
            )
        if code == 429:
            return (
                "Free-tier rate limit reached. Wait a minute and run again "
                "(or use --fast to spend fewer tokens)."
            )
        return f"Gemma API error {code}: {msg[:300]}"

    # ------------------------------------------------------------------ image
    def image_part(self, path: str) -> types.Part:
        p = Path(path)
        if not p.is_file():
            raise GemmaError(f"Image not found: {path}")
        mime = mimetypes.guess_type(p.name)[0] or ""
        if mime not in IMAGE_TYPES:
            raise GemmaError("Use a .png, .jpg, .webp or .gif screenshot.")
        try:  # documented route: upload, then reference
            f = self.client.files.upload(file=str(p))
            return types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type or mime)
        except Exception:  # fall back to sending the bytes inline
            return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)

    def _user_content(self, prompt: str, image_path: str | None):
        parts = []
        if image_path:
            parts.append(self.image_part(image_path))
        parts.append(types.Part.from_text(text=prompt))
        return types.Content(role="user", parts=parts)

    # ------------------------------------------------------------- simple ask
    def ask(self, prompt: str, system: str | None = None, image_path: str | None = None,
            thinking: str | None = None) -> str:
        config = self._config(system, None, thinking)
        resp = self._generate([self._user_content(prompt, image_path)], config)
        return self._text(resp)

    # ---------------------------------------------- function-calling agent loop
    def run_with_tools(self, prompt: str, system: str, declarations: list[dict],
                       functions: dict, image_path: str | None = None, on_tool=None) -> str:
        """Let Gemma call our functions (read repo files) until it writes the answer."""
        contents = [self._user_content(prompt, image_path)]
        config = self._config(system, declarations, None)
        for _ in range(self.max_tool_steps):
            resp = self._generate(contents, config)
            calls = resp.function_calls or []
            if not calls:
                return self._text(resp)
            contents.append(resp.candidates[0].content)  # keep Gemma's turn (and its signatures)
            parts = []
            for fc in calls:
                args = dict(fc.args or {})
                if on_tool:
                    on_tool(fc.name, args)
                fn = functions.get(fc.name)
                try:
                    result = fn(**args) if fn else f"Unknown tool: {fc.name}"
                except Exception as e:  # tool errors go back to the model, not the user
                    result = f"Tool error: {e}"
                parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=parts))
        # Tool budget used up: ask for the final answer with no tools.
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text="Tool budget used. Write the final coaching plan now from what you have seen."
                )],
            )
        )
        resp = self._generate(contents, self._config(system, None, None))
        return self._text(resp)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _text(resp) -> str:
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            raise GemmaError("Gemma returned an empty answer. Try again, or try --model 31b.")
        return text

    def check(self) -> str:
        """Tiny round trip used by `python main.py check`."""
        return self.ask("Reply with exactly: ready", thinking="minimal")
