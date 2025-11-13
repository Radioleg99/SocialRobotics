import asyncio
import io
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Ensure console prints UTF-8 (prefer reconfigure, fallback to TextIOWrapper)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
except Exception:
    pass

CONFIG_JSON_PATH = Path("config.json")
API_KEY_TXT_PATH = Path("api_key.txt")
SETTINGS_LOADED = False

# ==== Base configuration (overridable via config files) ====
OPENAI_SETTINGS = {
    # Leave blank; load_api_settings_from_files will populate these at runtime
    "api_key": "",
    "base_url": "https://api.openai.com",
    # Controller model: decides whether to display visible thinking
    "controller_model": "gpt-4.1-mini",
    "controller_temperature": 0.2,
    # Main reasoning model
    "reasoning_model": "gpt-4.1-mini",
    "reasoning_temperature": 0.4,
    # Visible thinking model (can later be swapped for a Furhat-specific endpoint)
    "thinking_model": "gpt-4.1-mini",
    "thinking_temperature": 0.2,
}

# Prefixes and gestures mapped to confidence tiers
CONFIDENCE_BEHAVIORS: Dict[str, Tuple[str, str]] = {
    "low": ("I'm not entirely sure, but ", "gentle head shake"),
    "medium": ("It seems that ", "steady gaze"),
    "high": ("I'm confident that ", "affirmative nod"),
}

MAX_THINKING_CUES = 4

# Controller prompt: must output JSON describing the plan
CONTROLLER_SYSTEM_PROMPT = (
    "You are the Furhat conversation controller. Output ONLY strict JSON with the keys:"
    ' {"need_thinking": true/false,'
    ' "confidence": "low/medium/high",'
    ' "thinking_notes": ["short thought 1","short thought 2"],'
    ' "reasoning_hint": "concise hint for the main model",'
    ' "answer": "final reply when no thinking is required" }.'
    "Return answer only when need_thinking=false; otherwise leave it empty."
    "Never add prose, comments, or Markdown."
)

# Main answer prompt: concise, friendly summary
REASONING_SYSTEM_PROMPT = (
    "You are a Furhat social robot. Reply in 2-3 friendly sentences,"
    " focusing on clear advice without exposing internal reasoning."
)

# Visible thinking prompt
THINKING_SYSTEM_PROMPT = (
    "You simulate Furhat's visible thinking. Emit 2-4 short English phrases,"
    " each under 12 words, describing ongoing cognition (e.g., 'Comparing options...')."
    "Sound natural, avoid final answers, and skip summaries."
)


def load_api_settings_from_files():
    """Inject API settings from config.json or api_key.txt."""
    global SETTINGS_LOADED
    if SETTINGS_LOADED:
        return

    api_key = ""
    config_data: Dict[str, Any] = {}

    if CONFIG_JSON_PATH.exists():
        try:
            with CONFIG_JSON_PATH.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
                if isinstance(loaded, dict):
                    config_data = loaded
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Failed to parse config.json: {err}") from err

    if config_data:
        for field in [
            "api_key",
            "base_url",
            "controller_model",
            "controller_temperature",
            "reasoning_model",
            "reasoning_temperature",
            "thinking_model",
            "thinking_temperature",
        ]:
            if field in config_data and config_data[field]:
                OPENAI_SETTINGS[field] = config_data[field]
        api_key = str(config_data.get("api_key", "")).strip()

    if not api_key and API_KEY_TXT_PATH.exists():
        with API_KEY_TXT_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    api_key = line
                    break

    if not api_key:
        raise RuntimeError(
            "API key not found. Add it to config.json (api_key field) or api_key.txt."
        )

    OPENAI_SETTINGS["api_key"] = api_key
    SETTINGS_LOADED = True


def cprint(text: str):
    """Print UTF-8 text safely; fallback to manual buffer writes if needed."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="ignore"))
            sys.stdout.flush()
        except Exception:
            pass


class ControllerModel:
    """Call the controller model to decide whether visible thinking is needed."""

    def __init__(self, question: str):
        load_api_settings_from_files()
        self.question = question
        self.api_key = OPENAI_SETTINGS["api_key"]
        if not self.api_key:
            raise RuntimeError("Please provide a valid API key in config.json or api_key.txt.")
        self.base_url = OPENAI_SETTINGS["base_url"].rstrip("/")
        self.model = OPENAI_SETTINGS.get("controller_model", OPENAI_SETTINGS["reasoning_model"])
        self.temperature = OPENAI_SETTINGS.get("controller_temperature", 0.2)

    def decide(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
                {"role": "user", "content": self.question},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return self._parse_json(content)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        candidate = text.strip()
        # Remove optional ```json fences
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Controller response is not valid JSON: {candidate}") from err


class ChatGPTSentenceStreamer:
    """Stream sentence-level chunks from the ChatGPT API (for thinking or answers)."""

    def __init__(
        self,
        user_content: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: str = "",
    ):
        load_api_settings_from_files()
        self.user_content = user_content
        self.model = model or OPENAI_SETTINGS["reasoning_model"]
        default_temp = OPENAI_SETTINGS["reasoning_temperature"]
        self.temperature = temperature if temperature is not None else default_temp
        self.system_prompt = system_prompt
        self.api_key = OPENAI_SETTINGS["api_key"]
        if not self.api_key:
            raise RuntimeError("Please provide a valid API key in config.json or api_key.txt.")
        self.base_url = OPENAI_SETTINGS["base_url"].rstrip("/")
        self.word_count = 0

    async def stream(self):
        """Asynchronously yield sentence-like chunks."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                for clause in self._generate_clauses():
                    loop.call_soon_threadsafe(queue.put_nowait, clause)
            except Exception as exc:  # pragma: no cover
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        thread.join()

    def _generate_clauses(self):
        buffer = ""
        for token in self._token_stream():
            buffer += token
            self.word_count = len(buffer.split())
            ready, buffer = self._pop_ready_clauses(buffer)
            for clause in ready:
                yield clause
        if buffer.strip():
            yield buffer.strip()

    def _token_stream(self):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": True,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_content},
            ],
        }

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=90) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line.split("data:", 1)[1].strip()
                else:
                    data = line.strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

    @staticmethod
    def _pop_ready_clauses(text: str) -> Tuple[List[str], str]:
        clauses: List[str] = []
        start = 0
        for idx, char in enumerate(text):
            if char in ".?!":
                clause = text[start : idx + 1].strip()
                if clause:
                    clauses.append(clause)
                start = idx + 1
        remainder = text[start:]
        return clauses, remainder


def build_thinking_prompt(question: str, notes: List[str]) -> str:
    filtered = [note for note in notes if note]
    joined = "\n".join(f"- {note}" for note in filtered) or "- Reviewing possible answers"
    return (
        f"User question: {question}\n"
        f"Preliminary thoughts:\n{joined}\n"
        "Generate visible-thinking phrases following the system instructions."
    )


def build_reasoning_prompt(question: str, hint: str) -> str:
    hint_part = f"\nReference hint: {hint}" if hint else ""
    return (
        f"User question: {question}"
        f"{hint_part}\n"
        "Summarize a solution in 2-3 sentences without sharing chain-of-thought."
    )


def normalize_thinking_notes(notes: Any) -> List[str]:
    if isinstance(notes, list):
        return [str(item) for item in notes if item]
    if isinstance(notes, str) and notes.strip():
        return [notes.strip()]
    return []


def resolve_confidence(hint: Optional[str], word_count: int) -> str:
    if hint:
        key = hint.strip().lower()
        if key in CONFIDENCE_BEHAVIORS:
            return key
    return estimate_confidence_from_words(word_count)


def estimate_confidence_from_words(word_count: int) -> str:
    """Fallback heuristic: longer answers imply higher confidence."""
    if word_count < 25:
        return "low"
    if word_count < 60:
        return "medium"
    return "high"


def is_meaningful_cue(text: str) -> bool:
    stripped = text.strip()
    stripped = stripped.strip(".!?…")  # remove trailing punctuation
    return bool(stripped)


class Orchestrator:
    """Orchestrates controller, thinking stream, and final answer delivery."""

    def __init__(self, question: str):
        self.question = question
        self.stop_thinking = asyncio.Event()
        self.controller = ControllerModel(question)
        self.decision: Dict[str, Any] = {}

    async def run(self):
        self.decision = self.controller.decide()
        need_thinking = bool(self.decision.get("need_thinking", False))
        confidence_hint = self.decision.get("confidence")
        cprint(f"Participant: {self.question}")

        if not need_thinking:
            await self._respond_directly(confidence_hint)
            return

        thinking_notes = normalize_thinking_notes(self.decision.get("thinking_notes"))
        reasoning_hint = self.decision.get("reasoning_hint", "")

        thinking_model = ChatGPTSentenceStreamer(
            user_content=build_thinking_prompt(self.question, thinking_notes),
            model=OPENAI_SETTINGS["thinking_model"],
            temperature=OPENAI_SETTINGS["thinking_temperature"],
            system_prompt=THINKING_SYSTEM_PROMPT,
        )
        reasoning_model = ChatGPTSentenceStreamer(
            user_content=build_reasoning_prompt(self.question, reasoning_hint),
            model=OPENAI_SETTINGS["reasoning_model"],
            temperature=OPENAI_SETTINGS["reasoning_temperature"],
            system_prompt=REASONING_SYSTEM_PROMPT,
        )

        thinking_task = asyncio.create_task(self._relay_thinking(thinking_model))
        try:
            await self._relay_answer(reasoning_model, confidence_hint)
        finally:
            self.stop_thinking.set()
            await thinking_task

    async def _respond_directly(self, confidence_hint: Optional[str]):
        answer = (self.decision.get("answer") or "").strip()
        if not answer:
            answer = "Sorry, I don't have a confident answer yet."
        confidence = confidence_hint if confidence_hint in CONFIDENCE_BEHAVIORS else "medium"
        _, gesture = CONFIDENCE_BEHAVIORS[confidence]
        cprint(
            "Robot switches directly to answer mode "
            f"(confidence={confidence}, gesture={gesture})"
        )
        cprint(f"Robot: {answer}")
        cprint(f"Robot (nonverbal): {gesture}")

    async def _relay_thinking(self, thinking_model: ChatGPTSentenceStreamer):
        emitted = 0
        async for cue in thinking_model.stream():
            if self.stop_thinking.is_set():
                break
            if not is_meaningful_cue(cue):
                continue
            cprint(f"Robot (thinking): {cue}")
            emitted += 1
            if emitted >= MAX_THINKING_CUES:
                break

    async def _relay_answer(
        self,
        reasoning_model: ChatGPTSentenceStreamer,
        confidence_hint: Optional[str],
    ):
        gesture = ""
        first_clause = True

        async for clause in reasoning_model.stream():
            if first_clause:
                self.stop_thinking.set()
                confidence_level = resolve_confidence(confidence_hint, reasoning_model.word_count)
                _, gesture = CONFIDENCE_BEHAVIORS[confidence_level]
                cprint(
                    "Robot hands off to answer mode "
                    f"(confidence={confidence_level}, gesture={gesture})"
                )
                first_clause = False

            cprint(f"Robot: {clause}")
        if gesture:
            cprint(f"Robot (nonverbal): {gesture}")


def main():
    question = input("Ask the robot a question: ") or "How do you show thinking?"
    load_api_settings_from_files()
    orchestrator = Orchestrator(question)
    try:
        asyncio.run(orchestrator.run())
    except RuntimeError as err:
        cprint(f"Configuration error: {err}")
    except requests.HTTPError as err:
        cprint(f"OpenAI API error: {err.response.text}")
    except Exception as err:  # pragma: no cover
        cprint(f"Unexpected error: {err}")


if __name__ == "__main__":
    main()
