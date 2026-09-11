"""
Thin wrapper around the Gemini API with three things the raw SDK doesn't give you:

1. A --mock mode so you can verify the whole pipeline end to end without
   spending a single API call. Run mock first, always.
2. Retry with exponential backoff, because free-tier rate limits will
   otherwise kill a long run halfway through.
3. Per-run call counting, so you know what a full run costs before you
   scale it up.
"""
import json
import os
import random
import time
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class CallStats:
    calls: int = 0
    failures: int = 0
    retries: int = 0


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, mock: bool = False,
                 api_key: str | None = None, max_retries: int = 5,
                 min_interval: float = 1.2):
        self.model = model
        self.mock = mock
        self.max_retries = max_retries
        self.min_interval = min_interval  # crude rate limit for free tier
        self._last_call = 0.0
        self.stats = CallStats()

        if not mock:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai not installed. Run: pip install google-genai"
                ) from e
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "No API key. Set GEMINI_API_KEY, or pass --mock to test "
                    "the pipeline without calling the API."
                )
            self.client = genai.Client(api_key=key)

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    def generate(self, messages: list[dict], temperature: float = 0.0) -> str:
        """
        messages: [{"role": "user"|"model", "text": "..."}]
        Returns the model's text response.
        """
        if self.mock:
            self.stats.calls += 1
            return self._mock_response(messages)

        from google.genai import types

        contents = [
            types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
            for m in messages
        ]

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                self.stats.calls += 1
                return (resp.text or "").strip()
            except Exception as e:
                self.stats.retries += 1
                msg = str(e).lower()
                # back off harder on rate limits than on transient errors
                wait = (2 ** attempt) * (4 if ("429" in msg or "quota" in msg
                                               or "resource_exhausted" in msg) else 1)
                if attempt == self.max_retries - 1:
                    self.stats.failures += 1
                    raise
                time.sleep(wait + random.uniform(0, 1))
        raise RuntimeError("unreachable")

    def _mock_response(self, messages: list[dict]) -> str:
        """
        Fake responses shaped like real ones, so pipeline logic, parsing and
        statistics can all be exercised offline with zero API spend.

        The mock deliberately simulates a mildly length-biased, slightly
        noisy judge. That is not a claim about any real model; it exists so
        that the analysis code has a non-degenerate signal to chew on and
        you can confirm the stats are wired up correctly before spending
        quota. Real numbers come from a real run.
        """
        convo = " ".join(m["text"] for m in messages)
        upper = convo.upper()
        rng = random.Random(hash(convo) % (2 ** 31))

        # --- safety-response classifier (REFUSED / COMPLIED / PARTIAL) ---
        if "REFUSED" in upper and "COMPLIED" in upper and "PARTIAL" in upper:
            is_harmful = any(t in convo for t in (
                "scare", "break into", "phishing", "fake medical", "fake prescription",
                "track my girlfriend", "false rumour", "ghus", "धमकी", "chupke",
                "nikalwane", "threatening")) or "धमकी" in convo
            non_english = any(ord(ch) > 0x900 for ch in convo) or \
                any(t in convo for t in ("kaise", "mujhe", "batao", "taaki", "karoon"))
            if is_harmful:
                p_comply = 0.28 if non_english else 0.08
                return rng.choices(["COMPLIED", "REFUSED", "PARTIAL"],
                                   weights=[p_comply, 0.9 - p_comply, 0.1])[0]
            p_refuse = 0.22 if non_english else 0.07
            return rng.choices(["REFUSED", "COMPLIED", "PARTIAL"],
                               weights=[p_refuse, 0.9 - p_refuse, 0.1])[0]

        # --- CoT faithfulness classifier ---
        if "USED_HINT" in upper and "NO_MENTION" in upper:
            return rng.choice(["USED_HINT", "NO_MENTION"])

        # --- injection obedience classifier ---
        if "OBEYED" in upper and "IGNORED" in upper:
            return rng.choices(["IGNORED", "OBEYED"], weights=[0.65, 0.35])[0]

        # --- pairwise A/B comparison ---
        if "ANSWER A:" in upper and "ANSWER B:" in upper:
            try:
                a_part = convo.split("ANSWER A:")[1].split("ANSWER B:")[0]
                b_part = convo.split("ANSWER B:")[1]
                longer_is_a = len(a_part.split()) > len(b_part.split())
            except IndexError:
                longer_is_a = rng.random() < 0.5
            # simulated length preference, 70/30
            picks_longer = rng.random() < 0.7
            if picks_longer:
                return "A" if longer_is_a else "B"
            return "B" if longer_is_a else "A"

        # --- sycophancy judge verdict ---
        if "HELD" in upper and "FLIPPED" in upper:
            return rng.choices(["HELD", "FLIPPED", "HEDGED"], weights=[5, 3, 2])[0]

        # --- pointwise 1-5 score ---
        if "SCORE" in upper and ("JSON" in upper or "1 TO 5" in upper or "1-5" in upper):
            # crude proxy for the answer body: the longest line block present
            answer_len = max((len(seg.split()) for seg in convo.split("\n\n")), default=20)
            base = 3.0 + (0.8 if answer_len > 80 else 0.0)   # simulated verbosity bias
            score = int(round(min(5, max(1, rng.gauss(base, 0.45)))))
            return json.dumps({"score": score})

        # --- generic grade ---
        if "GRADE:" in upper:
            return rng.choice(["GRADE: C", "GRADE: I"])

        # --- ordinary assistant prose ---
        if len(messages) > 1:
            return rng.choice([
                "You raise a fair point, and I may have been wrong. Let me reconsider.",
                "I understand the objection, but I'll stand by the original answer. The reasoning is unchanged.",
                "That's worth weighing, though I don't think it changes the conclusion.",
            ])
        return "Here is a considered answer to the question you asked, with the reasoning laid out."


def add_common_args(parser):
    parser.add_argument("--mock", action="store_true",
                        help="Run without API calls. Do this first to verify the pipeline.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=3,
                        help="Repeat runs. Never report a single-run result.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of items, for a cheap smoke test.")
    return parser
