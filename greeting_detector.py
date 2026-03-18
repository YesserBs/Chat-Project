# This module was vibe coded, but if vibe coding lets me create a library in no time
# so be it.. let's continue

"""
===============================================================================
GREETING DETECTOR LIBRARY
===============================================================================

Purpose
-------
This module provides a lightweight, rule-based greeting detector designed to be
used as a cheap pre-filter before calling a heavier API or LLM endpoint.

Typical use case:
- A user sends a message to your FastAPI backend
- Before forwarding that message to an expensive model/API, you call this
  detector
- If the message is only a greeting ("hello", "bonjour", "hey", "good morning")
  you can answer immediately without calling the expensive service
- If the message contains a real request, you continue normal processing

Why this exists
---------------
This detector is intentionally:
- fast
- local
- dependency-light
- easy to integrate
- easy to serialize to JSON

It is NOT meant to be a full NLP classifier.
It is a practical gatekeeper for obvious "only greeting" messages.

Supported languages
-------------------
- English
- French

What it detects
---------------
Examples likely classified as ONLY_GREETING:
- "hello"
- "hi"
- "bonjour"
- "salut"
- "good morning"
- "how are you"
- "ça va"
- "hey there"

Examples likely classified as NOT_ONLY_GREETING:
- "hello I need help"
- "bonjour je cherche une facture"
- "hi can you explain this"
- "good morning, where is my order?"

Public API
----------
This module is designed to expose a small, stable interface:

1) detect_greeting(text: str) -> GreetingResult
   Returns a structured dataclass with all details.

2) detect_greeting_dict(text: str) -> dict
   Returns the same result as a JSON-serializable dictionary.
   This is usually the most convenient function for FastAPI responses.

3) should_short_circuit(text: str, threshold: float = 0.85) -> bool
   Returns True if the message should be treated as a greeting-only message
   and can be answered locally without calling a heavy downstream API.

4) get_short_circuit_response(text: str, threshold: float = 0.85) -> Optional[dict]
   Returns a ready-to-send response payload if the input is considered a pure
   greeting. Otherwise returns None.

Returned structure
------------------
GreetingResult fields:

- text: original input text
- normalized: normalized version used internally
- tokens: tokenized normalized text
- score: float between 0.0 and 1.0
- label: one of:
    * ONLY_GREETING
    * LIKELY_GREETING
    * AMBIGUOUS
    * NOT_ONLY_GREETING
- suggested_reply: generic reply if appropriate, else None
- reasons: human-readable explanation list for debugging

Recommended production rule
---------------------------
For overload protection, the safest rule is:

    if result.label == "ONLY_GREETING":
        short-circuit and return a local greeting response

Or equivalently:

    if result.score >= THRESHOLD_ONLY_GREETING:
        short-circuit

FastAPI example
---------------
from fastapi import FastAPI
from pydantic import BaseModel
from greeting_detector import detect_greeting_dict, get_short_circuit_response

app = FastAPI()

class AskRequest(BaseModel):
    text: str

@app.post("/ask")
def ask(payload: AskRequest):
    short_response = get_short_circuit_response(payload.text)
    if short_response is not None:
        return short_response

    # Otherwise call your expensive API/model here
    model_response = {"answer": "real downstream response"}
    return {
        "type": "normal_request",
        "response": model_response
    }

Simple usage example
--------------------
result = detect_greeting("hello")
print(result.label)            # ONLY_GREETING
print(result.score)            # e.g. 0.93
print(result.suggested_reply)  # "Hello! How can I help you?"

json_result = detect_greeting_dict("bonjour")
print(json_result["label"])

Notes
-----
- A single shared detector instance is created at module level for efficiency
- The detector is stateless after initialization, so it is safe for normal
  server usage
- This module does not start a loop and does not read input()
- It is meant to be imported and called from your API layer

===============================================================================
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set


# =============================================================================
# SCORING CONSTANTS
# =============================================================================
# Tune these values to adjust the detector's sensitivity.
# Higher thresholds = stricter (fewer false positives, more false negatives).
# Lower thresholds = more tolerant (catches more greetings, may mis-classify).

# --- Label thresholds --------------------------------------------------------
# Score at or above this value -> ONLY_GREETING
THRESHOLD_ONLY_GREETING: float = 0.75      # lowered from 0.85 for more tolerance
# Score at or above this value -> LIKELY_GREETING
THRESHOLD_LIKELY_GREETING: float = 0.55    # lowered from 0.65
# Score at or above this value -> AMBIGUOUS
THRESHOLD_AMBIGUOUS: float = 0.30          # lowered from 0.40
# Below THRESHOLD_AMBIGUOUS   -> NOT_ONLY_GREETING

# --- Short-circuit default threshold -----------------------------------------
# Used by should_short_circuit() and get_short_circuit_response() as default.
# Aligned with THRESHOLD_ONLY_GREETING so both APIs behave consistently.
DEFAULT_SHORT_CIRCUIT_THRESHOLD: float = THRESHOLD_ONLY_GREETING

# --- Suggested-reply threshold -----------------------------------------------
# A local reply is only generated for strong, short greeting matches.
THRESHOLD_SUGGESTED_REPLY: float = THRESHOLD_ONLY_GREETING

# --- Base score bonuses -------------------------------------------------------
SCORE_BASE_GREETING_PRESENCE: float = 0.45   # greeting token found at all
SCORE_GREETING_AT_START: float = 0.18        # greeting is the first token
SCORE_MULTIPLE_GREETINGS: float = 0.10       # 2+ greeting tokens
SCORE_CLASSIC_PATTERN: float = 0.12         # e.g. "how are you", "ça va"

# Length bonuses (message token count)
SCORE_LENGTH_1: float = 0.30   # single token
SCORE_LENGTH_2: float = 0.24   # two tokens
SCORE_LENGTH_3: float = 0.18   # three tokens
SCORE_LENGTH_4: float = 0.12   # four tokens
SCORE_LENGTH_5: float = 0.02   # five tokens

# Length penalties
SCORE_LENGTH_6_PENALTY: float = 0.10   # six tokens (subtracted)
SCORE_LENGTH_LONG_PENALTY: float = 0.35  # 7+ tokens (subtracted)

# Soft-word and filler bonuses (per token, capped)
SCORE_SOFT_WORD_PER_TOKEN: float = 0.04
SCORE_SOFT_WORD_CAP: float = 0.08
SCORE_FILLER_PER_TOKEN: float = 0.02
SCORE_FILLER_CAP: float = 0.04

# --- Penalties ----------------------------------------------------------------
SCORE_REQUEST_MARKER_PER_TOKEN: float = 0.28   # per request marker found
SCORE_REQUEST_MARKER_CAP: float = 0.70         # max total request penalty
SCORE_CONTENT_MARKER_PER_TOKEN: float = 0.10   # per content marker (if n >= 3)
SCORE_CONTENT_MARKER_CAP: float = 0.25         # max total content penalty
SCORE_UNKNOWN_TOKEN_PER_TOKEN: float = 0.06    # per unrecognised token
SCORE_UNKNOWN_TOKEN_CAP: float = 0.24          # max total unknown penalty
SCORE_GREETING_BEFORE_REQUEST: float = 0.12   # greeting at pos 0 + request marker


# =============================================================================
# RESULT MODEL
# =============================================================================

@dataclass(frozen=True)
class GreetingResult:
    """
    Structured output of the greeting detector.

    Attributes:
        text:
            Original input text.

        normalized:
            Normalized text used internally for scoring.

        tokens:
            Token list extracted from the normalized text.

        score:
            Final confidence score between 0.0 and 1.0.

        label:
            Classification label:
            - ONLY_GREETING
            - LIKELY_GREETING
            - AMBIGUOUS
            - NOT_ONLY_GREETING

        suggested_reply:
            Generic local reply when the text is a short pure greeting.
            Otherwise None.

        reasons:
            Human-readable explanations of the scoring process.
            Useful for debugging, logging, or tuning rules.
    """
    text: str
    normalized: str
    tokens: List[str]
    score: float
    label: str
    suggested_reply: Optional[str]
    reasons: List[str]


# =============================================================================
# INTERNAL UTILITIES
# =============================================================================

def distance_1(word1: str, word2: str) -> bool:
    """
    Return True if two words are at edit distance <= 1.

    Supported edits:
    - one substitution
    - one insertion
    - one deletion

    This is used as a very lightweight fuzzy matcher for common typos
    like:
    - "bonjor" ~ "bonjour"
    - "helo" ~ "hello"

    Args:
        word1: first word
        word2: second word

    Returns:
        bool: True if edit distance <= 1, else False
    """
    if abs(len(word1) - len(word2)) > 1:
        return False

    if len(word1) == len(word2):
        diff = sum(1 for a, b in zip(word1, word2) if a != b)
        return diff <= 1

    short, long_ = (word1, word2) if len(word1) < len(word2) else (word2, word1)

    i = 0
    j = 0
    diff = 0

    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            diff += 1
            if diff > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1

    return True


# =============================================================================
# DETECTOR
# =============================================================================

class GreetingDetector:
    """
    Lightweight rule-based greeting detector for EN/FR messages.

    Design goals:
    - fast enough to run before a heavy API call
    - deterministic
    - easy to understand and tune
    - works on short conversational inputs

    Main method:
        score_text(text: str) -> GreetingResult
    """

    ONLY_GREETING = "ONLY_GREETING"
    LIKELY_GREETING = "LIKELY_GREETING"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_ONLY_GREETING = "NOT_ONLY_GREETING"

    def __init__(self) -> None:
        # ---------------------------------------------------------------------
        # SINGLE-WORD GREETINGS
        # ---------------------------------------------------------------------
        self.greeting_single: Set[str] = {
            # English
            "hi", "hello", "hey", "hiya", "yo", "sup",
            "wassup", "whatsup", "whatup", "howdy",
            "hallo", "greetings", "welcome",
            "morning", "evening",

            # French
            "bonjour", "salut", "coucou", "bonsoir",
            "bjr", "slt",

            # Common noisy variants / typos
            "saluuut", "saluuuut", "saalut",
            "banjour", "bonjout", "bsoir", "bjour"
        }

        # ---------------------------------------------------------------------
        # MULTI-WORD EXPRESSIONS -> normalized to one token
        # ---------------------------------------------------------------------
        self.multi_word_map: Dict[str, str] = {
            # English
            "good morning": "goodmorning",
            "good afternoon": "goodafternoon",
            "good evening": "goodevening",
            "how are you": "howareyou",
            "how are u": "howareyou",
            "what's up": "whatsup",
            "what is up": "whatsup",

            # French
            "comment ca va": "commentcava",
            "comment ça va": "commentcava",
            "comment tu vas": "commenttuvas",
            "ca va": "cava",
            "ça va": "cava",
        }

        # ---------------------------------------------------------------------
        # COMPOUND GREETINGS AFTER NORMALIZATION
        # ---------------------------------------------------------------------
        self.greeting_compound: Set[str] = {
            "goodmorning", "goodafternoon", "goodevening",
            "howareyou", "commentcava", "commenttuvas",
            "cava", "whatsup"
        }

        # ---------------------------------------------------------------------
        # SOFT WORDS: acceptable around greetings
        # ---------------------------------------------------------------------
        self.soft_words: Set[str] = {
            "there", "friend", "dear",   # EN
            "toi", "bien"                # FR
        }

        # ---------------------------------------------------------------------
        # WORDS THAT OFTEN INDICATE REAL INTENT / REQUEST
        # ---------------------------------------------------------------------
        self.request_markers: Set[str] = {
            # EN
            "need", "help", "issue", "problem", "question", "please",
            "can", "could", "would", "want", "order", "payment",
            "invoice", "error", "bug", "project", "meeting",
            "send", "check", "tell", "explain",
            "where", "when", "why", "what", "who",

            # FR
            "besoin", "aide", "probleme", "problème", "question",
            "svp", "stp", "peux", "pouvez", "voudrais", "veux",
            "commande", "paiement", "facture", "erreur", "bug",
            "projet", "reunion", "réunion",
            "envoyer", "verifier", "vérifier",
            "dire", "expliquer",
            "ou", "où", "quand", "pourquoi", "quoi", "qui",
            "jai", "aimerais", "souhaite", "cherche"
        }

        # ---------------------------------------------------------------------
        # WORDS SUGGESTING REAL MESSAGE CONTENT
        # ---------------------------------------------------------------------
        self.content_markers: Set[str] = {
            # EN
            "i", "my", "me", "we", "our", "you", "your",

            # FR
            "je", "j", "mon", "ma", "mes",
            "nous", "notre", "nos",
            "vous", "votre", "vos",
            "tu", "ton", "ta", "tes"
        }

        # ---------------------------------------------------------------------
        # FILLERS: mostly neutral
        # ---------------------------------------------------------------------
        self.allowed_fillers: Set[str] = {
            "uh", "um", "ok", "okay",
            "euh", "heu"
        }

        # ---------------------------------------------------------------------
        # PATTERNS THAT STRONGLY LOOK LIKE GREETING FORMS
        # ---------------------------------------------------------------------
        self.classic_patterns: Set[str] = {
            "howareyou", "commentcava", "commenttuvas",
            "cava", "goodmorning", "goodafternoon",
            "goodevening", "whatsup"
        }

    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------
    @staticmethod
    def strip_accents(text: str) -> str:
        """
        Remove accents from input text.

        Example:
            "ça va" -> "ca va"
        """
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def normalize(self, text: str) -> str:
        """
        Normalize input text before tokenization.

        Steps:
        - strip leading/trailing spaces
        - lowercase
        - unify apostrophes
        - remove accents
        - merge known multi-word greeting expressions
        - remove non-alphanumeric punctuation
        - compress spaces
        - normalize a few common forms

        Args:
            text: raw user input

        Returns:
            normalized string
        """
        text = text.strip().lower()
        text = text.replace("'", "'")
        text = self.strip_accents(text)

        for src, dst in self.multi_word_map.items():
            pattern = r"\b" + re.escape(src) + r"\b"
            text = re.sub(pattern, dst, text)

        text = text.replace("what'sup", "whatsup")
        text = text.replace("what s up", "whatsup")

        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Common French normalization
        text = text.replace("j ai", "jai")

        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Split normalized text into tokens.
        """
        return text.split() if text else []

    # -------------------------------------------------------------------------
    # TOKEN MATCHING
    # -------------------------------------------------------------------------
    def token_matches_vocab(self, token: str, vocab: Set[str]) -> bool:
        """
        Return True if token matches vocab exactly or with edit distance <= 1.
        """
        if token in vocab:
            return True
        return any(distance_1(token, word) for word in vocab)

    def is_greeting_token(self, token: str) -> bool:
        return (
            self.token_matches_vocab(token, self.greeting_single)
            or self.token_matches_vocab(token, self.greeting_compound)
        )

    def has_request_marker(self, token: str) -> bool:
        return self.token_matches_vocab(token, self.request_markers)

    def has_content_marker(self, token: str) -> bool:
        return self.token_matches_vocab(token, self.content_markers)

    def is_soft_word(self, token: str) -> bool:
        return self.token_matches_vocab(token, self.soft_words)

    def is_allowed_filler(self, token: str) -> bool:
        return self.token_matches_vocab(token, self.allowed_fillers)

    # -------------------------------------------------------------------------
    # LABELS / REPLIES
    # -------------------------------------------------------------------------
    def classify_label(self, score: float) -> str:
        """
        Convert a score into a label using the module-level threshold constants.
        """
        if score >= THRESHOLD_ONLY_GREETING:
            return self.ONLY_GREETING
        if score >= THRESHOLD_LIKELY_GREETING:
            return self.LIKELY_GREETING
        if score >= THRESHOLD_AMBIGUOUS:
            return self.AMBIGUOUS
        return self.NOT_ONLY_GREETING

    def build_generic_reply(self, tokens: List[str], score: float) -> Optional[str]:
        """
        Return a local reply only for short, strong greeting matches.

        This is intentionally strict because this reply may be used to bypass
        a heavy downstream API call.
        """
        if len(tokens) <= 2 and score >= THRESHOLD_SUGGESTED_REPLY:
            return "Hello! How can I help you?"
        return None

    # -------------------------------------------------------------------------
    # MAIN SCORING
    # -------------------------------------------------------------------------
    def score_text(self, text: str) -> GreetingResult:
        """
        Score an input text and return a structured result.

        Args:
            text: raw user message

        Returns:
            GreetingResult
        """
        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        reasons: List[str] = []

        if not tokens:
            reasons.append("empty input")
            return GreetingResult(
                text=text,
                normalized=normalized,
                tokens=tokens,
                score=0.0,
                label=self.NOT_ONLY_GREETING,
                suggested_reply=None,
                reasons=reasons
            )

        n = len(tokens)

        greeting_positions = [i for i, token in enumerate(tokens) if self.is_greeting_token(token)]
        greeting_count = len(greeting_positions)
        request_count = sum(self.has_request_marker(token) for token in tokens)
        content_count = sum(self.has_content_marker(token) for token in tokens)
        soft_count = sum(self.is_soft_word(token) for token in tokens)
        filler_count = sum(self.is_allowed_filler(token) for token in tokens)

        if greeting_count == 0:
            reasons.append("no greeting detected")
            return GreetingResult(
                text=text,
                normalized=normalized,
                tokens=tokens,
                score=0.0,
                label=self.NOT_ONLY_GREETING,
                suggested_reply=None,
                reasons=reasons
            )

        score = 0.0

        # Core greeting presence
        score += SCORE_BASE_GREETING_PRESENCE
        reasons.append("contains greeting token")

        # Greeting starts the sentence
        if greeting_positions[0] == 0:
            score += SCORE_GREETING_AT_START
            reasons.append("greeting appears at start")

        # Multiple greeting tokens
        if greeting_count >= 2:
            score += SCORE_MULTIPLE_GREETINGS
            reasons.append("multiple greeting tokens")

        # Shorter texts are more likely to be pure greetings
        if n == 1:
            score += SCORE_LENGTH_1
            reasons.append("very short message")
        elif n == 2:
            score += SCORE_LENGTH_2
            reasons.append("short message")
        elif n == 3:
            score += SCORE_LENGTH_3
            reasons.append("compact message")
        elif n == 4:
            score += SCORE_LENGTH_4
        elif n == 5:
            score += SCORE_LENGTH_5
        elif n == 6:
            score -= SCORE_LENGTH_6_PENALTY
            reasons.append("longer message reduces greeting confidence")
        else:
            score -= SCORE_LENGTH_LONG_PENALTY
            reasons.append("message too long for a pure greeting")

        # Known classical greeting patterns
        if any(self.token_matches_vocab(token, self.classic_patterns) for token in tokens):
            score += SCORE_CLASSIC_PATTERN
            reasons.append("contains classic greeting pattern")

        # Soft / filler bonuses
        if soft_count > 0:
            bonus = min(soft_count * SCORE_SOFT_WORD_PER_TOKEN, SCORE_SOFT_WORD_CAP)
            score += bonus
            reasons.append(f"soft-word bonus (+{bonus:.2f})")

        if filler_count > 0:
            bonus = min(filler_count * SCORE_FILLER_PER_TOKEN, SCORE_FILLER_CAP)
            score += bonus
            reasons.append(f"filler bonus (+{bonus:.2f})")

        # Strong penalties for real intent
        if request_count > 0:
            penalty = min(request_count * SCORE_REQUEST_MARKER_PER_TOKEN, SCORE_REQUEST_MARKER_CAP)
            score -= penalty
            reasons.append(f"request-marker penalty (-{penalty:.2f})")

        if content_count > 0 and n >= 3:
            penalty = min(content_count * SCORE_CONTENT_MARKER_PER_TOKEN, SCORE_CONTENT_MARKER_CAP)
            score -= penalty
            reasons.append(f"content-marker penalty (-{penalty:.2f})")

        # Penalize unknown content
        known = greeting_count + request_count + soft_count + filler_count + content_count
        unknown = max(0, n - known)

        if unknown > 0:
            penalty = min(unknown * SCORE_UNKNOWN_TOKEN_PER_TOKEN, SCORE_UNKNOWN_TOKEN_CAP)
            score -= penalty
            reasons.append(f"unknown-token penalty (-{penalty:.2f})")

        # Greeting followed by request is suspicious
        if greeting_positions[0] == 0 and request_count > 0:
            score -= SCORE_GREETING_BEFORE_REQUEST
            reasons.append("greeting followed by likely request")

        score = round(max(0.0, min(1.0, score)), 3)
        label = self.classify_label(score)
        reply = self.build_generic_reply(tokens, score)

        return GreetingResult(
            text=text,
            normalized=normalized,
            tokens=tokens,
            score=score,
            label=label,
            suggested_reply=reply,
            reasons=reasons
        )


# =============================================================================
# SHARED INSTANCE
# =============================================================================

# One shared instance for server usage.
# This avoids rebuilding the vocabularies on every request.
_DETECTOR = GreetingDetector()


# =============================================================================
# PUBLIC LIBRARY FUNCTIONS
# =============================================================================

def detect_greeting(text: str) -> GreetingResult:
    """
    Detect whether a message is only a greeting.

    This is the main library function if you want the full typed result.

    Args:
        text: raw user input

    Returns:
        GreetingResult

    Example:
        result = detect_greeting("hello")
        print(result.label)
        print(result.score)
    """
    return _DETECTOR.score_text(text)


def detect_greeting_dict(text: str) -> Dict[str, object]:
    """
    Detect whether a message is only a greeting and return a JSON-ready dict.

    This is usually the most convenient function for FastAPI responses.

    Args:
        text: raw user input

    Returns:
        dict

    Example:
        result = detect_greeting_dict("bonjour")
        return result
    """
    return asdict(_DETECTOR.score_text(text))


def should_short_circuit(text: str, threshold: float = DEFAULT_SHORT_CIRCUIT_THRESHOLD) -> bool:
    """
    Decide whether the input should be handled locally as a pure greeting.

    Recommended usage:
        if should_short_circuit(user_text):
            return local_response

    Args:
        text: raw user input
        threshold: score threshold for short-circuiting.
                   Defaults to DEFAULT_SHORT_CIRCUIT_THRESHOLD
                   (aligned with THRESHOLD_ONLY_GREETING).

    Returns:
        bool
    """
    result = _DETECTOR.score_text(text)
    return result.score >= threshold


def get_short_circuit_response(
    text: str,
    threshold: float = DEFAULT_SHORT_CIRCUIT_THRESHOLD
) -> Optional[Dict[str, object]]:
    """
    Return a ready-to-send API payload if the message is a pure greeting.

    This is a convenience helper for FastAPI or any backend route.
    If the message is not considered greeting-only, returns None.

    Args:
        text: raw user input
        threshold: confidence threshold used to short-circuit.
                   Defaults to DEFAULT_SHORT_CIRCUIT_THRESHOLD
                   (aligned with THRESHOLD_ONLY_GREETING).

    Returns:
        dict | None

    Example:
        response = get_short_circuit_response("hello")
        if response is not None:
            return response
        # otherwise continue to your heavy API
    """
    result = _DETECTOR.score_text(text)

    if result.score < threshold:
        return None

    return {
        "type": "greeting",
        "detector": asdict(result),
        "response": result.suggested_reply or "Hello!"
    }