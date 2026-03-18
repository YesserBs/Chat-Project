import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GreetingResult:
    """
    Structured output of the detector
    """
    text: str
    normalized: str
    tokens: List[str]
    score: float
    label: str
    suggested_reply: Optional[str]
    reasons: List[str]


def distance_1(word1: str, word2: str) -> bool:
    """
    Returns True if two words are at edit distance <= 1
    (one insertion, deletion or substitution allowed)
    """

    if abs(len(word1) - len(word2)) > 1:
        return False

    # Same length → at most 1 substitution
    if len(word1) == len(word2):
        diff = sum(1 for a, b in zip(word1, word2) if a != b)
        return diff <= 1

    # Length differs by 1 → insertion/deletion
    short, long_ = (word1, word2) if len(word1) < len(word2) else (word2, word1)

    i = j = diff = 0
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


class GreetingDetector:
    """
    Lightweight rule-based greeting detector (FR + EN)

    Goal:
    Detect if a sentence is ONLY a greeting (not a greeting + real request)

    Output:
    - score between 0 and 1
    - label
    - optional generic reply for very short greetings
    """

    def __init__(self):

        # -----------------------------
        # SINGLE-WORD GREETINGS
        # -----------------------------
        self.greeting_single = {
            # English
            "hi", "hello", "hey", "hiya", "yo", "sup",
            "wassup", "whatsup", "whatup", "howdy",
            "hallo", "greetings", "welcome",
            "morning", "evening",

            # French
            "bonjour", "salut", "coucou", "bonsoir",
            "bjr", "slt",
            "saluuut", "saluuuut", "saalut",
            "banjour", "bonjout", "bsoir", "bjour"
        }

        # -----------------------------
        # MULTI-WORD EXPRESSIONS → normalized to one token
        # -----------------------------
        self.multi_word_map = {
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

        # -----------------------------
        # COMPOUND GREETINGS (single tokens after normalization)
        # -----------------------------
        self.greeting_compound = {
            "goodmorning", "goodafternoon", "goodevening",
            "howareyou", "commentcava", "commenttuvas",
            "cava", "whatsup"
        }

        # -----------------------------
        # SOFT WORDS (acceptable in greetings)
        # -----------------------------
        self.soft_words = {
            "there", "friend", "dear",  # EN
            "toi", "bien"               # FR
        }

        # -----------------------------
        # WORDS INDICATING REAL INTENT (NOT pure greeting)
        # -----------------------------
        self.request_markers = {
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

        # -----------------------------
        # CONTENT STRUCTURE WORDS
        # -----------------------------
        self.content_markers = {
            # EN
            "i", "my", "me", "we", "our", "you", "your",
            # FR
            "je", "j", "mon", "ma", "mes",
            "nous", "notre", "nos",
            "vous", "votre", "vos",
            "tu", "ton", "ta", "tes"
        }

        # -----------------------------
        # FILLER WORDS (neutral)
        # -----------------------------
        self.allowed_fillers = {
            "uh", "um", "ok", "okay",
            "euh", "heu", "ok"
        }

    # -----------------------------
    # TEXT NORMALIZATION
    # -----------------------------
    def strip_accents(self, text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def normalize(self, text: str) -> str:
        """
        Normalize text:
        - lowercase
        - remove accents
        - merge known expressions
        - clean punctuation
        """
        text = text.strip().lower()
        text = text.replace("’", "'")
        text = self.strip_accents(text)

        # Merge multi-word greetings
        for src, dst in self.multi_word_map.items():
            pattern = r"\b" + re.escape(src) + r"\b"
            text = re.sub(pattern, dst, text)

        # Normalize variants of "what's up"
        text = text.replace("what'sup", "whatsup")
        text = text.replace("what s up", "whatsup")

        # Remove unwanted characters
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Common fix
        text = text.replace("j ai", "jai")

        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split() if text else []

    # -----------------------------
    # FUZZY MATCHING
    # -----------------------------
    def token_matches_vocab(self, token: str, vocab: set[str]) -> bool:
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

    # -----------------------------
    # LABEL CLASSIFICATION
    # -----------------------------
    def classify_label(self, score: float) -> str:
        if score >= 0.85:
            return "ONLY_GREETING"
        if score >= 0.65:
            return "LIKELY_GREETING"
        if score >= 0.40:
            return "AMBIGUOUS"
        return "NOT_ONLY_GREETING"

    # -----------------------------
    # GENERIC REPLY
    # -----------------------------
    def build_generic_reply(self, tokens: List[str], score: float) -> Optional[str]:
        """
        If message is very short and likely a greeting → return auto reply
        """
        if len(tokens) <= 2 and score > 0:
            return "Hello, I'm here. How can I help you?"
        return None

    # -----------------------------
    # MAIN SCORING FUNCTION
    # -----------------------------
    def score_text(self, text: str) -> GreetingResult:

        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        reasons: List[str] = []

        if not tokens:
            return GreetingResult(text, normalized, tokens, 0.0,
                                  "NOT_ONLY_GREETING", None,
                                  ["empty input"])

        n = len(tokens)

        greeting_positions = [i for i, t in enumerate(tokens) if self.is_greeting_token(t)]
        greeting_count = len(greeting_positions)
        request_count = sum(self.has_request_marker(t) for t in tokens)
        content_count = sum(self.has_content_marker(t) for t in tokens)
        soft_count = sum(self.is_soft_word(t) for t in tokens)
        filler_count = sum(self.is_allowed_filler(t) for t in tokens)

        score = 0.0

        # Must contain at least one greeting token
        if greeting_count == 0:
            return GreetingResult(text, normalized, tokens, 0.0,
                                  "NOT_ONLY_GREETING", None,
                                  ["no greeting detected"])

        score += 0.45

        if greeting_positions[0] == 0:
            score += 0.18

        if greeting_count >= 2:
            score += 0.10

        # Length scoring
        if n == 1:
            score += 0.30
        elif n == 2:
            score += 0.24
        elif n == 3:
            score += 0.18
        elif n == 4:
            score += 0.12
        elif n == 5:
            score += 0.02
        elif n == 6:
            score -= 0.10
        else:
            score -= 0.35

        # Classic patterns
        classic_patterns = {
            "howareyou", "commentcava", "commenttuvas",
            "cava", "goodmorning", "goodafternoon",
            "goodevening", "whatsup"
        }

        if any(self.token_matches_vocab(t, classic_patterns) for t in tokens):
            score += 0.12

        # Soft bonus
        score += min(soft_count * 0.04, 0.08)
        score += min(filler_count * 0.02, 0.04)

        # Penalties
        score -= min(request_count * 0.28, 0.70)

        if content_count > 0 and n >= 3:
            score -= min(content_count * 0.10, 0.25)

        known = greeting_count + request_count + soft_count + filler_count + content_count
        unknown = max(0, n - known)

        score -= min(unknown * 0.06, 0.24)

        if greeting_positions[0] == 0 and request_count > 0:
            score -= 0.12

        score = round(max(0.0, min(1.0, score)), 3)

        label = self.classify_label(score)
        reply = self.build_generic_reply(tokens, score)

        return GreetingResult(text, normalized, tokens, score, label, reply, reasons)


# -----------------------------
# TEST LOOP
# -----------------------------
def main():
    detector = GreetingDetector()

    print("Greeting Detector (FR/EN)")
    print("Type 'exit' to quit\n")

    while True:
        text = input("Input > ").strip()

        if text.lower() in {"exit", "quit"}:
            break

        result = detector.score_text(text)

        print("\n--- Result ---")
        print("Score :", result.score)
        print("Label :", result.label)
        print("Reply :", result.suggested_reply)
        print("----------------\n")


if __name__ == "__main__":
    main()