"""Non-scoring candidates: a reference/filler is not automatically a defect."""
from __future__ import annotations

import re

REFERENCE = re.compile(r"(?<!\w)(?:this|that|it|they|those|these|वो|यह|वह|उस|इस|ये|वही|इसका|उसका)(?!\w)", re.IGNORECASE)
REPAIR = re.compile(r"\b(?:I mean|rather|sorry|मतलब|मेरा मतलब)\b", re.IGNORECASE)
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")


def candidates(transcript: dict) -> list[dict]:
    found = []
    seen = set()
    for segment in transcript.get("segments", []):
        text = segment["text"]
        for kind, expression in (("reference_to_check", REFERENCE), ("self_repair_to_check", REPAIR), ("term_to_check", ACRONYM)):
            for match in expression.finditer(text):
                if kind == "term_to_check" and match.group() in seen:
                    continue
                if kind == "term_to_check":
                    seen.add(match.group())
                found.append({"candidate": kind, "segment_id": segment["id"], "quote": match.group(), "start_char": segment["start_char"] + match.start(), "end_char": segment["start_char"] + match.end(), "is_finding": False})
        words = list(re.finditer(r"\S+", text))
        for left, right in zip(words, words[1:]):
            if left.group() == right.group():
                found.append({"candidate": "exact_repeat_to_check", "segment_id": segment["id"], "quote": text[left.start():right.end()], "start_char": segment["start_char"] + left.start(), "end_char": segment["start_char"] + right.end(), "is_finding": False})
    return found


def delivery_metrics(transcript: dict) -> dict:
    text = transcript.get("text", "")
    count = len(text.split())
    duration = transcript.get("duration_seconds")
    audio = transcript.get("diagnostics", {})
    if not duration:
        duration = audio.get("duration_seconds")
    return {"word_count": count, "approximate_wpm": count * 60 / duration if duration and duration > 0 and count else None,
            "estimated_silence_percentage": audio.get("silence_percent"),
            "exact_repeat_candidates": sum(item["candidate"] == "exact_repeat_to_check" for item in candidates(transcript)),
            "filler_token_count": len(re.findall(r"(?<!\w)(?:um|uh|erm|मतलब|अम्म)(?!\w)", text, re.IGNORECASE)),
            "scored": False, "limitations": "Approximate transcript counts; ASR, Unicode tokenization and code-mixing limit interpretation. No acoustic cause is inferred."}
