"""Validated domain contracts. Public application interfaces use JSON-safe dictionaries."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class GymError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.status, self.retryable = code, message, status, retryable

    def as_dict(self) -> dict:
        error = {"code": self.code, "message": self.message, "retryable": self.retryable}
        if hasattr(self, "current_revision"):
            error["current_revision"] = self.current_revision
        return {"error": error}


def new_id() -> str:
    return str(uuid4())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_id(value: str, label: str = "ID") -> str:
    try:
        parsed = UUID(value)
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise GymError("validation_error", f"{label} must be a canonical UUID4.", 422) from None
    return value


def content_hash(value: Any) -> str:
    try:
        serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeError):
        raise GymError("validation_error", "Content must be finite JSON with valid Unicode text.", 422) from None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True, allow_inf_nan=False)


class Language(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"


class Mode(StrEnum):
    EXPLAIN = "explain"
    COMPARE = "compare"
    WORK_UPDATE = "work_update"
    PROPOSAL = "proposal"
    TECHNICAL_EXPLANATION = "technical_explanation"
    INTERVIEW = "interview"


class Stage(StrEnum):
    TOPIC = "topic"
    PREP = "prep"
    INITIAL_RECORDING = "initial_recording"
    INITIAL_TRANSCRIBING = "initial_transcribing"
    INITIAL_REVIEW = "initial_review"
    INITIAL_ANALYSIS = "initial_analysis"
    BREAKDOWN = "breakdown"
    CROSS_EXAM = "cross_exam"
    RETRY_RECORDING = "retry_recording"
    RETRY_TRANSCRIBING = "retry_transcribing"
    RETRY_REVIEW = "retry_review"
    RETRY_ANALYSIS = "retry_analysis"
    COMPARISON = "comparison"
    COMPLETE = "complete"


class ProviderCapabilities(StrictModel):
    available: bool
    provider_id: str
    model_id: str | None = None
    locality: Literal["local", "cloud"] = "local"
    languages: list[str] = Field(default_factory=list)
    structured_output: bool = False
    word_timing: Literal["none", "estimated", "aligned"] = "none"
    word_confidence: bool = False
    cancellation: Literal["cooperative", "terminate_owned", "client_only", "none"] = "none"
    release_supported: bool = False
    unavailable_reason: str | None = None
    configured: dict[str, JsonValue] = Field(default_factory=dict)
    verified: dict[str, JsonValue] = Field(default_factory=dict)


class Exercise(StrictModel):
    id: str = Field(default_factory=new_id)
    session_id: str | None = None
    schema_version: int = 1
    source: Literal["curated", "generated", "custom"] = "custom"
    title: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=12000)
    audience: str = Field(min_length=1, max_length=1000)
    mode: Literal["explain", "compare", "work_update", "proposal", "technical_explanation", "interview"]
    language: Literal["en", "hi", "hinglish"]
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "session_id")
    @classmethod
    def valid_ids(cls, value: str | None) -> str | None:
        return require_id(value) if value is not None else None


class Session(StrictModel):
    id: str = Field(default_factory=new_id)
    schema_version: int = 1
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    language: Literal["en", "hi", "hinglish"] = "en"
    mode: Literal["explain", "compare", "work_update", "proposal", "technical_explanation", "interview"] = "explain"
    stage: Literal["topic", "prep", "initial_recording", "initial_transcribing", "initial_review", "initial_analysis", "breakdown", "cross_exam", "retry_recording", "retry_transcribing", "retry_review", "retry_analysis", "comparison", "complete"] = "topic"
    revision: int = Field(default=0, ge=0)
    generation: int = Field(default=0, ge=0)
    exercise_id: str | None = None
    prep_bullets: list[str] = Field(default_factory=list, max_length=5)
    prep_seconds: int = Field(default=120, ge=120, le=300)
    planned_duration: int = 120
    retry_duration: int | None = None
    equal_retry_duration_confirmed: bool = False
    consent: dict[str, JsonValue] = Field(default_factory=dict)
    operation_key: str | None = None
    operation_hash: str | None = None
    completed_at: str | None = None
    questions_finished: bool = False
    questions_skipped: bool = False
    active_comparison_id: str | None = None

    @field_validator("id", "exercise_id", "active_comparison_id")
    @classmethod
    def valid_ids(cls, value: str | None) -> str | None:
        return require_id(value) if value is not None else None

    @field_validator("planned_duration")
    @classmethod
    def valid_duration(cls, value: int) -> int:
        if value not in (60, 120, 300, 600, 900):
            raise ValueError("Choose a 60, 120, 300, 600 or 900 second attempt.")
        return value

    @field_validator("prep_bullets")
    @classmethod
    def valid_bullets(cls, values: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 200 for item in values):
            raise ValueError("Use at most five nonempty bullets of at most 200 code points each.")
        return values

    @model_validator(mode="after")
    def valid_retry(self) -> Session:
        if self.retry_duration is not None:
            if not 30 <= self.retry_duration <= self.planned_duration:
                raise ValueError("Retry must be at least 30 seconds and no longer than the initial attempt.")
            if self.retry_duration == self.planned_duration and not self.equal_retry_duration_confirmed:
                raise ValueError("Equal retry duration requires an explicit choice.")
        return self


class Attempt(StrictModel):
    id: str = Field(default_factory=new_id)
    session_id: str
    role: Literal["initial", "cross_exam", "retry"]
    ordinal: int = Field(ge=1)
    status: Literal["created", "recording", "transcribing", "review", "accepted", "analyzing", "analyzed", "failed", "cancelled", "interrupted"] = "created"
    recording_id: str | None = None
    active_transcript_id: str | None = None
    question_id: str | None = None
    planned_duration: int = Field(default=120, ge=30, le=900)
    captured_duration: float | None = Field(default=None, ge=0, le=902)
    reviewed: bool = False
    operation_key: str | None = None
    generation: int = Field(default=0, ge=0)


class EvidenceRef(StrictModel):
    transcript_id: str
    segment_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)
    timing_kind: Literal["none", "text_only", "estimated", "aligned", "segment", "approximate", "unknown"] = "none"

    @model_validator(mode="after")
    def valid_span(self) -> EvidenceRef:
        if self.end_char < self.start_char or (self.start_s is not None and self.end_s is not None and self.end_s < self.start_s):
            raise ValueError("Evidence ends before it starts.")
        return self


class ReviewedRange(StrictModel):
    transcript_id: str
    segment_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


ISSUE_KINDS = frozenset({"missing_context", "ambiguous_reference", "unnamed_entity", "unclear_relationship", "implicit_comparison", "missing_cause", "buried_conclusion", "topic_jump", "unfinished_thought", "tangent", "unnecessary_detail", "repetition", "contradiction", "chronology", "ungrounded_term", "missing_consequence", "missing_transition", "question_not_answered", "obscuring_hedging", "information_overload", "listener_load", "word_retrieval", "obstructive_filler", "vague_quantity", "weak_call_to_action", "weak_proposal"})


class CommunicationIssue(StrictModel):
    id: str = Field(default_factory=new_id)
    kind: str
    severity: Literal["minor", "costly", "blocking"]
    source: Literal["likely_speaker_issue", "possible_transcription_error"]
    evidence: list[EvidenceRef] = Field(default_factory=list)
    reviewed_range: list[ReviewedRange] = Field(default_factory=list)
    listener_consequence: str
    corrective_pattern: str
    confidence: Literal["low", "medium", "high"]
    target: str = ""

    @field_validator("kind")
    @classmethod
    def known_kind(cls, value: str) -> str:
        if value not in ISSUE_KINDS:
            raise ValueError("Unknown communication issue kind.")
        return value

    @model_validator(mode="after")
    def grounded(self) -> CommunicationIssue:
        if not self.evidence and not self.reviewed_range:
            raise ValueError("Issues require evidence or reviewed coverage.")
        return self


class SessionQuery(StrictModel):
    language: Literal["en", "hi", "hinglish"] | None = None
    mode: str | None = None
    stage: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class TranscriptionRequest(StrictModel):
    request_id: str
    attempt_id: str
    source: str
    language: Literal["en", "hi", "hinglish"]
    checkpoint_dir: str


class CoachRequest(StrictModel):
    request_id: str
    task: str
    prompt_version: str
    prompt: str
    input: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class CoachResponse(StrictModel):
    text: str
    provider_id: str
    model_id: str
    duration_seconds: float = Field(ge=0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SynthesisRequest(StrictModel):
    request_id: str
    text: str = Field(min_length=1, max_length=4096)
    voice_id: str
    language: Literal["en", "hi", "hinglish"]


class TopicRequest(StrictModel):
    mode: str
    language: Literal["en", "hi", "hinglish"]
    exclude_ids: list[str] = Field(default_factory=list)


# Storage validates these envelopes in addition to task-specific application schemas.
# Arbitrary provider text is never interpreted as a validated analysis by this layer.
class RecordEnvelope(StrictModel):
    id: str
    schema_version: int = Field(default=1, ge=1)
    created_at: str
    updated_at: str
    payload: dict[str, JsonValue]

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return require_id(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("Timestamps must be timezone-aware UTC.")
        return value
