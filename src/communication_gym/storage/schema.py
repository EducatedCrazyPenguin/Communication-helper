"""Version-one relational metadata; payloads retain validated schema-versioned detail."""
from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, Float, ForeignKey, Index, Integer, JSON, MetaData, String, Table, Text, text

metadata = MetaData()
SCHEMA_REVISION = "0001_initial"


def foreign(name: str, target: str, *, nullable: bool = True, ondelete: str = "CASCADE") -> Column:
    return Column(name, String(36), ForeignKey(target, ondelete=ondelete, deferrable=True, initially="DEFERRED"), nullable=nullable, index=True)


def entity(name: str, *columns: Column, checks: tuple[str, ...] = ()) -> Table:
    return Table(
        name, metadata,
        Column("id", String(36), primary_key=True),
        Column("created_at", String(40), nullable=False, index=True),
        Column("updated_at", String(40), nullable=False),
        Column("schema_version", Integer, nullable=False, server_default="1"),
        Column("payload", JSON, nullable=False),
        *columns,
        CheckConstraint("json_valid(payload) AND json_type(payload) = 'object'", name=f"ck_{name}_json"),
        *(CheckConstraint(check, name=f"ck_{name}_{index}") for index, check in enumerate(checks)),
    )


sessions = entity(
    "sessions", Column("language", String(16), nullable=False, index=True),
    Column("mode", String(40), nullable=False, index=True), Column("stage", String(40), nullable=False, index=True),
    Column("revision", Integer, nullable=False), Column("generation", Integer, nullable=False, server_default="0"),
    foreign("exercise_id", "exercises.id", ondelete="SET NULL"),
    Column("operation_key", String(200), unique=True),
    checks=("revision >= 0", "language IN ('en','hi','hinglish')", "mode IN ('explain','compare','work_update','proposal','technical_explanation','interview')"),
)
exercises = entity("exercises", foreign("session_id", "sessions.id", nullable=False), Column("source", String(20), nullable=False), Column("mode", String(40)), Column("language", String(16)))
attempts = entity(
    "attempts", foreign("session_id", "sessions.id", nullable=False), Column("role", String(20), nullable=False, index=True),
    Column("ordinal", Integer, nullable=False), Column("status", String(24), nullable=False, index=True),
    foreign("recording_id", "audio_recordings.id", ondelete="SET NULL"),
    foreign("active_transcript_id", "transcripts.id", ondelete="SET NULL"),
    foreign("question_id", "cross_exam_questions.id", ondelete="CASCADE"),
    Column("operation_key", String(200)), Column("generation", Integer, nullable=False, server_default="0"),
    checks=("role IN ('initial','cross_exam','retry')", "ordinal > 0"),
)
Index("uq_attempt_operation", attempts.c.session_id, attempts.c.operation_key, unique=True, sqlite_where=attempts.c.operation_key.is_not(None))
Index("uq_attempt_ordinal", attempts.c.session_id, attempts.c.role, attempts.c.ordinal, unique=True)
Index("uq_accepted_attempt", attempts.c.session_id, attempts.c.role, unique=True, sqlite_where=text("role IN ('initial','retry') AND status IN ('accepted','analyzing','analyzed')"))
audio_recordings = entity("audio_recordings", foreign("session_id", "sessions.id", nullable=False), foreign("attempt_id", "attempts.id"), Column("status", String(20), nullable=False), Column("provenance", String(20), nullable=False), Column("relative_path", Text), Column("sha256", String(64)), Column("duration_seconds", Float))
transcripts = entity("transcripts", foreign("session_id", "sessions.id", nullable=False), foreign("attempt_id", "attempts.id", nullable=False), foreign("parent_id", "transcripts.id"), Column("revision", Integer, nullable=False), Column("kind", String(24), nullable=False), Column("content_hash", String(64), nullable=False), checks=("kind IN ('asr_original','human_corrected')", "revision > 0"))
Index("uq_transcript_revision", transcripts.c.attempt_id, transcripts.c.revision, unique=True)
transcript_segments = entity("transcript_segments", foreign("session_id", "sessions.id", nullable=False), foreign("attempt_id", "attempts.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), Column("ordinal", Integer, nullable=False), Column("start_char", Integer, nullable=False), Column("end_char", Integer, nullable=False))
Index("uq_segment_ordinal", transcript_segments.c.transcript_id, transcript_segments.c.ordinal, unique=True)
word_timings = entity("word_timings", foreign("session_id", "sessions.id", nullable=False), foreign("attempt_id", "attempts.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), foreign("segment_id", "transcript_segments.id", nullable=False), Column("ordinal", Integer, nullable=False), Column("start_char", Integer), Column("end_char", Integer))
Index("uq_word_ordinal", word_timings.c.segment_id, word_timings.c.ordinal, unique=True)
analyses = entity("analyses", foreign("session_id", "sessions.id", nullable=False), foreign("attempt_id", "attempts.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), Column("subsystem", String(80), nullable=False), Column("status", String(24), nullable=False, index=True), Column("cache_key", String(64), index=True), Column("generation", Integer, nullable=False, server_default="0"))
Index("uq_successful_analysis_cache", analyses.c.session_id, analyses.c.cache_key, unique=True, sqlite_where=text("status = 'succeeded' AND cache_key IS NOT NULL"))
communication_issues = entity("communication_issues", foreign("session_id", "sessions.id", nullable=False), foreign("analysis_id", "analyses.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), Column("kind", String(64), nullable=False, index=True), Column("source", String(40), nullable=False), Column("severity", String(16), nullable=False))
issue_evidence = entity("issue_evidence", foreign("session_id", "sessions.id", nullable=False), foreign("analysis_id", "analyses.id", nullable=False), foreign("issue_id", "communication_issues.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), foreign("segment_id", "transcript_segments.id", nullable=False), Column("start_char", Integer, nullable=False), Column("end_char", Integer, nullable=False))
score_snapshots = entity("score_snapshots", foreign("session_id", "sessions.id", nullable=False), foreign("analysis_id", "analyses.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), Column("rubric_version", String(80)))
listener_models = entity("listener_models", foreign("session_id", "sessions.id", nullable=False), foreign("analysis_id", "analyses.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), foreign("question_id", "cross_exam_questions.id"), Column("kind", String(40)))
coach_feedback = entity("coach_feedback", foreign("session_id", "sessions.id", nullable=False), foreign("analysis_id", "analyses.id", nullable=False), foreign("transcript_id", "transcripts.id", nullable=False), Column("ordinal", Integer, nullable=False))
cross_exam_questions = entity("cross_exam_questions", foreign("session_id", "sessions.id", nullable=False), foreign("analysis_id", "analyses.id"), Column("ordinal", Integer, nullable=False), Column("status", String(20), nullable=False, index=True), Column("operation_key", String(200)), checks=("status IN ('open','answered','skipped')", "ordinal > 0", "ordinal <= 10"))
Index("uq_open_question", cross_exam_questions.c.session_id, unique=True, sqlite_where=text("status = 'open'"))
Index("uq_question_ordinal", cross_exam_questions.c.session_id, cross_exam_questions.c.ordinal, unique=True)
retry_comparisons = entity("retry_comparisons", foreign("session_id", "sessions.id", nullable=False), foreign("initial_analysis_id", "analyses.id", nullable=False), foreign("retry_analysis_id", "analyses.id", nullable=False), Column("status", String(20), nullable=False))
personal_patterns = entity("personal_patterns", foreign("session_id", "sessions.id"), Column("kind", String(64), index=True), Column("group_key", String(300), index=True), Column("confidence", String(16)))
training_goals = entity("training_goals", foreign("session_id", "sessions.id"), Column("kind", String(64), index=True), Column("status", String(20), nullable=False))
jobs = entity("jobs", foreign("session_id", "sessions.id"), foreign("attempt_id", "attempts.id"), Column("operation", String(80), nullable=False, index=True), Column("status", String(24), nullable=False, index=True), Column("generation", Integer, nullable=False, server_default="0"), Column("operation_key", String(200)), Column("revision", Integer), checks=("status IN ('queued','running','cancel_requested','succeeded','failed','cancelled','interrupted')",))
Index("uq_job_operation", jobs.c.session_id, jobs.c.operation, jobs.c.operation_key, unique=True, sqlite_where=jobs.c.operation_key.is_not(None))
deletion_tasks = entity("deletion_tasks", Column("session_id", String(36), index=True), Column("status", String(24), nullable=False, index=True), Column("relative_path", Text), Column("attempts", Integer, nullable=False, server_default="0"))

TABLES = dict(metadata.tables)
IMMUTABLE_TABLES = frozenset({"exercises", "transcripts", "transcript_segments", "word_timings", "communication_issues", "issue_evidence", "score_snapshots", "listener_models", "coach_feedback", "retry_comparisons"})


def install_immutability(connection) -> None:
    for name in sorted(IMMUTABLE_TABLES):
        connection.exec_driver_sql(f"CREATE TRIGGER IF NOT EXISTS immutable_{name} BEFORE UPDATE ON {name} BEGIN SELECT RAISE(ABORT, 'immutable_record'); END")
    connection.exec_driver_sql("CREATE TRIGGER IF NOT EXISTS immutable_successful_analysis BEFORE UPDATE ON analyses WHEN OLD.status = 'succeeded' BEGIN SELECT RAISE(ABORT, 'immutable_record'); END")
