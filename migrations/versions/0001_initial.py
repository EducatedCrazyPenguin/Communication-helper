"""Initial durable training state and indexed evidence.

Revision ID: 0001_initial
Revises: None
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

# Frozen version-one descriptors: future schema changes belong in new revisions.
PARENTS = {
    "exercises": {"session_id": "sessions"},
    "attempts": {"session_id": "sessions", "recording_id": "audio_recordings", "active_transcript_id": "transcripts", "question_id": "cross_exam_questions"},
    "audio_recordings": {"session_id": "sessions", "attempt_id": "attempts"},
    "transcripts": {"session_id": "sessions", "attempt_id": "attempts", "parent_id": "transcripts"},
    "transcript_segments": {"session_id": "sessions", "attempt_id": "attempts", "transcript_id": "transcripts"},
    "word_timings": {"session_id": "sessions", "attempt_id": "attempts", "transcript_id": "transcripts", "segment_id": "transcript_segments"},
    "analyses": {"session_id": "sessions", "attempt_id": "attempts", "transcript_id": "transcripts"},
    "communication_issues": {"session_id": "sessions", "analysis_id": "analyses", "transcript_id": "transcripts"},
    "issue_evidence": {"session_id": "sessions", "analysis_id": "analyses", "issue_id": "communication_issues", "transcript_id": "transcripts", "segment_id": "transcript_segments"},
    "score_snapshots": {"session_id": "sessions", "analysis_id": "analyses", "transcript_id": "transcripts"},
    "listener_models": {"session_id": "sessions", "analysis_id": "analyses", "transcript_id": "transcripts", "question_id": "cross_exam_questions"},
    "coach_feedback": {"session_id": "sessions", "analysis_id": "analyses", "transcript_id": "transcripts"},
    "cross_exam_questions": {"session_id": "sessions", "analysis_id": "analyses"},
    "retry_comparisons": {"session_id": "sessions", "initial_analysis_id": "analyses", "retry_analysis_id": "analyses"},
    "personal_patterns": {"session_id": "sessions"},
    "training_goals": {"session_id": "sessions"},
    "jobs": {"session_id": "sessions", "attempt_id": "attempts"},
}
REQUIRED = {
    "sessions": {"language", "mode", "stage", "revision", "generation"},
    "exercises": {"session_id", "source"},
    "attempts": {"session_id", "role", "ordinal", "status", "generation"},
    "audio_recordings": {"session_id", "status", "provenance"},
    "transcripts": {"session_id", "attempt_id", "revision", "kind", "content_hash"},
    "transcript_segments": {"session_id", "attempt_id", "transcript_id", "ordinal", "start_char", "end_char"},
    "word_timings": {"session_id", "attempt_id", "transcript_id", "segment_id", "ordinal"},
    "analyses": {"session_id", "attempt_id", "transcript_id", "subsystem", "status", "generation"},
    "communication_issues": {"session_id", "analysis_id", "transcript_id", "kind", "source", "severity"},
    "issue_evidence": {"session_id", "analysis_id", "issue_id", "transcript_id", "segment_id", "start_char", "end_char"},
    "score_snapshots": {"session_id", "analysis_id", "transcript_id"},
    "listener_models": {"session_id", "analysis_id", "transcript_id"},
    "coach_feedback": {"session_id", "analysis_id", "transcript_id", "ordinal"},
    "cross_exam_questions": {"session_id", "ordinal", "status"},
    "retry_comparisons": {"session_id", "initial_analysis_id", "retry_analysis_id", "status"},
    "training_goals": {"status"},
    "jobs": {"operation", "status", "generation"},
    "deletion_tasks": {"status", "attempts"},
}
FIELDS = {
    "sessions": {"language": sa.String(16), "mode": sa.String(40), "stage": sa.String(40), "revision": sa.Integer(), "generation": sa.Integer(), "exercise_id": sa.String(36), "operation_key": sa.String(200)},
    "exercises": {"source": sa.String(20), "mode": sa.String(40), "language": sa.String(16)},
    "attempts": {"role": sa.String(20), "ordinal": sa.Integer(), "status": sa.String(24), "operation_key": sa.String(200), "generation": sa.Integer()},
    "audio_recordings": {"status": sa.String(20), "provenance": sa.String(20), "relative_path": sa.Text(), "sha256": sa.String(64), "duration_seconds": sa.Float()},
    "transcripts": {"revision": sa.Integer(), "kind": sa.String(24), "content_hash": sa.String(64)},
    "transcript_segments": {"ordinal": sa.Integer(), "start_char": sa.Integer(), "end_char": sa.Integer()},
    "word_timings": {"ordinal": sa.Integer(), "start_char": sa.Integer(), "end_char": sa.Integer()},
    "analyses": {"subsystem": sa.String(80), "status": sa.String(24), "cache_key": sa.String(64), "generation": sa.Integer()},
    "communication_issues": {"kind": sa.String(64), "source": sa.String(40), "severity": sa.String(16)},
    "issue_evidence": {"start_char": sa.Integer(), "end_char": sa.Integer()},
    "score_snapshots": {"rubric_version": sa.String(80)},
    "listener_models": {"kind": sa.String(40)},
    "coach_feedback": {"ordinal": sa.Integer()},
    "cross_exam_questions": {"ordinal": sa.Integer(), "status": sa.String(20), "operation_key": sa.String(200)},
    "retry_comparisons": {"status": sa.String(20)},
    "personal_patterns": {"kind": sa.String(64), "group_key": sa.String(300), "confidence": sa.String(16)},
    "training_goals": {"kind": sa.String(64), "status": sa.String(20)},
    "jobs": {"operation": sa.String(80), "status": sa.String(24), "generation": sa.Integer(), "operation_key": sa.String(200), "revision": sa.Integer()},
    "deletion_tasks": {"session_id": sa.String(36), "status": sa.String(24), "relative_path": sa.Text(), "attempts": sa.Integer()},
}
CHECKS = {
    "sessions": ["revision >= 0", "language IN ('en','hi','hinglish')", "mode IN ('explain','compare','work_update','proposal','technical_explanation','interview')"],
    "attempts": ["role IN ('initial','cross_exam','retry')", "ordinal > 0"],
    "transcripts": ["kind IN ('asr_original','human_corrected')", "revision > 0"],
    "cross_exam_questions": ["status IN ('open','answered','skipped')", "ordinal > 0", "ordinal <= 10"],
    "jobs": ["status IN ('queued','running','cancel_requested','succeeded','failed','cancelled','interrupted')"],
}
IMMUTABLE = ("exercises", "transcripts", "transcript_segments", "word_timings", "communication_issues", "issue_evidence", "score_snapshots", "listener_models", "coach_feedback", "retry_comparisons")
EXTRA_INDEXES = {
    "sessions": ("language", "mode", "stage"), "attempts": ("role", "status"), "analyses": ("status", "cache_key"),
    "communication_issues": ("kind",), "cross_exam_questions": ("status",), "personal_patterns": ("kind", "group_key"),
    "training_goals": ("kind",), "jobs": ("operation", "status"), "deletion_tasks": ("session_id", "status"),
}


def upgrade():
    for name, fields in FIELDS.items():
        columns = [sa.Column("id", sa.String(36), primary_key=True), sa.Column("created_at", sa.String(40), nullable=False), sa.Column("updated_at", sa.String(40), nullable=False), sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"), sa.Column("payload", sa.JSON(), nullable=False)]
        parents = dict(PARENTS.get(name, {}))
        if name == "sessions":
            parents["exercise_id"] = "exercises"
        for key, parent in parents.items():
            ondelete = "SET NULL" if key in ("exercise_id", "recording_id", "active_transcript_id") else "CASCADE"
            columns.append(sa.Column(key, sa.String(36), sa.ForeignKey(f"{parent}.id", ondelete=ondelete, deferrable=True, initially="DEFERRED"), nullable=key not in REQUIRED.get(name, set())))
        for key, column_type in fields.items():
            if key not in parents:
                default = "0" if key in ("generation", "attempts") else None
                columns.append(sa.Column(key, column_type, nullable=key not in REQUIRED.get(name, set()), server_default=default))
        columns.append(sa.CheckConstraint("json_valid(payload) AND json_type(payload) = 'object'", name=f"ck_{name}_json"))
        columns.extend(sa.CheckConstraint(check, name=f"ck_{name}_{index}") for index, check in enumerate(CHECKS.get(name, [])))
        op.create_table(name, *columns)
        for field in {"created_at", *parents, *EXTRA_INDEXES.get(name, ())}:
            op.create_index(f"ix_{name}_{field}", name, [field])
    op.create_index("uq_sessions_operation_key", "sessions", ["operation_key"], unique=True)
    for name, table, fields, condition in [
        ("uq_attempt_operation", "attempts", ["session_id", "operation_key"], "operation_key IS NOT NULL"),
        ("uq_attempt_ordinal", "attempts", ["session_id", "role", "ordinal"], None),
        ("uq_accepted_attempt", "attempts", ["session_id", "role"], "role IN ('initial','retry') AND status IN ('accepted','analyzing','analyzed')"),
        ("uq_transcript_revision", "transcripts", ["attempt_id", "revision"], None),
        ("uq_segment_ordinal", "transcript_segments", ["transcript_id", "ordinal"], None),
        ("uq_word_ordinal", "word_timings", ["segment_id", "ordinal"], None),
        ("uq_successful_analysis_cache", "analyses", ["session_id", "cache_key"], "status = 'succeeded' AND cache_key IS NOT NULL"),
        ("uq_open_question", "cross_exam_questions", ["session_id"], "status = 'open'"),
        ("uq_question_ordinal", "cross_exam_questions", ["session_id", "ordinal"], None),
        ("uq_job_operation", "jobs", ["session_id", "operation", "operation_key"], "operation_key IS NOT NULL"),
    ]:
        options = {"sqlite_where": sa.text(condition)} if condition else {}
        op.create_index(name, table, fields, unique=True, **options)
    for name in IMMUTABLE:
        op.execute(sa.text(f"CREATE TRIGGER immutable_{name} BEFORE UPDATE ON {name} BEGIN SELECT RAISE(ABORT, 'immutable_record'); END"))
    op.execute(sa.text("CREATE TRIGGER immutable_successful_analysis BEFORE UPDATE ON analyses WHEN OLD.status = 'succeeded' BEGIN SELECT RAISE(ABORT, 'immutable_record'); END"))


def downgrade():
    raise RuntimeError("Destructive schema downgrade is unsupported. Restore the pre-migration owned backup with the matching application version.")
