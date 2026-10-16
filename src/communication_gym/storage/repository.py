"""SQLite persistence with explicit collections, revisions and owned-file boundaries."""
from __future__ import annotations

import copy
import json
import shutil
import math
import sqlite3
import zipfile
import threading
from contextlib import closing, contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator
from uuid import UUID

from alembic import command
from alembic.config import Config
from filelock import FileLock, Timeout
from pydantic import TypeAdapter, JsonValue, ValidationError
from sqlalchemy import Integer, create_engine, delete, event, select, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from communication_gym.domain.models import Attempt, CommunicationIssue, EvidenceRef, Exercise, GymError, RecordEnvelope, Session, content_hash, new_id, require_id, utc_now
from communication_gym.storage.schema import IMMUTABLE_TABLES, SCHEMA_REVISION, TABLES

_JSON = TypeAdapter(dict[str, JsonValue])


def _json_copy(value: dict) -> dict:
    try:
        _JSON.validate_python(value, strict=True)
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (ValidationError, ValueError, TypeError, OverflowError):
        raise GymError("validation_error", "Provide a finite JSON object with supported fields.", 422) from None


class Repository:
    """One instance per owned data root. Each mutation is transactionally durable."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.database_path = self.root / "gym.sqlite3"
        self.engine: Engine | None = None
        self._lock: FileLock | None = None
        self._connection: ContextVar[Connection | None] = ContextVar(f"gym_connection_{id(self)}", default=None)
        self._init_guard = threading.Lock()

    def initialize(self) -> None:
        self._initialize(migrate=True)

    def _initialize(self, *, migrate: bool) -> None:
        with self._init_guard:
            if self.engine is not None:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            if self.database_path.is_symlink():
                raise GymError("unsafe_path", "The Gym database must remain inside its owned root.", 422)
            lock = FileLock(str(self.root / ".instance.lock"))
            try:
                lock.acquire(timeout=0)
            except Timeout:
                raise GymError("instance_running", "Communication Gym is already using this data directory. Close that instance first.", 409) from None
            self._lock = lock
            try:
                if (self.root / ".delete-all-pending").exists():
                    self.purge_all_files()
                marker = self.root / ".communication-gym-owned"
                if marker.exists() and (marker.is_symlink() or marker.read_text(encoding="utf-8") != "CommunicationGym:1\n"):
                    raise GymError("unsafe_data_root", "The selected directory is not a recognized Gym data root.", 409)
                if not marker.exists() and any(child.name not in {".instance.lock", "settings.json", "settings.json.tmp"} for child in self.root.iterdir()):
                    raise GymError("unsafe_data_root", "Choose an empty directory for Gym-owned storage; unrelated existing data will not be adopted.", 409)
                for directory in ("data", "backups", "logs", "exports", "staging"):
                    candidate = self.root / directory
                    if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
                        raise GymError("unsafe_path", "Managed storage directories cannot be links or junctions.", 422)
                marker.write_text("CommunicationGym:1\n", encoding="utf-8")
                (self.root / "data").mkdir(exist_ok=True)
                self.engine = create_engine(f"sqlite:///{self.database_path.as_posix()}", connect_args={"timeout": 10, "check_same_thread": False})
                event.listen(self.engine, "connect", self._configure_connection)
                if migrate:
                    self._migrate()
                    self.interrupt_jobs()
            except BaseException:
                self.close()
                raise

    @staticmethod
    def _configure_connection(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=FULL")
        finally:
            cursor.close()

    def _migrate(self, runner=None) -> None:
        assert self.engine is not None
        with self.engine.connect() as connection:
            exists = connection.exec_driver_sql("SELECT 1 FROM sqlite_master WHERE type='table' AND name='alembic_version'").scalar()
            current = connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar() if exists else None
            if current not in (None, SCHEMA_REVISION):
                raise GymError("unsupported_database_schema", "This database uses an unknown or newer schema. Use its matching application version; no data was changed.", 409)
            if current == SCHEMA_REVISION and runner is None:
                return
            other_tables = connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'").fetchall()
            if current is None and other_tables:
                raise GymError("unsupported_database_schema", "An unversioned nonempty database cannot be migrated safely. Restore a Gym backup or choose an empty data root.", 409)
        backup = None
        if self.database_path.exists():
            backup_directory = self.root / "backups"
            backup_directory.mkdir(exist_ok=True)
            backup = backup_directory / f"{new_id()}.sqlite3"
            with closing(sqlite3.connect(self.database_path)) as source, closing(sqlite3.connect(backup)) as target:
                source.backup(target)
        config = Config()
        config.set_main_option("script_location", str(Path(__file__).resolve().parents[3] / "migrations"))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{self.database_path.as_posix()}".replace("%", "%%"))
        try:
            with self.engine.begin() as connection:
                if runner is None:
                    config.attributes["connection"] = connection
                    command.upgrade(config, "head")
                else:
                    runner(connection)
        except Exception:
            self.engine.dispose()
            if backup is not None:
                with closing(sqlite3.connect(backup)) as source, closing(sqlite3.connect(self.database_path)) as target:
                    source.backup(target)
            raise GymError("database_migration_failed", "Migration failed; the pre-migration database was restored. Keep the owned backups and use the matching application version.", 500) from None

    def close(self, *, release_lock: bool = True) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
        if release_lock and self._lock is not None:
            self._lock.release()
            self._lock = None

    def purge_all_files(self) -> None:
        """Resume an explicitly confirmed purge; retain ownership until every file is gone."""
        marker = self.root / ".communication-gym-owned"
        pending = self.root / ".delete-all-pending"
        if marker.is_symlink() or not marker.is_file() or marker.read_text(encoding="utf-8") != "CommunicationGym:1\n":
            raise GymError("unsafe_data_root", "The data root ownership marker is invalid.", 409)
        if pending.exists() and (pending.is_symlink() or pending.read_text(encoding="utf-8") != "CommunicationGym:delete-all:1\n"):
            raise GymError("unsafe_data_root", "The pending deletion marker is invalid.", 409)
        # Preflight the whole tree before deleting anything, including nested junctions.
        self.assert_unlinked_tree(self.root)
        try:
            if not pending.exists():
                staging = self.root / ".delete-all-pending.tmp"
                staging.write_text("CommunicationGym:delete-all:1\n", encoding="utf-8")
                staging.replace(pending)
            for child in self.root.iterdir():
                if child.name in {marker.name, pending.name, ".instance.lock"}:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            # A crash between these two unlinks is safe: only the ownership marker remains.
            pending.unlink()
            marker.unlink()
        except OSError:
            raise GymError("deletion_pending", "A locked Gym file remains. Close playback and retry deletion; data is not reported removed.", 409, True) from None

    def assert_unlinked_tree(self, path: Path) -> None:
        """Reject links anywhere in a purge tree rather than following or adopting them."""
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise GymError("unsafe_path", "A linked artifact prevents deletion; remove the link deliberately.", 409)
        try:
            path.resolve().relative_to(self.root)
        except (OSError, ValueError):
            raise GymError("unsafe_path", "The artifact path escapes Gym storage.", 422) from None
        if path.is_dir():
            for child in path.iterdir():
                self.assert_unlinked_tree(child)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        existing = self._connection.get()
        if existing is not None:
            yield existing
            return
        if self.engine is None:
            raise GymError("repository_closed", "The local database is not open.", 503, True)
        try:
            with self.engine.connect() as connection:
                # Reserve the single SQLite writer before reading a revision. This avoids
                # deferred-transaction upgrade races across concurrent request threads.
                connection.exec_driver_sql("BEGIN IMMEDIATE")
                token = self._connection.set(connection)
                try:
                    yield connection
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
                finally:
                    self._connection.reset(token)
        except IntegrityError:
            raise GymError("conflict", "This change conflicts with an existing record or its parent. Reload the session.", 409) from None
        except SQLAlchemyError:
            raise GymError("storage_unavailable", "The local database could not complete the operation. Your previous checkpoint is unchanged.", 503, True) from None

    def _table(self, collection: str):
        if collection not in TABLES:
            raise GymError("validation_error", "Unknown storage collection.", 422)
        return TABLES[collection]

    def path(self, relative: str) -> Path:
        """Resolve artifact names only; no arbitrary paths, drive names or symlink escapes."""
        if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\x00" in relative:
            raise GymError("unsafe_path", "Use a managed relative artifact path.", 422)
        parts = PurePosixPath(relative).parts
        if PurePosixPath(relative).is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
            raise GymError("unsafe_path", "The artifact path must stay inside Gym storage.", 422)
        if not parts or parts[0] not in {"data", "exports", "backups", "logs", "staging"}:
            raise GymError("unsafe_path", "The artifact is outside the managed storage directories.", 422)
        if any(PureWindowsPath(part).is_reserved() or part.endswith((".", " ")) or any(ord(char) < 32 or char in '<>"|?*' for char in part) for part in parts):
            raise GymError("unsafe_path", "The artifact name contains an unsafe Windows path component.", 422)
        # Every artifact has a session/recording/export UUID owner, including filenames.
        uuid_present = False
        for part in parts[1:]:
            for candidate in (part, part.split(".", 1)[0]):
                try:
                    parsed = UUID(candidate)
                    uuid_present |= parsed.version == 4 and str(parsed) == candidate
                except (ValueError, AttributeError):
                    pass
        if not uuid_present:
            raise GymError("unsafe_path", "Managed artifacts require a UUID owner in their relative name.", 422)
        candidate = self.root.joinpath(*parts)
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            raise GymError("unsafe_path", "The artifact path escapes Gym storage.", 422) from None
        for parent in (candidate, *candidate.parents):
            if parent == self.root:
                break
            if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
                raise GymError("unsafe_path", "Symbolic links and junctions cannot own Gym artifacts.", 422)
        return resolved

    def _row(self, collection: str, record_id: str, connection: Connection) -> dict:
        require_id(record_id)
        table = self._table(collection)
        row = connection.execute(select(table.c.payload).where(table.c.id == record_id)).scalar_one_or_none()
        if row is None:
            raise GymError("not_found", "The requested record no longer exists.", 404)
        return copy.deepcopy(row)

    def get(self, collection: str, record_id: str) -> dict:
        with self.transaction() as connection:
            return self._row(collection, record_id, connection)

    def find(self, collection: str, **equal_filters) -> list[dict]:
        table = self._table(collection)
        statement = select(table.c.payload)
        for key, value in equal_filters.items():
            if key not in table.c or key == "payload":
                raise GymError("validation_error", "Filter on an indexed collection metadata field.", 422)
            statement = statement.where(table.c[key] == value)
        statement = statement.order_by(table.c.created_at, table.c.id)
        with self.transaction() as connection:
            return [copy.deepcopy(row) for row in connection.execute(statement).scalars()]

    def _prepare(self, collection: str, payload: dict, previous: dict | None, connection: Connection) -> dict:
        data = _json_copy(payload)
        data.setdefault("id", new_id())
        require_id(data["id"])
        data.setdefault("schema_version", 1)
        data["created_at"] = previous["created_at"] if previous else data.get("created_at", utc_now())
        data["updated_at"] = utc_now()
        for key in ("session_id", "attempt_id", "transcript_id", "analysis_id", "segment_id", "issue_id", "parent_id", "question_id", "recording_id", "active_transcript_id", "initial_analysis_id", "retry_analysis_id"):
            if data.get(key) is not None:
                require_id(data[key], key)
        for key in ("relative_path", "prepared_path", "original_path", "original_json_path", "checkpoint_path", "artifact_path", "export_path"):
            if data.get(key):
                self.path(data[key])
        # Resolve every supplied ownership chain, never trust a duplicated session ID.
        table = self._table(collection)
        parent_specs = (("attempt_id", "attempts"), ("transcript_id", "transcripts"), ("analysis_id", "analyses"), ("issue_id", "communication_issues"), ("segment_id", "transcript_segments"), ("question_id", "cross_exam_questions"))
        for key, parent_collection in parent_specs:
            if data.get(key) is None:
                continue
            parent = self._row(parent_collection, data[key], connection)
            for owner in ("session_id", "attempt_id", "transcript_id", "analysis_id"):
                if owner not in table.c or owner == key or parent.get(owner) is None:
                    continue
                if data.get(owner) not in (None, parent[owner]):
                    raise GymError("ownership_conflict", "The referenced records belong to different artifacts.", 409)
                data[owner] = parent[owner]
        if collection in ("audio_recordings", "transcripts") and data.get("session_id"):
            owner = f"data/sessions/{data['session_id']}/"
            for key in ("relative_path", "prepared_path", "original_path", "original_json_path", "checkpoint_path", "artifact_path"):
                if data.get(key) and not data[key].startswith(owner):
                    raise GymError("ownership_conflict", "The artifact path belongs to a different session.", 409)
        for column in table.c:
            if column.foreign_keys and not column.nullable and data.get(column.name) is None:
                raise GymError("validation_error", "The artifact is missing its required parent: " + column.name + ".", 422)
            if isinstance(column.type, Integer) and column.name in data and (type(data[column.name]) is not int and (data[column.name] is not None or not column.nullable)):
                raise GymError("validation_error", "Artifact counters and offsets must be integers.", 422)
        if collection == "audio_recordings" and data.get("duration_seconds") is not None and (type(data["duration_seconds"]) not in (int, float) or not math.isfinite(data["duration_seconds"]) or not 0 <= data["duration_seconds"] <= 902):
            raise GymError("validation_error", "Recording duration must be the decoded duration within the fifteen-minute limit.", 422)
        if data.get("session_id") is not None and collection != "deletion_tasks":
            self._row("sessions", data["session_id"], connection)
        if collection != "jobs" and data.get("job_id"):
            job = self._row("jobs", data["job_id"], connection)
            if job["status"] not in ("queued", "running") or (job.get("session_id") and job["session_id"] != data.get("session_id")):
                raise GymError("stale_generation", "This operation was cancelled, interrupted or superseded; its result was not saved.", 409)
            if "generation" in data and data["generation"] != job["generation"]:
                raise GymError("stale_generation", "This result belongs to an earlier operation generation.", 409)
        if collection == "sessions":
            try:
                data = Session.model_validate(data).model_dump(mode="json")
            except ValidationError:
                raise GymError("validation_error", "The session settings, preparation or duration are invalid.", 422) from None
        elif collection == "exercises":
            known = {key: value for key, value in data.items() if key in Exercise.model_fields}
            try:
                Exercise.model_validate(known)
            except ValidationError:
                raise GymError("validation_error", "Provide a complete exercise snapshot with title, prompt and audience.", 422) from None
        elif collection == "attempts":
            if data.get("role") not in ("initial", "cross_exam", "retry"):
                raise GymError("validation_error", "Choose an initial, cross-exam or retry attempt.", 422)
            data.setdefault("ordinal", 1)
            data.setdefault("status", "created")
            data.setdefault("generation", 0)
            if data["role"] == "cross_exam" and not data.get("question_id"):
                raise GymError("validation_error", "A cross-exam answer must reference its question.", 422)
            if data.get("recording_id"):
                recording = self._row("audio_recordings", data["recording_id"], connection)
                if recording.get("attempt_id") != data["id"]:
                    raise GymError("ownership_conflict", "The recording belongs to a different attempt.", 409)
            if data.get("active_transcript_id"):
                transcript = self._row("transcripts", data["active_transcript_id"], connection)
                if transcript["attempt_id"] != data["id"]:
                    raise GymError("ownership_conflict", "The transcript belongs to a different attempt.", 409)
            try:
                Attempt.model_validate({key: value for key, value in data.items() if key in Attempt.model_fields})
            except ValidationError:
                raise GymError("validation_error", "The attempt status, duration or metadata is invalid.", 422) from None
        elif collection == "audio_recordings":
            data.setdefault("status", "partial")
            data.setdefault("provenance", "microphone")
            if data["status"] not in ("partial", "ready", "deleted") or data["provenance"] not in ("microphone", "upload", "tts"):
                raise GymError("validation_error", "The recording status or provenance is invalid.", 422)
            for key in ("relative_path", "prepared_path", "original_path"):
                if data.get(key):
                    self.path(data[key])
        elif collection == "transcripts":
            data.setdefault("kind", "asr_original")
            if data["kind"] not in ("asr_original", "human_corrected") or not isinstance(data.get("text"), str):
                raise GymError("validation_error", "A transcript requires exact text and revision provenance.", 422)
            existing = connection.execute(select(TABLES[collection].c.payload).where(TABLES[collection].c.attempt_id == data["attempt_id"])).scalars().all()
            data.setdefault("revision", max((row["revision"] for row in existing), default=0) + 1)
            data["content_hash"] = content_hash(data["text"])
            if data["kind"] == "asr_original" and data.get("parent_id"):
                raise GymError("validation_error", "An original ASR transcript cannot be a correction.", 422)
            if data["kind"] == "human_corrected":
                parent = self._row("transcripts", data.get("parent_id", ""), connection)
                if parent["attempt_id"] != data["attempt_id"] or data["revision"] <= parent["revision"]:
                    raise GymError("ownership_conflict", "A correction must extend this attempt's earlier revision.", 409)
            self._validate_segments(data)
        elif collection == "analyses":
            data.setdefault("subsystem", "communication")
            data.setdefault("status", "succeeded")
            data.setdefault("generation", 0)
            if data["status"] not in ("queued", "running", "succeeded", "failed", "cancelled", "interrupted", "incomplete", "partial"):
                raise GymError("validation_error", "Invalid analysis status.", 422)
            if data["status"] == "succeeded":
                transcript = self._row("transcripts", data["transcript_id"], connection)
                sources = self.find("transcripts", session_id=data["session_id"]) if data["subsystem"] == "model_answer" else []
                self._validate_grounding(data, transcript, sources)
                for issue in data.get("critic", {}).get("issues", data.get("issues", [])):
                    try:
                        CommunicationIssue.model_validate({key: value for key, value in issue.items() if key in CommunicationIssue.model_fields})
                    except (ValidationError, AttributeError):
                        raise GymError("validation_error", "Analysis issues require validated severity, provenance and spoken evidence.", 422) from None
                session = self._row("sessions", data["session_id"], connection)
                exercise = self._row("exercises", session["exercise_id"], connection) if session.get("exercise_id") else None
                data["cache_key"] = content_hash({
                    "session_id": data["session_id"], "transcript_id": transcript["id"], "transcript_hash": transcript["content_hash"],
                    "exercise": exercise, "language": session["language"], "subsystem": data["subsystem"], "input_hash": data.get("input_hash"),
                    "provenance": data.get("provenance", {}), "parameters": data.get("parameters", data.get("generation_parameters", {})),
                    "versions": {key: data.get(key) for key in ("schema_version", "prompt_version", "rubric_version", "provider", "model", "context_tokens")},
                })
        elif collection in ("communication_issues", "issue_evidence", "score_snapshots", "listener_models", "coach_feedback"):
            transcript = self._row("transcripts", data["transcript_id"], connection)
            self._validate_grounding(data, transcript)
            if collection == "communication_issues":
                try:
                    CommunicationIssue.model_validate({key: value for key, value in data.items() if key in CommunicationIssue.model_fields})
                except ValidationError:
                    raise GymError("validation_error", "Communication findings must satisfy the evidence contract.", 422) from None
        elif collection == "cross_exam_questions":
            data.setdefault("status", "open")
            data.setdefault("ordinal", len(self.find("cross_exam_questions", session_id=data["session_id"])) + 1)
            if data["status"] not in ("open", "answered", "skipped") or not 1 <= data["ordinal"] <= 10 or not isinstance(data.get("question"), str) or not data["question"].strip():
                raise GymError("validation_error", "Provide one nonempty question within the ten-turn limit.", 422)
        elif collection == "retry_comparisons":
            data.setdefault("status", "succeeded")
            for key in ("initial_analysis_id", "retry_analysis_id"):
                analysis = self._row("analyses", data.get(key, ""), connection)
                if analysis["session_id"] != data["session_id"]:
                    raise GymError("ownership_conflict", "Comparison analyses must belong to this session.", 409)
        elif collection == "jobs":
            data.setdefault("status", "queued")
            data.setdefault("generation", 0)
            if data["status"] not in ("queued", "running", "cancel_requested", "succeeded", "failed", "cancelled", "interrupted"):
                raise GymError("validation_error", "Invalid job status.", 422)
            if previous and previous["status"] in ("succeeded", "failed", "cancelled", "interrupted") and data["status"] != previous["status"]:
                raise GymError("stale_generation", "A finished job cannot be restarted or accept a late result. Start a new generation.", 409)
            if previous and data["generation"] < previous.get("generation", 0):
                raise GymError("stale_generation", "A job generation cannot move backwards.", 409)
        elif collection == "deletion_tasks":
            data.setdefault("status", "queued")
            data.setdefault("attempts", 0)
            if data.get("relative_path"):
                self.path(data["relative_path"])
        elif collection == "training_goals":
            data.setdefault("status", "active")
        try:
            RecordEnvelope.model_validate({"id": data["id"], "schema_version": data["schema_version"], "created_at": data["created_at"], "updated_at": data["updated_at"], "payload": data})
        except ValidationError:
            raise GymError("validation_error", "Artifact IDs, UTC timestamps or schema metadata are invalid.", 422) from None
        return data

    @staticmethod
    def _validate_grounding(payload: dict, transcript: dict, additional_transcripts: list[dict] | None = None) -> None:
        sources = {source["id"]: source for source in [transcript, *(additional_transcripts or [])]}
        segments = {(source["id"], segment["id"]): segment for source in sources.values() for segment in source.get("segments", [])}

        def visit(value):
            if isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                if all(key in value for key in ("transcript_id", "segment_id", "start_char", "end_char")):
                    source = sources.get(value["transcript_id"])
                    segment = segments.get((value["transcript_id"], value["segment_id"]))
                    start, end = value["start_char"], value["end_char"]
                    if source is None or segment is None or type(start) is not int or type(end) is not int or not segment["start_char"] <= start <= end <= segment["end_char"]:
                        raise GymError("invalid_evidence", "Evidence must resolve inside its exact transcript segment.", 422)
                    if "quote" in value and source["text"][start:end] != value["quote"]:
                        raise GymError("invalid_evidence", "Evidence quotes must match the exact canonical Unicode span.", 422)
                    if "quote" in value:
                        try:
                            EvidenceRef.model_validate({key: item for key, item in value.items() if key in EvidenceRef.model_fields})
                        except ValidationError:
                            raise GymError("invalid_evidence", "Evidence timing or quote metadata is invalid.", 422) from None
                for item in value.values():
                    visit(item)

        visit(payload)
        dimensions = payload.get("dimensions", payload.get("scores", {}).get("dimensions", {}))
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100) for value in dimensions.values()):
            raise GymError("invalid_evidence", "Scores must be unavailable or finite values from zero to one hundred.", 422)

    @staticmethod
    def _validate_segments(transcript: dict) -> None:
        text = transcript["text"]
        transcript.setdefault("segments", [{"id": new_id(), "text": text}] if text else [])
        if not isinstance(transcript["segments"], list) or any(not isinstance(item, dict) or not isinstance(item.get("text"), str) for item in transcript["segments"]) or "\n".join(item["text"] for item in transcript["segments"]) != text:
            raise GymError("validation_error", "Canonical text must join the exact segment texts with one newline.", 422)
        cursor = 0
        for index, segment in enumerate(transcript.get("segments", [])):
            if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                raise GymError("validation_error", "Transcript segments require exact Unicode text.", 422)
            segment.setdefault("id", new_id())
            require_id(segment["id"])
            segment["transcript_id"] = transcript["id"]
            start = segment.setdefault("start_char", cursor)
            end = segment.setdefault("end_char", start + len(segment["text"]))
            if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text) or text[start:end] != segment["text"] or start < cursor:
                raise GymError("validation_error", "Transcript offsets must resolve to the exact canonical text.", 422)
            cursor = end + 1
            for word in segment.get("words", []):
                if not isinstance(word, dict) or not isinstance(word.get("text", word.get("word")), str):
                    raise GymError("validation_error", "Words require their original text.", 422)
                word.setdefault("id", new_id())
                require_id(word["id"])
                word["segment_id"] = segment["id"]
                word["transcript_id"] = transcript["id"]
                left, right = word.get("start_char"), word.get("end_char")
                if left is not None or right is not None:
                    token = word.get("text", word.get("word"))
                    if type(left) is not int or type(right) is not int or not start <= left <= right <= end or text[left:right] != token:
                        raise GymError("validation_error", "Word offsets must match their exact transcript span.", 422)

    def put(self, collection: str, payload: dict) -> dict:
        table = self._table(collection)
        if collection == "sessions":
            record_id = payload.get("id")
            if record_id:
                existing = self.find("sessions", id=record_id)
                if existing:
                    return self.update_session(record_id, {key: value for key, value in payload.items() if key not in {"id", "revision", "created_at", "updated_at"}}, payload.get("revision"))
            return self.create_session(payload)
        with self.transaction() as connection:
            record_id = payload.get("id") or new_id()
            require_id(record_id)
            previous = connection.execute(select(table.c.payload).where(table.c.id == record_id)).scalar_one_or_none()
            merged = {**previous, **payload, "id": record_id} if previous else {**payload, "id": record_id}
            if previous and (collection in IMMUTABLE_TABLES or (collection == "analyses" and previous.get("status") == "succeeded")):
                requested = {key: value for key, value in merged.items() if key != "updated_at"}
                stored = {key: value for key, value in previous.items() if key != "updated_at"}
                if requested == stored:
                    return copy.deepcopy(previous)
                raise GymError("immutable_record", "Historical evidence cannot be overwritten. Create a new revision or analysis.", 409)
            if previous:
                for key in ("session_id", "attempt_id", "transcript_id", "analysis_id", "segment_id", "issue_id", "parent_id", "question_id"):
                    if previous.get(key) is not None and merged.get(key) != previous[key]:
                        raise GymError("ownership_conflict", "A stored artifact cannot change its owner.", 409)
            if previous and collection == "audio_recordings" and previous["status"] in ("ready", "deleted"):
                for key in ("sha256", "media_type", "duration_seconds", "sample_rate", "size_bytes", "provenance", "relative_path", "prepared_path", "original_path"):
                    if merged.get(key) != previous.get(key) and not (key.endswith("path") and merged.get("status") == "deleted" and merged.get(key) is None):
                        raise GymError("immutable_record", "A sealed recording cannot be replaced. Record a new attempt instead.", 409)
                if previous["status"] == "deleted" and merged.get("status") != "deleted":
                    raise GymError("immutable_record", "Deleted audio cannot be restored by changing its metadata.", 409)
            data = self._prepare(collection, merged, previous, connection)
            if collection == "analyses" and data["status"] == "succeeded" and not previous:
                cached = self.find("analyses", session_id=data["session_id"], cache_key=data["cache_key"], status="succeeded")
                if cached:
                    return cached[0]
            values = {column.name: data.get(column.name) for column in table.c if column.name != "payload"}
            values["payload"] = data
            missing = [column.name for column in table.c if not column.nullable and column.name != "payload" and values.get(column.name) is None]
            if missing:
                raise GymError("validation_error", "The artifact is missing required ownership or metadata fields: " + ", ".join(missing) + ".", 422)
            if previous:
                connection.execute(update(table).where(table.c.id == record_id).values(**values))
            else:
                connection.execute(table.insert().values(**values))
                if collection == "transcripts":
                    self._store_transcript_children(data)
            if collection == "analyses" and data["status"] == "succeeded":
                self._store_analysis_children(data)
            return copy.deepcopy(data)

    def _store_transcript_children(self, transcript: dict) -> None:
        for ordinal, segment in enumerate(transcript.get("segments", []), 1):
            child = {**segment, "transcript_id": transcript["id"], "attempt_id": transcript["attempt_id"], "session_id": transcript["session_id"], "ordinal": ordinal}
            self.put("transcript_segments", child)
            for word_index, word in enumerate(segment.get("words", []), 1):
                self.put("word_timings", {**word, "transcript_id": transcript["id"], "attempt_id": transcript["attempt_id"], "session_id": transcript["session_id"], "segment_id": segment["id"], "ordinal": word_index})

    def _store_analysis_children(self, analysis: dict) -> None:
        owner = {"session_id": analysis["session_id"], "analysis_id": analysis["id"], "transcript_id": analysis["transcript_id"]}
        for issue in analysis.get("critic", {}).get("issues", analysis.get("issues", [])):
            if not isinstance(issue, dict):
                raise GymError("validation_error", "Validated issue records must be objects.", 422)
            stored = self.put("communication_issues", {**issue, **owner})
            for evidence in issue.get("evidence", []):
                if evidence.get("segment_id"):
                    self.put("issue_evidence", {**evidence, **owner, "id": new_id(), "issue_id": stored["id"]})
        scores = analysis.get("scores")
        if scores is not None:
            self.put("score_snapshots", {**scores, **owner, "id": new_id()})
        listener = analysis.get("listener")
        if listener is not None:
            self.put("listener_models", {**listener, **owner, "id": new_id(), "kind": "independent"})
        for ordinal, item in enumerate(analysis.get("feedback", []), 1):
            self.put("coach_feedback", {**item, **owner, "id": new_id(), "ordinal": ordinal})

    def create_session(self, payload: dict) -> dict:
        with self.transaction() as connection:
            data = _json_copy(payload)
            data.setdefault("operation_hash", content_hash({key: value for key, value in data.items() if key not in {"id", "operation_key", "created_at", "updated_at"}}))
            exercise = data.pop("exercise", None)
            data.setdefault("id", new_id())
            require_id(data["id"])
            if connection.execute(select(TABLES["sessions"].c.id).where(TABLES["sessions"].c.id == data["id"])).scalar_one_or_none():
                raise GymError("conflict", "The session already exists.", 409)
            if data.get("operation_key"):
                matches = self.find("sessions", operation_key=data["operation_key"])
                if matches:
                    if matches[0].get("operation_hash") != data.get("operation_hash"):
                        raise GymError("operation_conflict", "This operation key was already used with different input.", 409)
                    return self.get_session(matches[0]["id"])
            if exercise:
                exercise = {**exercise, "id": new_id(), "session_id": data["id"], "mode": data.get("mode", "explain"), "language": data.get("language", "en")}
                data["exercise_id"] = exercise["id"]
            data = self._prepare("sessions", data, None, connection)
            table = TABLES["sessions"]
            connection.execute(table.insert().values(**{column.name: data.get(column.name) for column in table.c if column.name != "payload"}, payload=data))
            if exercise:
                self.put("exercises", exercise)
            return self.get_session(data["id"])

    def update_session(self, session_id: str, changes: dict, expected_revision: int | None = None) -> dict:
        with self.transaction() as connection:
            previous = self._row("sessions", session_id, connection)
            if expected_revision is not None and (type(expected_revision) is not int or previous["revision"] != expected_revision):
                error = GymError("revision_conflict", "This session changed. Reload the current revision before continuing.", 409)
                error.current_revision = previous["revision"]
                raise error
            if set(changes) & {"id", "created_at", "updated_at", "revision"}:
                raise GymError("validation_error", "IDs, timestamps and revisions are managed by the repository.", 422)
            changing_exercise = any(key in changes and changes[key] != previous.get(key) for key in ("mode", "language", "exercise_id"))
            if changing_exercise:
                initial_attempts = self.find("attempts", session_id=session_id, role="initial")
                accepted_recording = any(item.get("reviewed") or (item.get("recording_id") and self.get("audio_recordings", item["recording_id"])["status"] == "ready") for item in initial_attempts)
                if accepted_recording:
                    raise GymError("exercise_locked", "Start a new session to change the exercise, language or mode.", 409)
                if changes.get("exercise_id"):
                    exercise = self._row("exercises", changes["exercise_id"], connection)
                    if exercise["session_id"] != session_id:
                        raise GymError("ownership_conflict", "Exercise snapshots belong to their original session.", 409)
            data = self._prepare("sessions", {**previous, **changes, "revision": previous["revision"] + 1}, previous, connection)
            table = TABLES["sessions"]
            result = connection.execute(update(table).where(table.c.id == session_id, table.c.revision == previous["revision"]).values(**{column.name: data.get(column.name) for column in table.c if column.name != "payload"}, payload=data))
            if result.rowcount != 1:
                raise GymError("revision_conflict", "The session changed. Reload it before continuing.", 409)
            return self.get_session(session_id)

    def advance_session(self, session_id: str, stage: str, expected_revision: int | None = None) -> dict:
        return self.update_session(session_id, {"stage": stage}, expected_revision)

    def get_attempt(self, attempt_id: str) -> dict:
        with self.transaction():
            attempt = self.get("attempts", attempt_id)
            attempt["transcripts"] = self.find("transcripts", attempt_id=attempt_id)
            active = attempt.get("active_transcript_id")
            attempt["transcript"] = self.get("transcripts", active) if active else (attempt["transcripts"][-1] if attempt["transcripts"] else None)
            attempt["recording"] = self.get("audio_recordings", attempt["recording_id"]) if attempt.get("recording_id") else None
            attempt["analyses"] = self.find("analyses", attempt_id=attempt_id)
            selected = [item for item in attempt["analyses"] if item.get("status") == "succeeded" and item["transcript_id"] == active and item.get("subsystem") in ("complete", "communication")]
            attempt["analysis"] = selected[-1] if selected else None
            return attempt

    def get_transcript(self, transcript_id: str) -> dict:
        return self.get("transcripts", transcript_id)

    def get_session(self, session_id: str) -> dict:
        with self.transaction():
            session = self.get("sessions", session_id)
            session["exercise"] = self.get("exercises", session["exercise_id"]) if session.get("exercise_id") else None
            session["attempts"] = sorted((self.get_attempt(row["id"]) for row in self.find("attempts", session_id=session_id)), key=lambda row: (row["created_at"], row["ordinal"]))
            session["questions"] = sorted(self.find("cross_exam_questions", session_id=session_id), key=lambda row: row["ordinal"])
            session["analyses"] = self.find("analyses", session_id=session_id)
            session["recordings"] = self.find("audio_recordings", session_id=session_id)
            comparisons = self.find("retry_comparisons", session_id=session_id)
            session["comparisons"] = comparisons
            active_comparison = session.get("active_comparison_id")
            session["comparison"] = next((row for row in comparisons if row["id"] == active_comparison), None)
            if not active_comparison and comparisons and session["stage"] in ("comparison", "complete"):
                session["comparison"] = comparisons[-1]
            for role in ("initial", "retry"):
                accepted = [row for row in session["attempts"] if row["role"] == role and row.get("reviewed") and row.get("active_transcript_id")]
                attempt = accepted[-1] if accepted else None
                session[f"{role}_attempt"] = attempt
                session[f"{role}_analysis"] = attempt.get("analysis") if attempt else None
            session["resume_stage"] = self._resume_stage(session)
            return session

    @staticmethod
    def _resume_stage(session: dict) -> str:
        stage = session["stage"]
        if stage == "complete":
            return stage
        for attempt in reversed(session["attempts"]):
            if attempt["status"] in ("failed", "cancelled", "interrupted"):
                continue
            if attempt.get("role") in ("initial", "retry"):
                prefix = attempt["role"]
                if attempt.get("transcript") and not attempt.get("reviewed"):
                    return f"{prefix}_review"
                if attempt.get("reviewed") and not attempt.get("analysis"):
                    return f"{prefix}_analysis"
            break
        return stage

    def list_sessions(self) -> list[dict]:
        with self.transaction():
            return [self.get_session(row["id"]) for row in reversed(self.find("sessions"))]

    def remove(self, collection: str, record_id: str) -> None:
        if collection == "sessions":
            self.delete_session(record_id)
            return
        if collection in IMMUTABLE_TABLES or collection == "analyses":
            raise GymError("immutable_record", "Delete the session to remove historical evidence.", 409)
        with self.transaction() as connection:
            self._row(collection, record_id, connection)
            connection.execute(delete(self._table(collection)).where(self._table(collection).c.id == record_id))

    def interrupt_jobs(self) -> None:
        with self.transaction():
            for status in ("queued", "running", "cancel_requested"):
                for job in self.find("jobs", status=status):
                    self.put("jobs", {**job, "status": "interrupted", "generation": job.get("generation", 0) + 1, "safe_error": "server_restarted", "finished_at": utc_now()})
                    if job.get("attempt_id"):
                        attempt = self.get("attempts", job["attempt_id"])
                        if attempt["status"] in ("recording", "transcribing", "analyzing"):
                            self.put("attempts", {**attempt, "status": "interrupted", "generation": attempt.get("generation", 0) + 1})

    def invalidate_session_derivatives(self, session_id: str) -> None:
        """Remove recomputable aggregate caches; immutable historical analyses remain."""
        with self.transaction() as connection:
            for collection in ("personal_patterns", "training_goals"):
                table = TABLES[collection]
                for row in self.find(collection):
                    refs = row.get("evidence_session_ids", row.get("session_ids", []))
                    evidence_refs = {item.get("session_id") for item in row.get("evidence", []) if isinstance(item, dict)}
                    if collection == "personal_patterns" or row.get("session_id") == session_id or session_id in refs or session_id in evidence_refs:
                        connection.execute(delete(table).where(table.c.id == row["id"]))

    def session_export_paths(self, session_id: str) -> set[str]:
        """Find complete or interrupted internal exports without trusting ZIP member paths."""
        require_id(session_id)
        paths = set()
        export_root = self.root / "exports"
        if not export_root.exists():
            return paths
        for directory in export_root.iterdir():
            self.path(directory.relative_to(self.root).as_posix())
            manifest = directory / "manifest.json"
            archive = directory / "session.zip"
            try:
                if manifest.is_file():
                    self.path(manifest.relative_to(self.root).as_posix())
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                elif archive.is_file():
                    self.path(archive.relative_to(self.root).as_posix())
                    # Existing exports store their manifest in the archive itself.
                    with zipfile.ZipFile(archive) as bundle:
                        payload = json.loads(bundle.read("manifest.json"))
                else:
                    continue
                if payload.get("session_id") == session_id:
                    paths.add(directory.relative_to(self.root).as_posix())
            except (OSError, ValueError, AttributeError, KeyError, zipfile.BadZipFile):
                raise GymError("export_manifest_invalid", "An internal export manifest could not be read safely; no session data was removed.", 409) from None
        return paths

    def delete_session(self, session_id: str) -> None:
        with self.transaction() as connection:
            session = self._row("sessions", session_id, connection)
            self.update_session(session_id, {"generation": session.get("generation", 0) + 1})
            # Session directory ownership also catches interrupted checkpoints/staging that
            # were never registered as ready artifacts. Purges survive deleting the parent.
            paths = {f"data/sessions/{session_id}"}
            # Migration snapshots can retain deleted transcript rows. They cannot be
            # surgically redacted safely, so retire all owned snapshots on deletion.
            backup_directory = self.root / "backups"
            if backup_directory.exists():
                for backup in backup_directory.iterdir():
                    relative = backup.relative_to(self.root).as_posix()
                    self.path(relative)
                    paths.add(relative)
            for collection in ("audio_recordings", "transcripts", "jobs"):
                for row in self.find(collection, session_id=session_id):
                    for key in ("relative_path", "prepared_path", "original_path", "original_json_path", "checkpoint_path", "artifact_path", "export_path"):
                        if isinstance(row.get(key), str):
                            self.path(row[key])
                            paths.add(row[key])
            paths.update(self.session_export_paths(session_id))
            for relative in sorted(paths):
                self.put("deletion_tasks", {"id": new_id(), "session_id": session_id, "relative_path": relative, "status": "queued"})
            self.invalidate_session_derivatives(session_id)
            # Break nullable circular references before cascading children.
            for attempt in self.find("attempts", session_id=session_id):
                self.put("attempts", {**attempt, "active_transcript_id": None, "recording_id": None})
            connection.execute(update(TABLES["sessions"]).where(TABLES["sessions"].c.id == session_id).values(exercise_id=None))
            connection.execute(delete(TABLES["sessions"]).where(TABLES["sessions"].c.id == session_id))
