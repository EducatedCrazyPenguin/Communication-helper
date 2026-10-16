"""Owned database migration, locking and immutable-record invariants."""
from contextlib import closing
from types import SimpleNamespace
import sqlite3

import pytest
from alembic import command

from communication_gym.application.sessions import SessionService
from communication_gym.domain.models import GymError, new_id
from communication_gym.storage.repository import Repository
from communication_gym.storage.schema import SCHEMA_REVISION, TABLES


EXPECTED = {"sessions", "exercises", "attempts", "audio_recordings", "transcripts", "transcript_segments", "word_timings", "analyses", "communication_issues", "issue_evidence", "score_snapshots", "listener_models", "coach_feedback", "cross_exam_questions", "retry_comparisons", "personal_patterns", "training_goals", "jobs", "deletion_tasks"}


def sample(repo):
    service = SessionService(repo)
    session = service.create({"language": "hinglish", "mode": "compare", "exercise": {"title": "दो बैटरी 🔋", "prompt": "A और B के roles बताओ.", "audience": "New colleague", "source": "custom"}})
    session = service.prep(session["id"], ["पहले नाम", "फिर relationship"], session["revision"])
    attempt = service.create_attempt(session["id"], "initial", expected_revision=session["revision"])
    transcript = repo.put("transcripts", {"attempt_id": attempt["id"], "text": "🔋 A main है.\nमतलब मतलब B backup है.", "segments": [{"text": "🔋 A main है."}, {"text": "मतलब मतलब B backup है."}], "original": {"raw_segments": [{"text": " मतलब मतलब "}], "diagnostics": {"chunk_stitch_edits": []}}})
    return session, attempt, transcript


def test_empty_database_migrates_all_named_tables_and_indexes(tmp_path):
    repo = Repository(tmp_path / "gym")
    try:
        repo.initialize()
        assert set(TABLES) == EXPECTED
        with repo.engine.connect() as connection:
            tables = {row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
            assert tables == EXPECTED | {"alembic_version"}
            assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar() == SCHEMA_REVISION
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
            assert connection.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"
            assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar() == 10000
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
            for name in EXPECTED:
                assert connection.exec_driver_sql(f"PRAGMA index_list('{name}')").fetchall()
        assert list((repo.root / "backups").glob("*.sqlite3"))
    finally:
        repo.close()


def test_populated_database_restart_preserves_ids_unicode_and_raw_payload(tmp_path):
    root = tmp_path / "gym"
    repo = Repository(root)
    repo.initialize()
    try:
        session, attempt, original = sample(repo)
        expected = repo.get("transcripts", original["id"])
    finally:
        repo.close()
    reopened = Repository(root)
    reopened.initialize()
    try:
        restored = reopened.get_session(session["id"])
        assert restored["attempts"][0]["id"] == attempt["id"]
        assert restored["exercise"]["title"] == "दो बैटरी 🔋"
        assert reopened.get("transcripts", original["id"]) == expected
        assert restored["attempts"][0]["transcript"]["original"]["raw_segments"][0]["text"] == " मतलब मतलब "
        assert len(reopened.find("transcript_segments", transcript_id=original["id"])) == 2
        with reopened.engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close()


def test_unknown_newer_schema_is_rejected_without_downgrading(tmp_path):
    root = tmp_path / "gym"
    repo = Repository(root)
    repo.initialize()
    repo.close()
    with closing(sqlite3.connect(root / "gym.sqlite3")) as connection, connection:
        connection.execute("UPDATE alembic_version SET version_num='9999_newer'")
    reopened = Repository(root)
    with pytest.raises(GymError) as error:
        reopened.initialize()
    assert error.value.code == "unsupported_database_schema"
    with closing(sqlite3.connect(root / "gym.sqlite3")) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "9999_newer"


def test_failed_migration_restores_its_owned_backup_and_releases_lock(tmp_path, monkeypatch):
    root = tmp_path / "gym"
    real_upgrade = command.upgrade

    def broken(config, target):
        connection = config.attributes["connection"]
        connection.exec_driver_sql("CREATE TABLE migration_damage (id INTEGER PRIMARY KEY)")
        raise RuntimeError("Synthetic migration interruption")

    monkeypatch.setattr(command, "upgrade", broken)
    repo = Repository(root)
    with pytest.raises(GymError) as error:
        repo.initialize()
    assert error.value.code == "database_migration_failed"
    assert repo.engine is None
    backups = list((root / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    with closing(sqlite3.connect(root / "gym.sqlite3")) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='migration_damage'").fetchone() is None
    monkeypatch.setattr(command, "upgrade", real_upgrade)
    recovered = Repository(root)
    recovered.initialize()
    try:
        assert recovered.list_sessions() == []
    finally:
        recovered.close()
    for backup in backups:
        backup.unlink()
        assert not backup.exists()


def test_second_instance_is_refused_but_lock_is_reusable(tmp_path):
    first = Repository(tmp_path / "gym")
    second = Repository(tmp_path / "gym")
    first.initialize()
    try:
        with pytest.raises(GymError) as error:
            second.initialize()
        assert error.value.code == "instance_running"
    finally:
        first.close()
    second.initialize()
    second.close()


def test_restart_marks_jobs_interrupted_and_blocks_old_generation(tmp_path):
    repo = Repository(tmp_path / "gym")
    repo.initialize()
    session, attempt, transcript = sample(repo)
    repo.put("attempts", {"id": attempt["id"], "status": "transcribing"})
    job = repo.put("jobs", {"operation": "transcribe", "session_id": session["id"], "attempt_id": attempt["id"], "status": "running", "generation": 0})
    repo.close()
    repo.initialize()
    try:
        assert repo.get("jobs", job["id"])["status"] == "interrupted"
        assert repo.get("attempts", attempt["id"])["status"] == "interrupted"
        with pytest.raises(GymError) as error:
            repo.put("transcripts", {"attempt_id": attempt["id"], "job_id": job["id"], "generation": 0, "text": "A stale result"})
        assert error.value.code == "stale_generation"
        assert repo.get("transcripts", transcript["id"])["text"].startswith("🔋")
    finally:
        repo.close()


def test_immutable_original_enforced_below_repository(tmp_path):
    repo = Repository(tmp_path / "gym")
    repo.initialize()
    try:
        session, attempt, transcript = sample(repo)
        with pytest.raises(sqlite3.IntegrityError):
            with closing(sqlite3.connect(repo.database_path)) as connection, connection:
                connection.execute("UPDATE transcripts SET content_hash='tampered' WHERE id=?", (transcript["id"],))
        assert repo.get("transcripts", transcript["id"])["content_hash"] == transcript["content_hash"]
    finally:
        repo.close()


def test_unowned_nonempty_directory_is_not_adopted(tmp_path):
    root = tmp_path / "not-gym"
    root.mkdir()
    unrelated = root / "keep.txt"
    unrelated.write_text("User-owned", encoding="utf-8")
    with pytest.raises(GymError) as error:
        Repository(root).initialize()
    assert error.value.code == "unsafe_data_root"
    assert unrelated.read_text(encoding="utf-8") == "User-owned"
    assert not (root / ".communication-gym-owned").exists()


def test_path_links_cannot_escape_owned_root(tmp_path):
    repo = Repository(tmp_path / "gym")
    repo.initialize()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = repo.root / "data" / new_id()
    try:
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("This account cannot create a symbolic link.")
        with pytest.raises(GymError) as error:
            repo.path(f"data/{link.name}/audio.wav")
        assert error.value.code == "unsafe_path"
    finally:
        repo.close()


def test_migration_backup_is_immediately_removable(tmp_path):
    repo = Repository(tmp_path / "gym")
    repo.initialize()
    try:
        backups = list((repo.root / "backups").glob("*.sqlite3"))
        assert backups
        # Windows rejects this while either backup handle remains open.
        for backup in backups:
            backup.unlink()
            assert not backup.exists()
        assert repo.list_sessions() == []
    finally:
        repo.close()


@pytest.mark.asyncio
async def test_migration_handles_do_not_block_complete_storage_deletion(tmp_path):
    from communication_gym.application.privacy import PrivacyService

    repo = Repository(tmp_path / "gym")
    repo.initialize()
    sample(repo)
    # Exercise the populated-database backup path as well as the initial snapshot.
    repo._migrate(runner=lambda connection: None)
    closed = []
    deleted_credentials = []

    async def close_runtime():
        closed.append(True)

    runtime = SimpleNamespace(
        repo=repo,
        close=close_runtime,
        credentials=SimpleNamespace(delete=deleted_credentials.append),
    )
    try:
        result = await PrivacyService(runtime).delete_all("DELETE ALL")
        assert result["deleted"] is True
        assert closed == [True] and deleted_credentials == ["coach"]
        assert list(repo.root.iterdir()) == []
    finally:
        repo.close()
