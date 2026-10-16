"""Synthetic, model-free privacy and lifecycle regressions using isolated storage."""
import asyncio
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from communication_gym.application.jobs import JobContext
from communication_gym.application.privacy import PrivacyService
from communication_gym.application.runtime import Runtime
from communication_gym.domain.models import GymError, new_id
from communication_gym.providers.processes import OwnedProcess
from communication_gym.settings import CredentialStore, Settings
from communication_gym.storage.repository import Repository


@pytest.fixture
def local_runtime(tmp_path):
    repo = Repository(tmp_path / 'gym')
    repo.initialize()
    runtime = Runtime(repo, Settings(), CredentialStore(isolated=True))
    yield runtime
    repo.close()


def session_for(runtime):
    return runtime.sessions.create({'language': 'hinglish', 'mode': 'compare', 'exercise': {'source': 'custom', 'title': 'Synthetic batteries', 'prompt': 'A और B explain करो.', 'audience': 'A new listener'}})


def evidence_for(runtime, session):
    repo = runtime.repo
    session = runtime.sessions.prep(session['id'], [], session['revision'])
    attempt = runtime.sessions.create_attempt(session['id'], 'initial', expected_revision=session['revision'])
    rid, tid = new_id(), new_id()
    directory = f'data/sessions/{session["id"]}'
    source = repo.path(f'{directory}/recordings/{rid}/source.bin')
    source.parent.mkdir(parents=True)
    source.write_bytes(b'synthetic original audio')
    prepared = source.with_name('prepared.wav')
    prepared.write_bytes(b'synthetic prepared audio')
    recording = repo.put('audio_recordings', {'id': rid, 'attempt_id': attempt['id'], 'relative_path': source.relative_to(repo.root).as_posix(), 'prepared_path': prepared.relative_to(repo.root).as_posix(), 'status': 'ready', 'media_type': 'audio/wav'})
    repo.put('attempts', {**attempt, 'recording_id': rid})
    original = repo.path(f'{directory}/transcripts/{tid}.json')
    original.parent.mkdir(parents=True)
    raw = {'text': 'A main है. मतलब मतलब B backup है.', 'raw_segments': [{'text': ' मतलब मतलब '}], 'source': 'C:/private/synthetic.wav', 'config': {'Authorization': 'synthetic-secret', 'repo_path': 'C:/private/source', 'candidate': 'qwen3'}}
    original.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    transcript = repo.put('transcripts', {'id': tid, 'attempt_id': attempt['id'], 'text': raw['text'], 'segments': [{'text': raw['text']}], 'original_json_path': original.relative_to(repo.root).as_posix()})
    return attempt, recording, transcript, original


def test_export_keeps_original_evidence_and_safe_versioned_names(local_runtime):
    runtime = local_runtime
    session = session_for(runtime)
    attempt, recording, transcript, original = evidence_for(runtime, session)
    corrected = runtime.repo.put('transcripts', {'attempt_id': attempt['id'], 'kind': 'human_corrected', 'parent_id': transcript['id'], 'text': 'A main है. B backup है.'})
    before = runtime.repo.get_session(session['id'])
    original_bytes = original.read_bytes()
    privacy = PrivacyService(runtime)
    exported = privacy.export(session['id'], include_audio=True)
    with zipfile.ZipFile(runtime.repo.path(exported['relative_path'])) as archive:
        names = archive.namelist()
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts and ':' not in name and '\\' not in name for name in names)
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['schema_version'] == 1 and manifest['session_id'] == session['id']
        raw = json.loads(archive.read(f'evidence/{transcript["id"]}/asr-original.json'))
        assert raw['raw_segments'] == [{'text': ' मतलब मतलब '}]
        assert raw['config'] == {'candidate': 'qwen3'}
        assert 'source' not in raw
        exported_session = json.loads(archive.read('session.json'))
        assert exported_session['exercise']['source'] == 'custom'
        assert exported_session['attempts'][0]['transcripts'][0]['text'] == transcript['text']
        assert {item['id']: item['text'] for item in exported_session['attempts'][0]['transcripts']} == {transcript['id']: transcript['text'], corrected['id']: corrected['text']}
        assert archive.read(f'audio/{recording["id"]}/original.bin') == b'synthetic original audio'
        assert archive.read(f'audio/{recording["id"]}/prepared.wav') == b'synthetic prepared audio'
        for item in manifest['files']:
            if 'sha256' in item:
                assert hashlib.sha256(archive.read(item['path'])).hexdigest() == item['sha256']
        assert b'C:/private' not in archive.read('session.json') + archive.read(f'evidence/{transcript["id"]}/asr-original.json')
    assert runtime.repo.get_session(session['id']) == before
    assert original.read_bytes() == original_bytes
    text_only = privacy.export(session['id'])
    with zipfile.ZipFile(runtime.repo.path(text_only['relative_path'])) as archive:
        assert not any(name.startswith('audio/') for name in archive.namelist())


@pytest.mark.asyncio
async def test_audio_deletion_retains_transcripts_and_removes_internal_exports(local_runtime):
    runtime = local_runtime
    session = session_for(runtime)
    attempt, recording, transcript, original = evidence_for(runtime, session)
    privacy = PrivacyService(runtime)
    exported = privacy.export(session['id'], include_audio=True)
    checkpoint = runtime.repo.path(f'data/sessions/{session["id"]}/checkpoints/{new_id()}/audio.wav')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic checkpoint audio')
    result = await privacy.delete_audio(session['id'])
    assert result['audio_deleted'] and result['metrics_retained']
    assert runtime.repo.get('transcripts', transcript['id']) == transcript
    assert original.is_file()
    assert runtime.repo.get('audio_recordings', recording['id'])['status'] == 'deleted'
    assert not runtime.repo.path(recording['relative_path']).exists()
    assert not runtime.repo.path(exported['relative_path']).exists()
    assert not checkpoint.exists()


@pytest.mark.asyncio
async def test_audio_deletion_does_not_purge_a_transcript_sibling(local_runtime):
    runtime = local_runtime
    session = session_for(runtime)
    attempt, recording, transcript, original = evidence_for(runtime, session)
    loose = runtime.repo.path(f'data/sessions/{session["id"]}/{new_id()}.wav')
    loose.write_bytes(b'synthetic loose audio')
    runtime.repo.put('audio_recordings', {'session_id': session['id'], 'relative_path': loose.relative_to(runtime.repo.root).as_posix(), 'status': 'ready'})
    assert (await PrivacyService(runtime).delete_audio(session['id']))['audio_deleted']
    assert original.is_file() and not loose.exists()


@pytest.mark.asyncio
async def test_locked_session_purge_resumes_after_repository_restart(local_runtime, monkeypatch):
    runtime = local_runtime
    session = session_for(runtime)
    evidence_for(runtime, session)
    other = session_for(runtime)
    privacy = PrivacyService(runtime)
    target = f'data/sessions/{session["id"]}'
    real_purge = privacy._purge
    def locked(path):
        if path == runtime.repo.path(target):
            raise PermissionError('Synthetic lock')
        return real_purge(path)
    monkeypatch.setattr(privacy, '_purge', locked)
    result = await privacy.delete_session(session['id'])
    assert not result['deleted'] and result['pending_deletions']
    runtime.repo.close()
    runtime.repo.initialize()
    assert PrivacyService(runtime).drain() == []
    assert not runtime.repo.path(target).exists()
    assert runtime.repo.get_session(other['id'])['id'] == other['id']


@pytest.mark.asyncio
async def test_session_delete_removes_only_its_internal_exports(local_runtime):
    runtime = local_runtime
    first, other = session_for(runtime), session_for(runtime)
    privacy = PrivacyService(runtime)
    first_export = privacy.export(first['id'])
    other_export = privacy.export(other['id'])
    assert (await privacy.delete_session(first['id']))['deleted']
    assert not runtime.repo.path(first_export['relative_path']).exists()
    assert runtime.repo.path(other_export['relative_path']).is_file()


@pytest.mark.asyncio
async def test_full_delete_keeps_intent_on_lock_and_resumes_on_startup(local_runtime, monkeypatch):
    runtime = local_runtime
    session_for(runtime)
    runtime.credentials.set('coach', 'synthetic-secret')
    locked = runtime.repo.root / 'settings.json'
    locked.parent.mkdir(exist_ok=True)
    locked.write_text('Synthetic log', encoding='utf-8')
    real_unlink = Path.unlink
    def deny_locked(path, *args, **kwargs):
        if path == locked:
            raise PermissionError('Synthetic lock')
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', deny_locked)
    with pytest.raises(GymError) as error:
        await PrivacyService(runtime).delete_all('DELETE ALL')
    assert error.value.code == 'deletion_pending'
    assert (runtime.repo.root / '.communication-gym-owned').is_file()
    assert (runtime.repo.root / '.delete-all-pending').is_file()
    assert runtime.credentials.get('coach') is None
    runtime.repo.close()
    monkeypatch.setattr(Path, 'unlink', real_unlink)
    runtime.repo.initialize()
    assert runtime.repo.list_sessions() == []
    assert not locked.exists()
    assert not (runtime.repo.root / '.delete-all-pending').exists()


def test_artifact_paths_cannot_claim_another_session(local_runtime):
    runtime = local_runtime
    first, other = session_for(runtime), session_for(runtime)
    with pytest.raises(GymError) as error:
        runtime.repo.put('audio_recordings', {'session_id': first['id'], 'relative_path': f'data/sessions/{other["id"]}/recordings/{new_id()}/source.wav'})
    assert error.value.code == 'ownership_conflict'


@pytest.mark.asyncio
async def test_job_generation_change_blocks_a_late_context_commit(local_runtime):
    runtime = local_runtime
    session = session_for(runtime)
    job = runtime.repo.put('jobs', {'session_id': session['id'], 'operation': 'synthetic', 'status': 'running', 'generation': 0})
    context = JobContext(runtime.jobs, job)
    runtime.repo.put('jobs', {**job, 'generation': 1})
    with pytest.raises(asyncio.CancelledError):
        context.commit(lambda: runtime.repo.update_session(session['id'], {'prep_bullets': ['late']}))
    assert runtime.repo.get_session(session['id'])['prep_bullets'] == []


@pytest.mark.asyncio
async def test_job_shutdown_refuses_new_work(local_runtime):
    runtime = local_runtime
    await runtime.jobs.close()
    async def work(context):
        raise AssertionError('A closed manager must not start work')
    with pytest.raises(GymError) as error:
        runtime.jobs.start('synthetic', work)
    assert error.value.code == 'runtime_closing'


@pytest.mark.asyncio
async def test_cancelled_process_close_reaps_only_owned_process():
    owned = await OwnedProcess.start([sys.executable, '-u', '-c', 'import time;time.sleep(120)'])
    sentinel = await OwnedProcess.start([sys.executable, '-u', '-c', 'import time;time.sleep(120)'])
    try:
        closing = asyncio.create_task(owned.close(grace_seconds=120))
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert owned.returncode is not None
        assert sentinel.returncode is None
    finally:
        await owned.terminate()
        await sentinel.terminate()


@pytest.mark.asyncio
async def test_full_delete_rejects_nested_links_before_removing_any_data(local_runtime, tmp_path):
    runtime = local_runtime
    session = session_for(runtime)
    outside = tmp_path / 'shared'
    outside.mkdir()
    sentinel = outside / 'model.bin'
    sentinel.write_bytes(b'shared synthetic model')
    link = runtime.repo.path(f'data/sessions/{session["id"]}/nested')
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('This account cannot create a symbolic link.')
    try:
        with pytest.raises(GymError) as error:
            await PrivacyService(runtime).delete_all('DELETE ALL')
        assert error.value.code == 'unsafe_path'
        assert sentinel.read_bytes() == b'shared synthetic model'
        assert runtime.repo.get_session(session['id'])['id'] == session['id']
        assert not (runtime.repo.root / '.delete-all-pending').exists()
    finally:
        link.unlink()
