"""Synthetic framed processes only: no model loading, network or personal audio."""
import asyncio
import io
import json
import os
from pathlib import Path
import struct
import sys

import pytest

from communication_gym.providers.processes import OwnedProcess
from communication_gym.worker.protocol import MAX_FRAME_BYTES, ProtocolError, decode_payload, encode_frame, read_frame, read_frame_async


def message(op="result", payload=None):
    return {"version": 1, "request_id": "synthetic", "op": op, "payload": payload or {}}


def assert_process_exited(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
        if not handle:
            assert ctypes.get_last_error() == 87  # PID no longer exists.
            return
        try:
            assert kernel.WaitForSingleObject(handle, 5000) == 0
            code = wintypes.DWORD()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            assert code.value != 259
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        # An exited descendant may briefly remain a reparented zombie.
        stat = Path(f"/proc/{pid}/stat")
        assert stat.exists() and stat.read_text().split()[2] == "Z"


class SplitReader(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 2))


def test_split_frames_are_read_incrementally():
    original = message(payload={"text": "🙂 हाँ um um"})
    stream = SplitReader(encode_frame(original))
    assert read_frame(stream) == original
    assert read_frame(stream) is None


@pytest.mark.parametrize("data", [struct.pack("!I", 0), struct.pack("!I", MAX_FRAME_BYTES + 1),
    struct.pack("!I", 3) + b"{}", struct.pack("!I", 3) + b"???", struct.pack("!I", 1) + b"\xff"])
def test_invalid_or_truncated_frames_fail_closed(data):
    with pytest.raises(ProtocolError):
        read_frame(io.BytesIO(data))


@pytest.mark.parametrize("value", [{"version": 2}, {"version": 1}, [], {"version": 1, "request_id": "x", "op": "result", "payload": []}])
def test_protocol_rejects_incompatible_shape(value):
    with pytest.raises(ProtocolError):
        decode_payload(json.dumps(value).encode())


def test_protocol_rejects_nonfinite_json():
    with pytest.raises(ProtocolError):
        decode_payload(b'{"version":1,"request_id":"x","op":"result","payload":{"x":NaN}}')


@pytest.mark.asyncio
async def test_async_split_frame_and_crash():
    stream = asyncio.StreamReader()
    packet = encode_frame(message())
    task = asyncio.create_task(read_frame_async(stream))
    for byte in packet:
        stream.feed_data(bytes([byte]))
        await asyncio.sleep(0)
    assert await task == message()
    stream.feed_eof()
    with pytest.raises(asyncio.IncompleteReadError):
        await read_frame_async(stream)


@pytest.mark.asyncio
async def test_owned_process_shutdown_does_not_touch_unrelated_process(tmp_path):
    # A contained child publishes its own PID; unrelated sentinel is independently owned.
    child_script = "import time; time.sleep(120)"
    parent_script = ("import subprocess,sys,time; "
                     f"p=subprocess.Popen([sys.executable,'-c',{child_script!r}]); "
                     "print(p.pid,flush=True); time.sleep(120)")
    sentinel = await OwnedProcess.start([sys.executable, "-u", "-c", child_script])
    owned = await OwnedProcess.start([sys.executable, "-u", "-c", parent_script])
    child_pid = int(await asyncio.wait_for(owned.process.stdout.readline(), 10))
    try:
        assert owned.owns_pid(owned.pid)
        assert owned.owns_pid(child_pid)
        assert not owned.owns_pid(sentinel.pid)
        await owned.close(grace_seconds=0.05)
        assert owned.returncode is not None
        assert sentinel.returncode is None
        assert_process_exited(child_pid)
    finally:
        await owned.terminate()
        await sentinel.terminate()


@pytest.mark.asyncio
async def test_model_free_worker_is_persistent_and_release_is_idempotent(tmp_path):
    worker = Path(__file__).parents[1] / "src" / "communication_gym" / "worker" / "main.py"
    owned = await OwnedProcess.start([sys.executable, "-B", "-u", str(worker), "--repo", str(tmp_path), "--root", str(tmp_path)])
    try:
        ready = await asyncio.wait_for(read_frame_async(owned.process.stdout), 10)
        assert ready["op"] == "ready"
        pid = ready["payload"]["pid"]
        assert owned.owns_pid(owned.pid)
        assert owned.owns_pid(pid)
        for index, operation in enumerate(("capabilities", "release", "release")):
            request = {"version": 1, "request_id": str(index), "op": operation, "payload": {}}
            owned.process.stdin.write(encode_frame(request))
            await owned.process.stdin.drain()
            response = await asyncio.wait_for(read_frame_async(owned.process.stdout), 10)
            assert response["op"] == "result"
            if operation == "capabilities":
                assert response["payload"]["inference_verified"] is False
                assert response["payload"]["pid"] == pid
                assert owned.owns_pid(pid)
        owned.process.stdin.close()
        await owned.close(grace_seconds=5)
        assert owned.returncode == 0
        assert_process_exited(pid)
        assert_process_exited(owned.pid)
    finally:
        await owned.terminate()


@pytest.mark.asyncio
async def test_worker_rejects_invalid_frame_and_exits_without_model(tmp_path):
    worker = Path(__file__).parents[1] / "src" / "communication_gym" / "worker" / "main.py"
    owned = await OwnedProcess.start([sys.executable, "-B", "-u", str(worker), "--repo", str(tmp_path), "--root", str(tmp_path)])
    try:
        await asyncio.wait_for(read_frame_async(owned.process.stdout), 10)
        owned.process.stdin.write(struct.pack("!I", MAX_FRAME_BYTES + 1))
        await owned.process.stdin.drain()
        await owned.close(grace_seconds=5)
        assert owned.returncode == 0
    finally:
        await owned.terminate()


@pytest.fixture
def synthetic_provider(tmp_path, monkeypatch):
    from communication_gym.providers.stt import SpeechProvider
    provider = SpeechProvider({}, tmp_path)
    fixture = Path(__file__).parent / "fakes" / "framed_worker.py"

    async def start():
        if provider.pid:
            return
        await provider._dispose()
        provider._owned = await OwnedProcess.start([sys.executable, "-B", "-u", str(fixture)])
        ready = await asyncio.wait_for(read_frame_async(provider._owned.process.stdout), 10)
        assert ready["op"] == "ready"
        assert provider._owned.owns_pid(ready["payload"]["pid"])
        provider._worker_pid = ready["payload"]["pid"]
        provider._reader = asyncio.create_task(provider._read_responses(provider._owned))
        provider._stderr = asyncio.create_task(provider._drain_stderr(provider._owned))

    monkeypatch.setattr(provider, "_ensure_started", start)
    return provider


@pytest.mark.asyncio
async def test_cancel_is_responsive_and_retains_worker_for_next_operation(synthetic_provider):
    from communication_gym.domain.models import GymError
    provider = synthetic_provider
    operation = asyncio.create_task(provider._rpc("transcribe", {}))
    while provider._active is None:
        await asyncio.sleep(0.005)
    pid = provider.pid
    try:
        await provider.cancel()
        with pytest.raises(GymError) as caught:
            await operation
        assert caught.value.code == "cancelled"
        assert provider.pid == pid
        assert (await provider._rpc("release", {}))["released"] is True
    finally:
        await provider._dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["progress_forever", "silence"])
async def test_deadline_is_not_extended_by_progress(synthetic_provider, operation):
    from communication_gym.domain.models import GymError
    provider = synthetic_provider
    try:
        with pytest.raises(GymError) as caught:
            await provider._rpc(operation, {}, timeout=0.05)
        assert caught.value.code == "speech_timeout"
        assert (await provider._rpc("release", {}))["released"]
    finally:
        await provider._dispose()


@pytest.mark.asyncio
async def test_crash_fails_pending_request_and_next_operation_respawns(synthetic_provider):
    from communication_gym.domain.models import GymError
    provider = synthetic_provider
    try:
        with pytest.raises(GymError) as caught:
            await provider._rpc("crash", {}, timeout=5)
        assert caught.value.code == "speech_worker_stopped"
        assert (await provider._rpc("release", {}))["released"]
    finally:
        await provider._dispose()


@pytest.mark.asyncio
async def test_uncooperative_cancel_reaps_owned_worker(synthetic_provider):
    from communication_gym.domain.models import GymError
    provider = synthetic_provider
    operation = asyncio.create_task(provider._rpc("ignore_cancel", {}))
    while provider._active is None:
        await asyncio.sleep(0.005)
    try:
        await provider.cancel()
        with pytest.raises(GymError):
            await operation
        assert provider.pid is None
    finally:
        await provider._dispose()


@pytest.mark.asyncio
async def test_provider_reports_actual_worker_and_reaps_worker_and_launcher(tmp_path):
    from communication_gym.providers.stt import SpeechProvider
    repo = tmp_path / "repo"
    package = repo / "voice_to_text"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    root = tmp_path / "data"
    root.mkdir()
    provider = SpeechProvider({"python_path": sys.executable, "repo_path": str(repo)}, root)
    sentinel = await OwnedProcess.start([sys.executable, "-u", "-c", "import time; time.sleep(120)"])
    try:
        first = await provider.capabilities()
        worker_pid = first["pid"]
        launcher = provider._owned
        assert provider.pid == worker_pid
        assert launcher.pid == launcher.process.pid
        assert launcher.owns_pid(worker_pid) and launcher.owns_pid(launcher.pid)
        assert not launcher.owns_pid(sentinel.pid)
        for _ in range(2):
            await provider.release()
            capabilities = await provider.capabilities()
            assert capabilities["pid"] == provider.pid == worker_pid
            assert provider._owned is launcher
        await provider.close()
        assert provider.pid is None
        assert launcher.returncode is not None
        assert_process_exited(worker_pid)
        assert_process_exited(launcher.pid)
        assert sentinel.returncode is None
    finally:
        await provider.close()
        await sentinel.terminate()
