"""Explicit test doubles; production provider selection never imports this module."""
import asyncio
import copy
from collections import deque


class FakeCoach:
    def __init__(self, responses=()):
        self.responses = deque(copy.deepcopy(list(responses)))
        self.requests = []
        self.release_count = 0
        self.closed = False

    async def capabilities(self):
        return {"available": True, "provider_id": "test-only", "model_id": "fixture", "locality": "local", "structured_output": True}

    async def complete(self, task, system, payload, schema):
        self.requests.append(copy.deepcopy({"task": task, "system": system, "payload": payload, "schema": schema}))
        if not self.responses:
            raise AssertionError("No fixture response was supplied for this call")
        result = self.responses.popleft()
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)

    async def release(self):
        self.release_count += 1

    async def close(self):
        self.closed = True


class ControlledSpeech:
    def __init__(self, results=()):
        self.results = deque(copy.deepcopy(list(results)))
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()
        self.cancelled = False
        self.release_count = 0
        self.closed = False
        self.calls = []

    async def capabilities(self):
        return {"available": True, "provider_id": "test-only", "locality": "local", "word_timing": "none"}

    async def transcribe(self, source, language, checkpoint_dir, progress=None):
        self.calls.append({"source": source, "language": language, "checkpoint_dir": checkpoint_dir})
        self.started.set()
        await self.proceed.wait()
        if self.cancelled:
            raise asyncio.CancelledError
        if not self.results:
            raise AssertionError("No transcription fixture was supplied")
        return copy.deepcopy(self.results.popleft())

    async def cancel(self):
        self.cancelled = True
        self.proceed.set()

    async def release(self):
        self.release_count += 1

    async def close(self):
        self.closed = True
        await self.cancel()
