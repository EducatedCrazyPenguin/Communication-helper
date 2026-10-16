"""One owned expensive workload at a time; never arbitrate other applications."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager

class ResourceCoordinator:
    def __init__(self, speech=None, coach=None, tts=None):
        self.speech=speech
        self.coach=coach
        self.tts=tts
        self._lock=asyncio.Lock()
        self.active: str | None=None
        self.previous: str | None=None

    @asynccontextmanager
    async def lease(self, workload: str):
        async with self._lock:
            # Always release incompatible residency, even after failed work.
            if workload!='stt' and self.speech is not None:
                await self.speech.release()
            if workload!='coach' and self.coach is not None:
                await self.coach.release()
            if workload!='tts' and self.tts is not None:
                await self.tts.release()
            self.active=workload
            try:
                yield
            finally:
                self.previous=workload
                self.active=None

    async def close(self):
        for provider in (self.speech,self.coach,self.tts):
            if provider is not None:
                await provider.close()
