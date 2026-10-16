"""Contain only processes launched by Gym; never discover or kill by image name."""
from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import subprocess
from pathlib import Path
from typing import Sequence


class _WindowsJob:
    def __init__(self) -> None:
        from ctypes import wintypes

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, pid: int) -> None:
        from ctypes import wintypes

        process = self.kernel.OpenProcess(0x0100 | 0x0001 | 0x0800, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
            # CREATE_SUSPENDED ensures no child can escape before job assignment.
            class THREADENTRY32(ctypes.Structure):
                _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                            ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                            ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
                            ("dwFlags", wintypes.DWORD)]
            self.kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
            self.kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            self.kernel.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
            self.kernel.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
            self.kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.kernel.OpenThread.restype = wintypes.HANDLE
            self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
            self.kernel.ResumeThread.restype = wintypes.DWORD
            snapshot = self.kernel.CreateToolhelp32Snapshot(0x00000004, 0)
            if snapshot == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            resumed = False
            try:
                entry = THREADENTRY32()
                entry.dwSize = ctypes.sizeof(entry)
                present = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
                while present:
                    if entry.th32OwnerProcessID == pid:
                        thread = self.kernel.OpenThread(0x0002, False, entry.th32ThreadID)
                        if not thread:
                            raise ctypes.WinError(ctypes.get_last_error())
                        try:
                            if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
                                raise ctypes.WinError(ctypes.get_last_error())
                            resumed = True
                        finally:
                            self.kernel.CloseHandle(thread)
                    present = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            finally:
                self.kernel.CloseHandle(snapshot)
            if not resumed:
                raise OSError("Owned process primary thread was not found")
        finally:
            self.kernel.CloseHandle(process)

    def contains(self, pid: int) -> bool:
        from ctypes import wintypes

        if not self.handle:
            return False
        process = self.kernel.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        try:
            self.kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
            self.kernel.IsProcessInJob.restype = wintypes.BOOL
            member = wintypes.BOOL()
            if not self.kernel.IsProcessInJob(process, self.handle, ctypes.byref(member)):
                raise ctypes.WinError(ctypes.get_last_error())
            return bool(member.value)
        finally:
            self.kernel.CloseHandle(process)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class OwnedProcess:
    """Async subprocess with race-free Windows kill-on-close tree ownership."""

    def __init__(self, process: asyncio.subprocess.Process, job: _WindowsJob | None) -> None:
        self.process = process
        self._job = job

    @classmethod
    async def start(cls, argv: Sequence[str], *, env: dict[str, str] | None = None,
                    cwd: Path | None = None, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE) -> OwnedProcess:
        job = _WindowsJob() if os.name == "nt" else None
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004}
                   if os.name == "nt" else {"start_new_session": True})
        process = None
        try:
            process = await asyncio.create_subprocess_exec(*argv, env=env, cwd=cwd, stdin=stdin,
                                                           stdout=stdout, stderr=stderr, **options)
            if job:
                job.assign_and_resume(process.pid)
            return cls(process, job)
        except BaseException:
            if job:
                job.close()
            if process is not None:
                if process.returncode is None:
                    process.kill()
                await process.wait()
            raise

    @property
    def pid(self) -> int:
        """PID of the launched executable, which can be a Windows venv redirector."""
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def owns_pid(self, pid: int) -> bool:
        """Check tree ownership without conflating a launcher with its actual worker."""
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        if self._job:
            return self._job.contains(pid)
        if os.name == "nt":
            return False
        try:
            return os.getpgid(pid) == self.pid
        except ProcessLookupError:
            return False

    async def terminate(self) -> None:
        if self._job:
            self._job.close()
            self._job = None
        elif os.name != "nt":
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif self.process.returncode is None:
            self.process.kill()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except TimeoutError:
            if self.process.returncode is None:
                self.process.kill()
            await self.process.wait()

    async def close(self, grace_seconds: float = 8) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(self.process.wait()), timeout=grace_seconds)
        except asyncio.CancelledError:
            await self.terminate()
            raise
        except TimeoutError:
            await self.terminate()
        finally:
            # Also reap children that survived a successfully exited parent.
            if self._job:
                self._job.close()
                self._job = None
            elif os.name != "nt":
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
