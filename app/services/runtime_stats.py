"""Catatan RAM proses tanpa dependency tambahan; tidak mencatat teks atau secret."""
import logging
import sys
import time

logger = logging.getLogger("uvicorn.error")


def memory_mib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            return {"rss_mib": round(counter.WorkingSetSize / 1048576, 2),
                    "peak_rss_mib": round(counter.PeakWorkingSetSize / 1048576, 2)}
    if sys.platform.startswith("linux"):
        from pathlib import Path
        try:
            fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
            return {"rss_mib": round(int(fields["VmRSS"].split()[0]) / 1024, 2),
                    "peak_rss_mib": round(int(fields["VmHWM"].split()[0]) / 1024, 2)}
        except (OSError, KeyError, ValueError):
            pass
    return {"rss_mib": None, "peak_rss_mib": None}


def log_stage(stage, started):
    logger.info("startup stage=%s elapsed_s=%.2f memory=%s", stage,
                time.perf_counter() - started, memory_mib())
