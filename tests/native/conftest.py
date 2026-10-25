"""Keep compiled test images stable between consecutive Windows process starts."""
from __future__ import annotations

import os
from pathlib import Path
import time

import pytest


@pytest.fixture(scope="module")
def native_executable_guard(request):
    """Exclude write opens until teardown; each harness invocation still runs once."""
    if os.name != "nt":
        return lambda path: None

    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL

    def protect(path: Path):
        # GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, OPEN_EXISTING.
        # A previous writer can briefly remain after compilation. Only acquiring
        # this fixture handle retries sharing violations, before any case runs.
        for attempt in range(20):
            handle = create(str(path), 0x80000000, 0x1 | 0x4, None, 3, 0, None)
            if handle != ctypes.c_void_p(-1).value:
                break
            error = ctypes.get_last_error()
            if error != 32 or attempt == 19:
                raise ctypes.WinError(error)
            time.sleep(0.1)

        def release():
            if not close(handle):
                raise ctypes.WinError(ctypes.get_last_error())

        request.addfinalizer(release)

    return protect
