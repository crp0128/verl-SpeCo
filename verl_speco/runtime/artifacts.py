# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid
published_files_UNSUPPORTED = {errno.EPERM, errno.EXDEV, errno.ENOSYS, errno.ENOTSUP}

def sync_directory(path: Path, *, allow_unsupported: bool = True) -> None:
    if os.name == 'nt':
        return
    fd = os.open(path, os.O_DIRECTORY)
    try:
        try:
            os.fsync(fd)
        except OSError as error:
            if not allow_unsupported or error.errno not in published_files_UNSUPPORTED | {errno.EINVAL}:
                raise
    finally:
        os.close(fd)

def publish_file(source: Path, destination: Path) -> None:
    """Publish payload first; its independent manifest authorizes consumption."""
    try:
        os.replace(source, destination)
    except OSError as error:
        if error.errno not in published_files_UNSUPPORTED:
            raise
        with source.open('rb') as reader, destination.open('wb') as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise RuntimeError('published payload checksum differs')
        source.unlink()

def publish_json(path: Path, payload: dict) -> Path:
    """Fallback commits immutable snapshots with a checksummed ready marker.

    No reader trusts partially written or orphan snapshots. A malformed ready
    marker from an interrupted write is ignored; an intact marker with a bad
    payload is rejected. Previous committed snapshots remain available.
    """
    data = (json.dumps(payload, sort_keys=True) + '\n').encode()
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix='.' + path.name, delete=False) as writer:
        temporary = Path(writer.name)
        writer.write(data)
        writer.flush()
        os.fsync(writer.fileno())
    try:
        os.replace(temporary, path)
        sync_directory(path.parent)
        return path
    except OSError as error:
        if error.errno not in published_files_UNSUPPORTED:
            raise
    snapshot = path.with_name(path.name + f'.snapshot-{time.time_ns():020d}-{uuid.uuid4().hex}.json')
    with snapshot.open('xb') as writer:
        writer.write(data)
        writer.flush()
        os.fsync(writer.fileno())
    marker = snapshot.with_suffix('.ready')
    with marker.open('x') as writer:
        json.dump({'file': snapshot.name, 'sha256': hashlib.sha256(data).hexdigest()}, writer)
        writer.write('\n')
        writer.flush()
        os.fsync(writer.fileno())
    temporary.unlink()
    return snapshot

def published_files_committed_snapshot_bytes(path: Path, *, max_bytes: int | None=None) -> tuple[Path, bytes] | None:
    for marker in sorted(path.parent.glob(path.name + '.snapshot-*.ready'), reverse=True):
        try:
            descriptor = json.loads(marker.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(descriptor, dict) or descriptor.get('file') != marker.with_suffix('.json').name or (not isinstance(descriptor.get('sha256'), str)) or (len(descriptor['sha256']) != 64):
            continue
        snapshot = marker.with_suffix('.json')
        data = published_files_read_json_bytes(snapshot, max_bytes=max_bytes)
        if hashlib.sha256(data).hexdigest() != descriptor['sha256']:
            raise ValueError(f'committed JSON snapshot checksum differs: {snapshot}')
        return (snapshot, data)
    return None

def published_files_read_json_bytes(path: Path, *, max_bytes: int | None=None) -> bytes:
    if max_bytes is None:
        return path.read_bytes()
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('published JSON byte budget must be positive')
    with path.open('rb') as reader:
        data = reader.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError('published JSON exceeds byte budget')
    return data

def published_files_committed_snapshot(path: Path, *, max_bytes: int | None=None) -> Path | None:
    committed = published_files_committed_snapshot_bytes(path, max_bytes=max_bytes)
    return committed[0] if committed is not None else None

def read_published_json(path: Path, *, max_bytes: int | None=None) -> dict:
    committed = published_files_committed_snapshot_bytes(path, max_bytes=max_bytes)
    data = committed[1] if committed is not None else published_files_read_json_bytes(path, max_bytes=max_bytes)
    return json.loads(data)

def published_json_paths(directory: Path, pattern: str) -> list[Path]:
    bases = set(directory.glob(pattern))
    for marker in directory.glob(pattern + '.snapshot-*.ready'):
        base = marker.with_name(marker.name.split('.snapshot-', 1)[0])
        if published_files_committed_snapshot(base) is not None:
            bases.add(base)
    return sorted(bases)


try:
    import fcntl
except ImportError:

    class _WindowsLocks:
        LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = (1, 2, 4, 8)
        F_ADD_SEALS, F_GET_SEALS = (1033, 1034)
        F_SEAL_SEAL, F_SEAL_SHRINK, F_SEAL_GROW, F_SEAL_WRITE = (1, 2, 4, 8)

        @staticmethod
        def flock(fd: int, operation: int) -> None:
            import ctypes
            from ctypes import wintypes
            import msvcrt

            class Overlapped(ctypes.Structure):
                _fields_ = [('internal', ctypes.c_size_t), ('internal_high', ctypes.c_size_t), ('offset', wintypes.DWORD), ('offset_high', wintypes.DWORD), ('event', wintypes.HANDLE)]
            api = ctypes.WinDLL('kernel32', use_last_error=True)
            handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
            overlap = Overlapped()
            if operation & _WindowsLocks.LOCK_UN:
                call = api.UnlockFileEx
                call.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
                ok = call(handle, 0, 1, 0, ctypes.byref(overlap))
            else:
                call = api.LockFileEx
                call.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
                flags = (2 if operation & _WindowsLocks.LOCK_EX else 0) | (1 if operation & _WindowsLocks.LOCK_NB else 0)
                ok = call(handle, flags, 0, 1, 0, ctypes.byref(overlap))
            if not ok:
                error = ctypes.get_last_error()
                if error == 33:
                    raise BlockingIOError(error, 'file lock is held')
                raise ctypes.WinError(error)

        @staticmethod
        def fcntl(*args):
            raise NotImplementedError('sealed memfd frames require Linux fcntl')
    fcntl = _WindowsLocks()

def require_memfd() -> None:
    if not hasattr(os, 'memfd_create'):
        raise RuntimeError('native sealed frame transport requires Linux memfd support')
