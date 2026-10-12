# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from contextlib import contextmanager
from threading import local, get_ident, Lock
import torch
from verl_speco.transport.native_transfer import AscendFeatureTransport

class SourcePublicationTransport(AscendFeatureTransport):

    def __init__(self, **kwargs):
        self._publication_local = local()
        self._publication_owner = None
        self._publication_lock = Lock()
        super().__init__(**kwargs)

    @contextmanager
    def publication_scope(self):
        if self.device.type != 'npu':
            raise ValueError('source publication stream requires the actual NPU transport')
        if not self._publication_lock.acquire(blocking=False):
            raise RuntimeError('source publication stream is already owned')
        try:
            owner = get_ident()
            if self._publication_owner is not None and self._publication_owner != owner:
                raise RuntimeError('source publication stream changed its owning thread')
            self._publication_owner = owner
            if getattr(self._publication_local, 'active', None) is not None:
                raise RuntimeError('source publication scopes cannot nest')
            stream = getattr(self._publication_local, 'stream', None)
            if stream is None:
                stream = torch.npu.Stream(device=self.device)
                self._publication_local.stream = stream
            with torch.npu.stream(stream):
                self._publication_local.active = stream
                try:
                    yield
                finally:
                    try:
                        stream.synchronize()
                    finally:
                        self._publication_local.active = None
        finally:
            self._publication_lock.release()

    def _synchronize(self):
        stream = getattr(self._publication_local, 'active', None)
        if stream is None:
            return super()._synchronize()
        stream.synchronize()
