"""Explicit same-process content verification, not a persistent stat cache.

Each immutable file is content-hashed on first use and again at scope exit.
Between those boundaries identity/ctime/mtime guards reject changes; mutable
manifests and outputs are never enrolled implicitly. This assumes the ordinary
trusted local writer/filesystem model, as the existing compiler session does.
"""
from __future__ import annotations
from contextvars import ContextVar
import hashlib
from pathlib import Path
import os
import stat
import time

_ACTIVE = ContextVar('frisk_integrity_session', default=None)
CONTRACT = 'frisk-content-verified-process-session-v1'


def _path(value):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink immutable input rejected: ' + str(path))
    if not path.is_file():
        raise ValueError('regular immutable input required: ' + str(path))
    return path


def _stamp(path):
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('regular immutable input required')
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink)


def _hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def active():
    return _ACTIVE.get()


def verified_sha(path, expected=None, *, force=False):
    session = active()
    if session is None:
        actual = _hash(_path(path))
        if expected is not None and actual != expected:
            raise ValueError('immutable input content differs: ' + str(path))
        return actual
    return session.sha(path, expected, force=force)


class VerificationSession:
    """A bounded verification scope; no cross-process/cache-hit assumptions."""
    def __init__(self):
        self.files = {}
        self.inventories = {}
        self.closed = False
        self.token = None
        self.content_hashes = 0
        self.content_bytes = 0
        self.reused_checks = 0
        self.guard_checks = 0
        self.started_ns = time.perf_counter_ns()

    def __enter__(self):
        if active() is not None or self.token is not None or self.closed:
            raise ValueError('fresh nonnested verification session required')
        self.token = _ACTIVE.set(self)
        return self

    def inventory(self, name, values, current):
        expected=tuple(sorted(values))
        if name in self.inventories:
            prior,_=self.inventories[name]
            if expected!=prior:
                raise ValueError('immutable input inventory changed: ' + name)
        else:
            self.inventories[name]=(expected,current)
        if tuple(sorted(current()))!=expected:
            raise ValueError('immutable input inventory changed during check: ' + name)

    def enrolled(self, path):
        return str(Path(path).absolute()) in self.files

    def _content(self, path):
        before = _stamp(path)
        value = _hash(path)
        after = _stamp(path)
        self.content_hashes += 1
        self.content_bytes += before[5]
        if before != after:
            raise ValueError('immutable input changed during content read: ' + str(path))
        return value, after

    def sha(self, value, expected=None, *, force=False):
        if self.closed:
            raise ValueError('verification session is closed')
        path = _path(value)
        key = str(path)
        if key not in self.files:
            digest, stamp = self._content(path)
            if expected is not None and digest != expected:
                raise ValueError('immutable input content differs: ' + key)
            self.files[key] = (digest, stamp)
        else:
            digest, stamp = self.files[key]
            self.guard_checks += 1
            if _stamp(path) != stamp:
                raise ValueError('immutable input changed inside verification session: ' + key)
            if expected is not None and digest != expected:
                raise ValueError('immutable input requested with a different digest: ' + key)
            if force:
                actual, current = self._content(path)
                if actual != digest or current != stamp:
                    raise ValueError('immutable input changed during forced verification: ' + key)
            else:
                self.reused_checks += 1
        return digest

    def finish(self):
        if self.closed:
            return
        try:
            for name,(expected,current) in self.inventories.items():
                if tuple(sorted(current()))!=expected:
                    raise ValueError('immutable input inventory changed before exit: ' + name)
            # Re-read the contents, even when no filesystem witness changed.
            for name, (expected, stamp) in self.files.items():
                path = _path(name)
                if _stamp(path) != stamp:
                    raise ValueError('immutable input changed before session exit: ' + name)
                actual, current = self._content(path)
                if actual != expected or current != stamp:
                    raise ValueError('immutable input content changed before session exit: ' + name)
        finally:
            self.closed = True

    def __exit__(self, kind, value, traceback):
        try:
            self.finish()
        finally:
            if self.token is not None:
                _ACTIVE.reset(self.token)
                self.token = None

    def summary(self):
        return dict(contract=CONTRACT, files=len(self.files), inventories=len(self.inventories), closed=self.closed,
            full_content_hashes=self.content_hashes, full_content_bytes=self.content_bytes,
            reused_content_checks=self.reused_checks, identity_guards=self.guard_checks,
            wall_ns=time.perf_counter_ns()-self.started_ns,
            content_verification_boundary='first use and full scope exit; changed metadata fails closed',
            persistent_stat_cache=False)
