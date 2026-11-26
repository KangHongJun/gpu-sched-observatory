"""Hold back SIGINT/SIGTERM while an MPS session stops its daemon."""

from __future__ import annotations

import signal
import warnings

from gso.bench.mps import UnsafeGpuError


class GuardedSession:
    """Wraps a session so that SIGINT/SIGTERM are held back while its ``__exit__`` stops the MPS
    daemon (a half-finished stop could leave the daemon running); a held signal is honoured
    right after. A leftover-daemon RuntimeWarning/RuntimeError is passed to ``on_warning``."""

    def __init__(self, cm, on_warning):
        self.cm, self.on_warning = cm, on_warning

    def __enter__(self):
        return self.cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        pending, saved = [], {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved[sig] = signal.signal(sig, lambda *_: pending.append(True))
        except ValueError:  # not the main thread: cannot defer
            saved = {}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    self.cm.__exit__(exc_type, exc, tb)
                except RuntimeError as e:
                    self.on_warning(f"MPS session did not stop cleanly: {e}")
                    if exc_type is None:
                        raise UnsafeGpuError(
                            f"MPS session did not stop cleanly: {e}"
                        ) from e
            for w in caught:
                self.on_warning(f"MPS session exit: {w.message}")
        finally:
            for sig, old in saved.items():
                signal.signal(sig, old)
        if pending and exc_type is None:
            raise KeyboardInterrupt
        return False
