#!/usr/bin/env python3
"""
SIGHUP-immune launcher for torch.distributed.run (torchrun).

Why this exists:
PyTorch's `torch.distributed.elastic` explicitly registers its own SIGHUP signal handler,
overriding `nohup`'s SIG_IGN. When an SSH session disconnects after an idle timeout,
the kernel TTY hangup sends SIGHUP (signal 1) and kills standard `nohup torchrun`.
This wrapper locks SIGHUP to SIG_IGN so SSH disconnects never interrupt multi-hour runs.
"""

import signal
import sys

if hasattr(signal, "SIGHUP"):
    _orig_signal = signal.signal
    _orig_signal(signal.SIGHUP, signal.SIG_IGN)

    def _sighup_safe_signal(sig, handler):
        if sig == signal.SIGHUP:
            return signal.SIG_IGN
        return _orig_signal(sig, handler)

    signal.signal = _sighup_safe_signal

from torch.distributed.run import main

if __name__ == "__main__":
    sys.exit(main())
