from __future__ import annotations

import os
from pathlib import Path


def test_numba_cache_dir_is_configured_and_writable():
    cache_dir = Path(os.environ["NUMBA_CACHE_DIR"])
    probe = cache_dir / f"pytest-write-probe-{os.getpid()}"

    try:
        probe.write_text("writable", encoding="utf-8")
        assert probe.read_text(encoding="utf-8") == "writable"
    finally:
        probe.unlink(missing_ok=True)
