from __future__ import annotations

import subprocess
import sys


def test_fastapi_testclient_does_not_use_deprecated_httpx_fallback() -> None:
    script = "\n".join(
        (
            "import warnings",
            "warnings.simplefilter('error')",
            "import fastapi.testclient",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
