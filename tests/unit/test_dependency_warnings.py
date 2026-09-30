import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["retrieval.tokenizer", "inspiration.operators"])
def test_known_cold_dependency_warnings_do_not_change_global_warning_policy(
    module: str, tmp_path: Path
) -> None:
    program = (
        "import warnings\n"
        "before = list(warnings.filters)\n"
        f"import experience_hub.{module}\n"
        "assert warnings.filters == before\n"
        "with warnings.catch_warnings(record=True) as seen:\n"
        " warnings.simplefilter('always')\n"
        " warnings.warn_explicit('invalid escape sequence synthetic', "
        "SyntaxWarning, 'unrelated.py', 1, module='unrelated')\n"
        " assert len(seen) == 1\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONPYCACHEPREFIX": str(tmp_path / "cold")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
