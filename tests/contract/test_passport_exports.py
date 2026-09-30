import subprocess
import sys

import pytest

from experience_hub.passports.errors import PassportError


@pytest.mark.parametrize("first", ["passports", "capture", "experiences"])
def test_import_order_is_independent(first: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import experience_hub.{first}; "
                "from experience_hub.passports import verify_passport_bytes; "
                "from experience_hub.capture.sanitization import scan_text_fields"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("invalid", 400),
        ("size_limit", 413),
        ("sensitive_content", 400),
        ("not_found", 404),
        ("decision_conflict", 409),
        ("equivalent_ambiguous", 409),
        ("restore_required", 409),
        ("derivation_unsupported", 400),
        ("provenance_limit", 400),
        ("file_invalid", 400),
        ("output_conflict", 409),
        ("publication_unsupported", 400),
    ],
)
def test_errors_are_fixed_and_input_free(code: str, status: int) -> None:
    error = PassportError(code)
    assert error.code == f"passport_{code}"
    assert error.status_code == status
    assert error.details == {}
