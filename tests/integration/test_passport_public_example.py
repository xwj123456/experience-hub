from pathlib import Path

from tests.passport_example_fixtures import capture_example

from experience_hub.passports import verify_passport_bytes

EXAMPLE = Path(__file__).resolve().parents[2] / (
    "examples/passports/capture-recovery.passport.json"
)


async def test_public_example_is_an_actual_repeatable_capture_export(
    tmp_path: Path,
) -> None:
    first = await capture_example(tmp_path / "first?capture.sqlite3")
    second = await capture_example(tmp_path / "second#capture.sqlite3")
    assert first.canonical_bytes == second.canonical_bytes == EXAMPLE.read_bytes()
    verified = verify_passport_bytes(EXAMPLE.read_bytes())
    assert verified.report.embedded_excerpt_count == 1
    assert verified.report.reference_only_count == 0
    assert verified.report.publisher_identity == "unverified"
    assert verified.report.semantic_assessment == "not_assessed"
    assert verified.report.persisted is False
