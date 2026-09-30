import inspect
from pathlib import Path

from tests.passport_export_fixtures import export_runtime

from experience_hub.experiences.evidence_snapshots import (
    ExperienceEvidenceSnapshotReader,
)
from experience_hub.passports.export import PassportExportService


async def test_bootstrap_exposes_one_wired_export_service(tmp_path: Path) -> None:
    async with export_runtime(tmp_path / "export.sqlite3").initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:
        assert isinstance(container.passport_export_service, PassportExportService)
        assert isinstance(
            container.experience_evidence_snapshot_reader,
            ExperienceEvidenceSnapshotReader,
        )
        assert container.passport_export_service._passports is container.passport_query
        assert container.passport_export_service._evidence is (
            container.experience_evidence_snapshot_reader
        )


def test_export_and_reader_signatures_are_keyword_only() -> None:
    for operation, names in (
        (ExperienceEvidenceSnapshotReader.read, ("session", "version")),
        (
            PassportExportService.export,
            (
                "session",
                "owner_agent_id",
                "experience_id",
                "version_id",
                "declaration",
                "parent_adoption_id",
            ),
        ),
    ):
        parameters = inspect.signature(operation).parameters
        for name in names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(PassportExportService.export)
        .parameters["parent_adoption_id"]
        .default
        is None
    )
