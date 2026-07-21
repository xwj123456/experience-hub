from pathlib import Path

from fastapi.testclient import TestClient

from experience_hub.api.app import create_app
from experience_hub.config import Settings
from experience_hub.runtime import ApplicationRuntime

EXPECTED_REDUCER_VERSIONS = {
    "agent_reputation": 1,
    "capsule_state": 1,
    "experience_state": 1,
    "experience_terms": 1,
    "idea_state": 1,
    "inbox_items": 1,
    "inspiration_run_state": 1,
    "mechanism_incubation": 1,
}


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


def test_health_becomes_ready_after_real_runtime_initialization(
    tmp_path: Path,
) -> None:
    runtime = ApplicationRuntime(settings=_settings(tmp_path / "health.sqlite3"))
    app = create_app(runtime=runtime)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "data": {
            "reducer_versions": EXPECTED_REDUCER_VERSIONS,
            "schema_revision": "0005_inspiration_falsifiers",
            "status": "ready",
            "version": "0.1.0",
        }
    }


def test_health_is_the_only_unversioned_exception(
    tmp_path: Path,
) -> None:
    runtime = ApplicationRuntime(settings=_settings(tmp_path / "health-route.sqlite3"))
    app = create_app(runtime=runtime)

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 404
