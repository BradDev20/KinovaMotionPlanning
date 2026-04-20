from __future__ import annotations

import json
from pathlib import Path

ARTIFACT_SEMANTICS_VERSION = "full_body_collision_v2"
ARTIFACT_SEMANTICS_DESCRIPTION = (
    "Full-body robot-vs-obstacle and self-collision validity with physical obstacle contacts "
    "and MuJoCo-based full-body obstacle cost."
)
LEGACY_ARTIFACT_SEMANTICS_VERSION = "legacy_pre_full_body_collision_v2"


def artifact_semantics_payload() -> dict[str, str]:
    return {
        "artifact_semantics_version": ARTIFACT_SEMANTICS_VERSION,
        "artifact_semantics_description": ARTIFACT_SEMANTICS_DESCRIPTION,
    }


def semantics_version_from_payload(payload: dict) -> str:
    return str(payload.get("artifact_semantics_version", LEGACY_ARTIFACT_SEMANTICS_VERSION))


def load_json_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_current_artifact_semantics(path: Path, *, artifact_label: str) -> dict:
    payload = load_json_payload(path)
    version = semantics_version_from_payload(payload)
    if version != ARTIFACT_SEMANTICS_VERSION:
        raise RuntimeError(
            f"{artifact_label} at {path} uses artifact semantics '{version}'. "
            f"Expected '{ARTIFACT_SEMANTICS_VERSION}'. Rebuild/recollect/retrain under the repaired "
            "full-body collision semantics before continuing."
        )
    return payload
