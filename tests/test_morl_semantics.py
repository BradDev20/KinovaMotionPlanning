from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.morl.semantics import (
    ARTIFACT_SEMANTICS_VERSION,
    LEGACY_ARTIFACT_SEMANTICS_VERSION,
    require_current_artifact_semantics,
    semantics_version_from_payload,
)


class ArtifactSemanticsTests(unittest.TestCase):
    def test_semantics_version_defaults_to_legacy_when_missing(self):
        self.assertEqual(semantics_version_from_payload({}), LEGACY_ARTIFACT_SEMANTICS_VERSION)

    def test_require_current_artifact_semantics_rejects_legacy_payload(self):
        artifact_dir = Path("test_artifacts") / "semantics_tmp"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload_path = artifact_dir / "legacy_summary.json"
        payload_path.write_text(json.dumps({"planner_mode": "max"}, indent=2), encoding="utf-8")

        with self.assertRaises(RuntimeError):
            require_current_artifact_semantics(payload_path, artifact_label="Dataset summary")

    def test_require_current_artifact_semantics_accepts_current_payload(self):
        artifact_dir = Path("test_artifacts") / "semantics_tmp"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload_path = artifact_dir / "current_summary.json"
        payload_path.write_text(
            json.dumps({"artifact_semantics_version": ARTIFACT_SEMANTICS_VERSION}, indent=2),
            encoding="utf-8",
        )

        payload = require_current_artifact_semantics(payload_path, artifact_label="Dataset summary")
        self.assertEqual(payload["artifact_semantics_version"], ARTIFACT_SEMANTICS_VERSION)


if __name__ == "__main__":
    unittest.main()
