"""Ensure the pinned native patch matches the build manifest."""

import hashlib
import tomllib
from pathlib import Path


def test_pinned_ascend_patch_hashes_match_the_build_manifest() -> None:
    root = Path(__file__).resolve().parents[4] / "third_party" / "mooncake"
    manifest = tomllib.loads((root / "build-manifest-ascend.toml").read_text())
    for patch in manifest["patches"]:
        assert hashlib.sha256((root / patch["path"]).read_bytes()).hexdigest() == patch["sha256"]
