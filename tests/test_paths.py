"""Weight-cache location and the registry resolver.

These guard the property that matters for packaging: nothing the library downloads
may land inside the installed package. Writing into ``site-packages`` breaks
read-only and containerised installs, defeats Docker layer caching, and stops two
environments sharing one multi-gigabyte download.
"""
import hashlib
import os
from pathlib import Path

import pytest

import physiotrack
from physiotrack import Models
from physiotrack._paths import (cache_root, legacy_weights_dir, migrate_weight_cache,
                               weights_dir)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point the cache at a temporary directory for the duration of a test."""
    monkeypatch.setenv("PHYSIOTRACK_HOME", str(tmp_path))
    return tmp_path


class TestCacheLocation:
    def test_weights_never_land_inside_the_package(self):
        package_dir = Path(physiotrack.__file__).parent
        target = weights_dir().resolve()
        assert package_dir.resolve() not in target.parents
        assert target != package_dir.resolve()

    def test_weights_dir_is_created(self, isolated_home):
        assert weights_dir().is_dir()

    def test_physiotrack_home_wins(self, isolated_home):
        assert weights_dir() == isolated_home / "weights"

    def test_home_is_read_per_call_not_at_import(self, tmp_path, monkeypatch):
        # Set after `import physiotrack` already happened, which is what a notebook
        # or a test suite does. A value cached at import time would ignore this.
        monkeypatch.setenv("PHYSIOTRACK_HOME", str(tmp_path))
        assert cache_root() == tmp_path

    def test_xdg_cache_home_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PHYSIOTRACK_HOME", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cache_root() == tmp_path / "physiotrack"

    def test_physiotrack_home_overrides_xdg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PHYSIOTRACK_HOME", str(tmp_path / "explicit"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert cache_root() == tmp_path / "explicit"


class TestResolve:
    def test_rejects_non_registry_input(self):
        with pytest.raises(ValueError):
            Models.resolve("Depth.ZipDepth.base")  # a string, not a member

    def test_weight_free_marker_resolves_to_none(self):
        # The geometric canonicalizer is an algorithm, not a checkpoint.
        assert Models.resolve(Models.Pose3D.Canonicalizer.Models.GEOMETRIC) is None

    def test_returns_cached_file_without_downloading(self, isolated_home, monkeypatch):
        model = Models.Depth.ZipDepth.base
        cached = isolated_home / "weights" / model.value
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"not a real checkpoint")

        def fail(*a, **k):
            raise AssertionError("resolve() re-downloaded an already-cached file")

        monkeypatch.setattr(Models, "download_model", staticmethod(fail))
        assert Models.resolve(model) == str(cached)

    def test_resolves_under_the_cache_root(self, isolated_home, monkeypatch):
        model = Models.Depth.ZipDepth.base
        seen = {}

        def fake_download(member, download_path=None):
            seen["path"] = download_path
            return os.path.join(str(weights_dir()), member.value)

        monkeypatch.setattr(Models, "download_model", staticmethod(fake_download))
        result = Models.resolve(model)
        assert result.startswith(str(isolated_home))


class TestLegacyMigration:
    def test_legacy_dir_is_inside_the_package(self):
        # It has to be: that is the location we are migrating away from.
        assert legacy_weights_dir().parent.parent == Path(physiotrack.__file__).parent

    def test_dry_run_moves_nothing(self, isolated_home, monkeypatch, tmp_path):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "w.pth").write_bytes(b"x" * 32)
        monkeypatch.setattr("physiotrack._paths.legacy_weights_dir", lambda: legacy)

        moved = migrate_weight_cache(dry_run=True)
        assert [s.name for s, _ in moved] == ["w.pth"]
        assert (legacy / "w.pth").exists(), "dry run must not touch the filesystem"
        assert not (weights_dir() / "w.pth").exists()

    def test_migration_moves_files_and_preserves_subdirectories(
            self, isolated_home, monkeypatch, tmp_path):
        legacy = tmp_path / "legacy"
        (legacy / "MB_train_h36m").mkdir(parents=True)
        (legacy / "flat.pth").write_bytes(b"a" * 16)
        (legacy / "MB_train_h36m" / "best_epoch.bin").write_bytes(b"b" * 16)
        monkeypatch.setattr("physiotrack._paths.legacy_weights_dir", lambda: legacy)

        migrate_weight_cache()
        target = weights_dir()
        assert (target / "flat.pth").read_bytes() == b"a" * 16
        # MotionBERT members carry a subdirectory in their value; it must survive.
        assert (target / "MB_train_h36m" / "best_epoch.bin").read_bytes() == b"b" * 16
        assert not (legacy / "flat.pth").exists(), "files are moved, not copied"

    def test_already_cached_files_are_not_overwritten(self, isolated_home, monkeypatch,
                                                     tmp_path):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "w.pth").write_bytes(b"stale")
        target = weights_dir()
        (target / "w.pth").write_bytes(b"current")
        monkeypatch.setattr("physiotrack._paths.legacy_weights_dir", lambda: legacy)

        migrate_weight_cache()
        assert (target / "w.pth").read_bytes() == b"current"
        assert not (legacy / "w.pth").exists(), "stale duplicate should be dropped"

    def test_missing_legacy_dir_is_not_an_error(self, isolated_home, monkeypatch, tmp_path):
        monkeypatch.setattr("physiotrack._paths.legacy_weights_dir",
                            lambda: tmp_path / "does-not-exist")
        assert migrate_weight_cache() == []


def test_no_module_derives_its_own_weights_path():
    """The cache location must be stated once, not restated per module.

    Twenty-one call sites used to rebuild ``<package>/modules/model_data`` by hand.
    If one reappears, the cache silently splits in two.
    """
    package_dir = Path(physiotrack.__file__).parent
    offenders = []
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "'model_data'" in text or '"model_data"' in text:
            offenders.append(str(path.relative_to(package_dir)))
    # _paths.py owns the legacy location for migration purposes.
    offenders = [o for o in offenders if o != "_paths.py"]
    assert offenders == [], f"weights path derived outside _paths.py: {offenders}"


class TestFaceRegistry:
    def test_face_task_is_listed_and_validated(self):
        assert "Face.Landmarks.face_landmarker" in Models.list(task="face")
        assert Models.info(Models.Face.Orientation.VR)["task"] == "Face"
        Models.validate_face_model(Models.Face.Gaze.eth_xgaze_resnet18, "Gaze")
        with pytest.raises(ValueError, match="Valid models"):
            Models.validate_face_model(Models.Face.Gaze.eth_xgaze_resnet18, "Landmarks")

    def test_every_third_party_face_weight_has_a_pinned_verified_source(self):
        for group in ("Landmarks", "Expression", "Gaze"):
            for member in getattr(Models.Face, group):
                url, sha256 = Models._FACE_SOURCES[member.value]
                assert url.startswith("https://") and "latest" not in url
                assert len(sha256) == 64

    def test_a_checksum_mismatch_is_refused_and_leaves_nothing(self, isolated_home,
                                                               monkeypatch):
        import requests

        class Response:
            headers = {"content-length": "5"}

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield b"bytes"

        monkeypatch.setattr(requests, "get", lambda *a, **k: Response())
        target = isolated_home / "weights"
        with pytest.raises(IOError, match="Checksum mismatch"):
            Models._download_file("https://example.invalid/w", "w.bin", target,
                                  sha256="0" * 64)
        assert list(target.iterdir()) == []
        good = hashlib.sha256(b"bytes").hexdigest()
        path = Models._download_file("https://example.invalid/w", "w.bin", target,
                                     sha256=good)
        assert Path(path).read_bytes() == b"bytes"
