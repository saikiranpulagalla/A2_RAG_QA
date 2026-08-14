import hashlib
import importlib.util
import re
import zipfile
from pathlib import Path

import pytest


def _load_build_release():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_release", root / "scripts" / "build_release.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_release_excludes_nested_copy_and_pdfs():
    build_release = _load_build_release()
    files = [path.relative_to(build_release.ROOT).as_posix() for path in build_release.collect_release_files()]

    assert "README.md" in files
    assert "scripts/build_release.py" in files
    assert not any(path.startswith("A2_RAG_QA-main/") for path in files)
    assert not any(path.startswith("historical/") for path in files)
    assert not any(path.endswith(".pdf") for path in files)
    assert not any(path.startswith("results/") for path in files)


def test_default_release_name_is_the_current_canonical_artifact():
    build_release = _load_build_release()

    assert build_release.DEFAULT_OUTPUT.name == "a2-rag-release.zip"


def test_release_output_must_be_a_zip_directly_inside_dist():
    build_release = _load_build_release()

    with pytest.raises(ValueError, match="dist"):
        build_release.build_release(build_release.ROOT / "README.md", dry_run=True)
    with pytest.raises(ValueError, match="dist"):
        build_release.build_release(build_release.ROOT / "data" / "release.zip", dry_run=True)
    assert build_release.build_release(build_release.ROOT / "dist" / "custom-release.zip", dry_run=True)[1] == "dry-run"


def test_release_can_include_pdfs_when_explicit():
    build_release = _load_build_release()
    files = [path.relative_to(build_release.ROOT).as_posix() for path in build_release.collect_release_files(include_pdfs=True)]

    assert not any(path.endswith(".pdf") for path in files)
    assert not any(path.startswith("historical/") for path in files)
    assert "docs/ENGINEERING_REPORT.md" not in files


def test_release_policy_excludes_secrets_and_environment_variants(tmp_path):
    build_release = _load_build_release()
    original_root = build_release.ROOT
    build_release.ROOT = tmp_path
    try:
        for name in (".env", ".env.local", "secrets.toml", "private.pem"):
            path = tmp_path / name
            path.write_text("sensitive", encoding="utf-8")
            assert build_release.should_include(path) is False
        example = tmp_path / ".env.example"
        example.write_text("OPENAI_API_KEY=", encoding="utf-8")
        assert build_release.should_include(example) is True
    finally:
        build_release.ROOT = original_root


def test_release_build_is_reproducible(tmp_path):
    build_release = _load_build_release()
    first = build_release.ROOT / "dist" / "repro-one.zip"
    second = build_release.ROOT / "dist" / "repro-two.zip"
    try:
        _count, first_digest = build_release.build_release(first)
        _count, second_digest = build_release.build_release(second)
        assert first_digest == second_digest
        assert first.read_bytes() == second.read_bytes()
    finally:
        for path in (first, second, first.with_suffix(".zip.sha256"), second.with_suffix(".zip.sha256")):
            if path.exists():
                path.unlink()


def test_release_manifest_hashes_match_archived_bytes(tmp_path):
    build_release = _load_build_release()
    output = build_release.ROOT / "dist" / "manifest-verification.zip"
    sidecar = output.with_suffix(".zip.sha256")
    try:
        file_count, _digest = build_release.build_release(output)
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            manifest = archive.read("RELEASE_MANIFEST.txt").decode("utf-8")
            entries = {
                name: digest
                for digest, name in re.findall(
                    r"^([0-9a-f]{64})  (.+)$", manifest, flags=re.MULTILINE
                )
            }

            assert len(entries) == file_count
            assert set(entries) == set(names) - {"RELEASE_MANIFEST.txt"}
            for name, expected_digest in entries.items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected_digest
    finally:
        for path in (output, sidecar):
            if path.exists():
                path.unlink()
