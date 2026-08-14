"""Build a clean release zip without duplicate or stale packaging artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
DEFAULT_OUTPUT = DIST_DIR / "a2-rag-release.zip"

EXCLUDED_DIRS = {
    ".agents",
    ".cache",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "A2_RAG_QA-main",
    "historical",
    "__pycache__",
    "dist",
    "env",
    "results",
    "venv",
}

EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_FILES = {
    "docs/ENGINEERING_REPORT.md",
    "docs/NEXT_LEVEL_RAG_UPGRADES.md",
    "docs/V3_SENIOR_UPGRADE_REPORT.md",
    "docs/V4_SENIOR_UPGRADE_REPORT.md",
    "docs/V5_RUNTIME_HARDENING_REPORT.md",
    "docs/V6_AUDIT_REMEDIATION_REPORT.md",
    "docs/V7_INDEPENDENT_AUDIT_REMEDIATION.md",
}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".jks", ".keystore"}
SENSITIVE_NAMES = {"credentials.json", "service-account.json", "secrets.toml"}
TEXT_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "assigned provider key": re.compile(
        r"(?im)^\s*(?:OPENAI|OPENROUTER|GOOGLE)_API_KEY\s*=\s*(?!your-|replace|example|\.\.\.|$)[^\s#]{12,}\s*$"
    ),
}


def _is_inside_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def should_include(path: Path, *, include_pdfs: bool = False) -> bool:
    """Return whether a repository file belongs in the default release zip."""
    if path.is_symlink() or not path.is_file() or not _is_inside_root(path):
        return False
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return False
    if rel.as_posix() in EXCLUDED_FILES:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    lower_name = path.name.lower()
    if lower_name in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
        return False
    if lower_name == ".env" or (lower_name.startswith(".env.") and lower_name != ".env.example"):
        return False
    if path.suffix.lower() == ".pdf" and not include_pdfs:
        return False
    return True


def collect_release_files(*, include_pdfs: bool = False) -> List[Path]:
    files: List[Path] = []
    for directory, dirnames, filenames in os.walk(ROOT, topdown=True, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in EXCLUDED_DIRS and not (directory_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = directory_path / filename
            if should_include(path, include_pdfs=include_pdfs):
                files.append(path)
    return sorted(files)


def find_embedded_secrets(files: Iterable[Path]) -> List[str]:
    """Return high-confidence secret findings from release-bound text files."""
    findings: List[str] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{path.relative_to(ROOT).as_posix()}: {label}")
    return findings


def _manifest_lines(files: Iterable[Path], *, include_pdfs: bool) -> List[str]:
    lines = [
        "A2-RAG release manifest",
        "",
        "Default exclusions:",
        "- A2_RAG_QA-main duplicate comparison copy",
        "- historical legacy archives",
        "- generated Python/test/lint caches",
        "- local virtual environments",
        "- .git/.agents/.codex metadata",
        "- generated evaluation results",
        "- superseded engineering and audit reports",
        "- .env variants, credentials, private keys, and secret stores",
        "- root PDFs unless --include-pdfs is passed",
        "",
        f"PDFs included: {include_pdfs}",
        "",
        "Files:",
    ]
    lines.extend(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}"
        for path in files
    )
    return lines


def _write_reproducible_archive(output: Path, files: Sequence[Path], manifest: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
            manifest_info = zipfile.ZipInfo("RELEASE_MANIFEST.txt", date_time=(1980, 1, 1, 0, 0, 0))
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            manifest_info.external_attr = 0o100644 << 16
            archive.writestr(manifest_info, manifest.encode("utf-8"))
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def verify_release_archive(output: Path, files: Sequence[Path]) -> None:
    """Verify archive membership, manifest hashes, and deterministic metadata."""
    expected = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if set(names) != set(expected) | {"RELEASE_MANIFEST.txt"}:
            raise ValueError("Release archive membership differs from the selected source files")
        if any(entry.date_time != (1980, 1, 1, 0, 0, 0) for entry in archive.infolist()):
            raise ValueError("Release archive contains non-deterministic timestamps")

        manifest = archive.read("RELEASE_MANIFEST.txt").decode("utf-8")
        manifest_entries = {
            name: digest
            for digest, name in re.findall(r"^([0-9a-f]{64})  (.+)$", manifest, flags=re.MULTILINE)
        }
        if manifest_entries != expected:
            raise ValueError("Release manifest does not match selected source hashes")
        for name, expected_digest in expected.items():
            actual_digest = hashlib.sha256(archive.read(name)).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(f"Release archive hash mismatch for {name}")


def build_release(output: Path = DEFAULT_OUTPUT, *, include_pdfs: bool = False, dry_run: bool = False) -> tuple[int, str]:
    output = output if output.is_absolute() else ROOT / output
    output = output.resolve()
    if output.parent != DIST_DIR.resolve() or output.suffix.lower() != ".zip":
        raise ValueError(f"Release output must be a .zip directly inside {DIST_DIR}")

    files = collect_release_files(include_pdfs=include_pdfs)
    secret_findings = find_embedded_secrets(files)
    if secret_findings:
        raise ValueError("Refusing to package possible secrets:\n- " + "\n- ".join(secret_findings))
    manifest = "\n".join(_manifest_lines(files, include_pdfs=include_pdfs)) + "\n"

    if dry_run:
        return len(files), "dry-run"

    _write_reproducible_archive(output, files, manifest)
    verify_release_archive(output, files)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return len(files), digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Release zip path inside the project root.")
    parser.add_argument("--include-pdfs", action="store_true", help="Include root PDF reports in the release zip.")
    parser.add_argument("--dry-run", action="store_true", help="List release count without writing a zip.")
    args = parser.parse_args()

    count, result = build_release(args.output, include_pdfs=args.include_pdfs, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Release dry run: {count} files would be included.")
    else:
        print(f"Release built: {args.output.resolve()}")
        print(f"Included files: {count}")
        print(f"SHA256: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
