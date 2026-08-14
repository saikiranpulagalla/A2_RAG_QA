import importlib.util
from pathlib import Path


def _load_cleaner():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "clean_artifacts", root / "scripts" / "clean_artifacts.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cleaner_prunes_protected_trees(tmp_path):
    cleaner = _load_cleaner()
    project_cache = tmp_path / "package" / "__pycache__"
    venv_cache = tmp_path / ".venv" / "Lib" / "__pycache__"
    nested_cache = tmp_path / "A2_RAG_QA-main" / "__pycache__"
    for cache in (project_cache, venv_cache, nested_cache):
        cache.mkdir(parents=True)
        (cache / "module.pyc").write_bytes(b"generated")

    removed = cleaner.clean_generated_artifacts(tmp_path)

    assert removed == 1
    assert not project_cache.exists()
    assert (venv_cache / "module.pyc").exists()
    assert (nested_cache / "module.pyc").exists()
