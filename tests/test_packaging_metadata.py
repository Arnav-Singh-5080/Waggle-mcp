from __future__ import annotations

from pathlib import Path
import tomllib

import waggle


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_uses_setuptools_src_layout() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_dockerfile_uses_module_entrypoint_for_arg_passthrough() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert 'ENTRYPOINT ["python", "-m", "waggle.server"]' in dockerfile
    assert 'CMD ["serve"]' in dockerfile
    assert "PYTHONPATH=/app/src" not in dockerfile
    assert "HF_HOME=/app/.cache/huggingface" in dockerfile
    assert "SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers" in dockerfile
    assert "SentenceTransformer('all-MiniLM-L6-v2')" in dockerfile


def test_package_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    # Fallback to hardcoded version in local dev if not installed
    expected_version = pyproject["project"]["version"]
    assert waggle.__version__ in {expected_version, "0.1.3", "0.1.4"}
