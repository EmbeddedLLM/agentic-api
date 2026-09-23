from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
INSTALL_GUIDE = REPO_ROOT / "docs" / "guides" / "python-installation.md"


def test_documented_python_installs_match_the_published_website_version() -> None:
    guide = INSTALL_GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    index = DOCS_INDEX.read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "website" / "lib" / "quickstart.ts").read_text(encoding="utf-8")
    match = re.search(r"PUBLISHED_VERSION = '([^']+)'", quickstart)
    assert match is not None
    version = match.group(1)

    for document in (readme, guide):
        assert f"python -m pip install agentic-api=={version}" in document
        assert f'python -m pip install "agentic-api[local]=={version}"' in document
        assert f"uvx --from agentic-api=={version} agentic --version" in document

    for document in (readme, index, guide):
        assert f"https://pypi.org/project/agentic-api/{version}/" in document
        assert "not published on PyPI" not in document
        assert "This release produces wheel artifacts" not in document
        assert "After PyPI publication" not in document
        assert "after the PyPI publication gate" not in document
        assert "uvx pip install" not in document

    assert "python -m pip install /absolute/path/to/agentic_api-PLATFORM.whl" in guide
    assert f"uv pip install agentic-api=={version}" in readme
    assert f'uv pip install "agentic-api[local]=={version}"' in readme
    assert f"uvx --from agentic-api=={version} agentic-api doctor" in readme
    assert (
        f"uvx --from agentic-api=={version} agentic-api serve --vllm-base-url http://existing-vllm:8000"
        in readme
    )


def test_python_install_guide_covers_workflows_and_backend_language() -> None:
    guide = INSTALL_GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "agentic-api serve --vllm-base-url http://existing-vllm:8000" in guide
    assert "agentic-api serve --model Qwen/Qwen3-30B-A3B-FP8" in guide
    assert "agentic-api doctor --mode remote" in guide
    assert "agentic run codex --model MODEL_ID" in guide
    assert "agentic run claude --model SERVED_MODEL_ALIAS" in guide
    assert "vLLM is one supported backend, not part of the Agentic API product name" in readme
    assert "The Rust-native `agentic` CLI remains supported" in guide


def test_python_install_guide_documents_known_good_model_profiles() -> None:
    guide = INSTALL_GUIDE.read_text(encoding="utf-8")

    assert "documentation data, not an allowlist" in guide
    assert "Qwen/Qwen3-30B-A3B-FP8" in guide
    assert "qwen3-30b-a3b-fp8" in guide
    assert "vllm serve Qwen/Qwen3-30B-A3B-FP8 --reasoning-parser deepseek_r1 --port 5050" in guide
    assert "vllm serve Qwen/Qwen3-30B-A3B-FP8 --tool-call-parser hermes --enable-auto-tool-choice --port 5050" in guide
    assert "Qwen/Qwen3.5-35B-A3B-FP8" in guide
    assert "Qwen/Qwen3.8-27B-FP8" in guide
