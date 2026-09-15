"""Refresh the Python and Rust CLI references before MkDocs discovers documentation files."""

import subprocess
import sys
from pathlib import Path


def on_pre_build(config):
    root = Path(config.config_file_path).resolve().parent
    subprocess.run([sys.executable, str(root / "scripts/generate_python_cli_docs.py")], check=True)
    rust = subprocess.run(
        ["cargo", "run", "--quiet", "--locked", "--manifest-path", str(root / "Cargo.toml"), "-p", "agentic-cli-docs"],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    (root / "docs/reference/rust-cli.md").write_text(rust.stdout, encoding="utf-8")


def on_page_markdown(markdown, page, **kwargs):
    if page.file.src_uri in {"reference/python-cli.md", "reference/rust-cli.md"}:
        # There is no tracked Markdown file for the theme's "Edit this page" action.
        page.edit_url = None
    return markdown
