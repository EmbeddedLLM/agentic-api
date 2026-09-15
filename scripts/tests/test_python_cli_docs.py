"""The docs generator must follow the real parser without starting a gateway."""

import argparse
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate_python_cli_docs.py"


class PythonCliDocsTests(unittest.TestCase):
    def test_new_nested_commands_flags_and_defaults_appear_automatically(self):
        generate = runpy.run_path(str(GENERATOR))["render_reference"]
        parser = argparse.ArgumentParser(prog="example")
        command = parser.add_subparsers().add_parser("new-command")
        nested = command.add_subparsers().add_parser("inspect")
        nested.add_argument("--limit", type=int, default=42, help="Maximum items")
        markdown = generate(parser, "1.2.3")
        self.assertIn("## example new-command inspect", markdown)
        self.assertIn("--limit LIMIT", markdown)
        self.assertIn("Maximum items (default: 42)", markdown)

    def test_real_source_parser_needs_no_wheel_vllm_or_gateway(self):
        namespace = runpy.run_path(str(GENERATOR))
        parser, version = namespace["load_source_parser"]()
        markdown = namespace["render_reference"](parser, version)
        for command in ("serve", "doctor", "version"):
            self.assertIn(f"## agentic-api {command}", markdown)
        self.assertIn("--vllm-base-url", markdown)
        self.assertIn("OPENAI_API_KEY", markdown)
        self.assertNotIn("agentic_api.launcher", sys.modules)
        self.assertNotIn("vllm", sys.modules)
        options = parser.parse_args(["serve", "--vllm-base-url", "http://localhost:8000"]).options
        self.assertEqual(options.mode, "remote")

    def test_check_rejects_stale_docs_without_overwriting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.md"
            command = [sys.executable, str(GENERATOR), "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run([*command, "--check"], check=True, capture_output=True)
            output.write_text("stale\n", encoding="utf-8")
            result = subprocess.run([*command, "--check"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("out of date", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "stale\n")


if __name__ == "__main__":
    unittest.main()
