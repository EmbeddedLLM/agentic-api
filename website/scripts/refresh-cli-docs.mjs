import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';

const run = promisify(execFile);

export async function refreshCliDocs({
  projectDir = resolve('.'),
  python = process.env.PYTHON || 'python3',
  cargo = process.env.CARGO || 'cargo',
} = {}) {
  const root = resolve(projectDir, '..');
  const snapshot = join(projectDir, 'content/python-cli.md');
  const rustSnapshot = join(projectDir, 'content/rust-cli.md');
  let repository = true;
  try {
    await access(join(root, 'Cargo.toml'));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    repository = false;
  }
  if (repository) {
    // A broken source build must fail, rather than quietly publishing stale help.
    await run(python, [
      join(root, 'scripts/generate_python_cli_docs.py'),
      '--output',
      snapshot,
    ]);
    const { stdout } = await run(cargo, [
      'run',
      '--quiet',
      '--locked',
      '--manifest-path',
      join(root, 'Cargo.toml'),
      '-p',
      'agentic-cli-docs',
    ]);
    assert.match(stdout, /^# Rust CLI reference\r?\n/);
    await mkdir(dirname(rustSnapshot), { recursive: true });
    await writeFile(rustSnapshot, stdout);
  }
  for (const [kind, title] of [
    ['python', 'Python'],
    ['rust', 'Rust'],
  ]) {
    const markdown = await readFile(
      join(projectDir, `content/${kind}-cli.md`),
      'utf8',
    );
    assert.ok(markdown.startsWith(`# ${title} CLI reference\n`));
    const published = join(projectDir, `public/docs/latest/${kind}-cli.md`);
    await mkdir(dirname(published), { recursive: true });
    await writeFile(published, markdown);
  }
}
