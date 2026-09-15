import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { refreshCliDocs } from './refresh-cli-docs.mjs';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'agentic-cli-docs-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const projectDir = join(root, 'website');
  await mkdir(join(projectDir, 'content'), { recursive: true });
  await writeFile(
    join(projectDir, 'content/python-cli.md'),
    '# Python CLI reference\n\nBundled help.\n',
  );
  await writeFile(
    join(projectDir, 'content/rust-cli.md'),
    '# Rust CLI reference\n\nBundled Rust help.\n',
  );
  return { root, projectDir };
}

test('a standalone site exports bundled CLI help without Python or Cargo', async (t) => {
  const { projectDir } = await fixture(t);
  await refreshCliDocs({
    projectDir,
    python: '/nonexistent-python',
    cargo: '/nonexistent-cargo',
  });
  assert.equal(
    await readFile(
      join(projectDir, 'public/docs/latest/python-cli.md'),
      'utf8',
    ),
    '# Python CLI reference\n\nBundled help.\n',
  );
  assert.equal(
    await readFile(join(projectDir, 'public/docs/latest/rust-cli.md'), 'utf8'),
    '# Rust CLI reference\n\nBundled Rust help.\n',
  );
});

test('a broken repository generator fails instead of publishing stale CLI docs', async (t) => {
  const { root, projectDir } = await fixture(t);
  await writeFile(join(root, 'Cargo.toml'), '[workspace]\n');
  await assert.rejects(refreshCliDocs({ projectDir }));
  await assert.rejects(
    readFile(join(projectDir, 'public/docs/latest/python-cli.md')),
    { code: 'ENOENT' },
  );
});
