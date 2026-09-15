import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { refreshRoadmap } from './refresh-roadmap.mjs';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'agentic-roadmap-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const projectDir = join(root, 'website');
  await mkdir(join(projectDir, 'content'), { recursive: true });
  await mkdir(join(projectDir, 'public'));
  await writeFile(
    join(projectDir, 'content/roadmap.md'),
    '# Snapshot\n\nBundled goals.\n',
  );
  await writeFile(join(projectDir, 'public/roadmap.md'), '# Old export\n');
  return { root, projectDir };
}

test('a repository roadmap change replaces both the rendered source and Markdown export', async (t) => {
  const { root, projectDir } = await fixture(t);
  await writeFile(
    join(root, 'ROADMAP.md'),
    '# Current roadmap\n\nUpdated goals.\n',
  );
  await refreshRoadmap({ projectDir });
  for (const file of ['content/roadmap.md', 'public/roadmap.md'])
    assert.equal(
      await readFile(join(projectDir, file), 'utf8'),
      '# Current roadmap\n\nUpdated goals.\n',
    );
});

test('a standalone site exports its bundled roadmap without fetching GitHub', async (t) => {
  const { projectDir } = await fixture(t);
  await refreshRoadmap({ projectDir });
  assert.equal(
    await readFile(join(projectDir, 'public/roadmap.md'), 'utf8'),
    '# Snapshot\n\nBundled goals.\n',
  );
});

test('an invalid repository roadmap fails without publishing the old snapshot', async (t) => {
  const { root, projectDir } = await fixture(t);
  await writeFile(join(root, 'ROADMAP.md'), '');
  await assert.rejects(
    refreshRoadmap({ projectDir }),
    /Roadmap must start with a title/,
  );
  assert.equal(
    await readFile(join(projectDir, 'public/roadmap.md'), 'utf8'),
    '# Old export\n',
  );
});
