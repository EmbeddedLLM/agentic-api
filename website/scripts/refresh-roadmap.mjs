import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export async function refreshRoadmap({ projectDir = resolve('.') } = {}) {
  const source = resolve(projectDir, '../ROADMAP.md');
  const snapshot = join(projectDir, 'content/roadmap.md');
  const published = join(projectDir, 'public/roadmap.md');
  let markdown;
  try {
    markdown = await readFile(source, 'utf8');
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    markdown = await readFile(snapshot, 'utf8');
  }
  assert.match(markdown, /^# [^\r\n]+\r?\n/, 'Roadmap must start with a title');
  await mkdir(dirname(snapshot), { recursive: true });
  await mkdir(dirname(published), { recursive: true });
  await writeFile(snapshot, markdown);
  await writeFile(published, markdown);
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  await refreshRoadmap();
  console.log('Refreshed the roadmap page source and Markdown export.');
}
