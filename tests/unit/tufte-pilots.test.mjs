import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const CONTENT = join(ROOT, 'content-zh');

async function markdownFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await markdownFiles(path));
    if (entry.isFile() && entry.name.endsWith('.md')) files.push(path);
  }
  return files;
}

test('数学词条旁注试点只保留一个受控 marginfigure', async () => {
  const pilot = await readFile(join(CONTENT, 'linear-algebra/lengths-and-distances.md'), 'utf8');
  assert.equal((pilot.match(/^> \[!marginfigure\]/gm) || []).length, 1);
  assert.match(pilot, /> !\[三种距离球的轮廓对比\]\(\/assets\/linear-algebra\/svg\/lengths-and-distances\.1\.svg\)/);
});

test('模型词条 fullwidth 试点在浏览器尺寸证据前保持未采用', async () => {
  const pilot = await readFile(join(CONTENT, 'training-nn/mnist-mlp-training-loop.md'), 'utf8');
  assert.doesNotMatch(pilot, /^> \[!fullwidth\]/m);
});

test('布局试点没有批量扩散到其他同步词条', async () => {
  const files = await markdownFiles(CONTENT);
  const layoutMarkers = [];
  for (const file of files) {
    const source = await readFile(file, 'utf8');
    for (const match of source.matchAll(/^> \[!(marginfigure|fullwidth|epigraph)\]/gm)) {
      layoutMarkers.push({ file, type: match[1] });
    }
  }
  assert.deepEqual(layoutMarkers.map(({ type }) => type), ['marginfigure']);
  assert.match(layoutMarkers[0].file, /linear-algebra\/lengths-and-distances\.md$/);
});
