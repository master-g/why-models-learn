#!/usr/bin/env node
/**
 * 构建产物插图闸:扫 dist 下全部 html,插图引用必须可解析。
 *
 * 两类失败:
 *   1. `<img src="/assets/...">` 指向的文件在 dist 下不存在(同步漏拷 → 线上 404)
 *   2. `<img src="svg/...">` 漏网相对路径(rewrite 插件未改写,浏览器会按页面路径解析 → 404)
 *
 * 挂在 npm postbuild 生命周期上,与 check-mjx-errors 串联。
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { extname, resolve, join } from 'node:path';

function* walkHtml(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = resolve(dir, entry.name);
    if (entry.isDirectory()) yield* walkHtml(p);
    else if (entry.isFile() && extname(entry.name) === '.html') yield p;
  }
}

const dir = resolve(process.argv[2] || 'dist');
if (!existsSync(dir)) {
  console.error(`[check-svg] 目录不存在: ${dir}`);
  process.exit(1);
}

const assetRe = /<img[^>]+src="(\/assets\/[^"]+)"/g;
const relativeRe = /<img[^>]+src="(svg\/[^"]+)"/g;
const checked = new Set();
let failed = false;

for (const file of walkHtml(dir)) {
  const html = readFileSync(file, 'utf8');
  let m;
  while ((m = assetRe.exec(html)) !== null) {
    const src = m[1];
    if (checked.has(src)) continue;
    checked.add(src);
    if (!existsSync(join(dir, src))) {
      console.error(`[check-svg] 404: ${src}(引用于 ${file})`);
      failed = true;
    }
  }
  while ((m = relativeRe.exec(html)) !== null) {
    console.error(`[check-svg] 未改写的相对路径: ${m[1]}(在 ${file})`);
    failed = true;
  }
}

if (failed) process.exit(1);
console.log(`[check-svg] ${checked.size} 个插图引用全部可解析`);
