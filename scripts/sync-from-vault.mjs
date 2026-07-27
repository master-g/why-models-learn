#!/usr/bin/env node
/**
 * 单向同步:Obsidian vault 飞地 → content-zh/。
 *
 * 事实源:vault 的 `03 - AREAS/learning/nn-to-llm/<slug>.md`(feynman 卡片)。
 * 只同步 `status: complete | reference`;`active` 草稿不出 vault。
 *
 * 适配规则(写作无感,脚本负责):
 *   - `[[slug]]` / `[[slug|显示文本]]` → 站内相对链接(按 sections.yaml 的 slug→章节映射)
 *   - 指向飞地外笔记或未知 slug 的 wikilink → 降级为纯文本并警告
 *   - Obsidian callout 标记 `> [!tldr]` 剥掉,内容保留为 blockquote
 *   - frontmatter 只透传 title(必填)与 tags(可选),vault-only 字段不透传
 *
 * content-zh/ 是纯同步产物,不手改;不在目标集里的旧文件会被清除。
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import yaml from 'js-yaml';
import { parseFrontmatter, splitFrontmatter } from './lib/frontmatter.mjs';

const VAULT_DIR = join(homedir(), 'Documents/ObsidianVaults/Main/03 - AREAS/learning/nn-to-llm');
const OUT_ROOT = 'content-zh';
const SYNCABLE_STATUS = new Set(['complete', 'reference']);

function loadSlugSections() {
  const data = yaml.load(readFileSync('sections.yaml', 'utf8'));
  const map = new Map();
  for (const sec of data.sections || []) {
    for (const slug of sec.entries || []) map.set(slug, sec.dir);
  }
  return map;
}

function rewriteWikilinks(body, slugSections, warn) {
  return body.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (match, target, alias) => {
    const slug = target.trim();
    const section = slugSections.get(slug);
    if (!section) {
      warn(`wikilink 无法解析,降级为纯文本: [[${target}]]`);
      return (alias || target).trim();
    }
    const text = (alias || slug).trim();
    return `[${text}](../${section}/${slug}/)`;
  });
}

function stripCallouts(body) {
  // `> [!tldr] 标题` → `> **标题**`;无标题的标记行 → `>`
  return body.replace(/^> \[!\w+\][ \t]*(.*)$/gm, (_m, title) =>
    title.trim() ? `> **${title.trim()}**` : '>',
  );
}

function main() {
  if (!existsSync(VAULT_DIR)) {
    console.log(`[sync] vault 飞地不存在: ${VAULT_DIR}(尚未创建,无词条可同步)`);
    return;
  }

  const slugSections = loadSlugSections();
  const warnings = [];
  const warn = (msg) => warnings.push(msg);
  const targets = new Map(); // 'section/slug' -> file content

  for (const file of readdirSync(VAULT_DIR)) {
    if (!file.endsWith('.md') || file.startsWith('_')) continue;
    const slug = file.slice(0, -3);
    const section = slugSections.get(slug);
    if (!section) {
      warn(`${file}: slug 不在 sections.yaml,跳过`);
      continue;
    }

    const raw = readFileSync(join(VAULT_DIR, file), 'utf8');
    const { frontmatter, body } = splitFrontmatter(raw);
    const data = parseFrontmatter(frontmatter) || {};

    if (!SYNCABLE_STATUS.has(data.status)) {
      console.log(`[sync] 跳过(status: ${data.status ?? 'missing'}): ${slug}`);
      continue;
    }
    if (typeof data.title !== 'string' || !data.title.trim()) {
      warn(`${file}: 缺少 title 字段,跳过`);
      continue;
    }

    const outFrontmatter = ['---', `title: ${JSON.stringify(data.title)}`];
    if (Array.isArray(data.tags) && data.tags.length) {
      outFrontmatter.push(`tags: ${JSON.stringify(data.tags)}`);
    }
    outFrontmatter.push('---');

    const outBody = stripCallouts(rewriteWikilinks(body, slugSections, warn));
    targets.set(`${section}/${slug}`, `${outFrontmatter.join('\n')}\n${outBody}`);
  }

  // 清除不在目标集里的旧词条(保留 .gitkeep)。
  if (existsSync(OUT_ROOT)) {
    for (const dir of readdirSync(OUT_ROOT, { withFileTypes: true })) {
      if (!dir.isDirectory()) continue;
      for (const file of readdirSync(join(OUT_ROOT, dir.name))) {
        if (!file.endsWith('.md')) continue;
        const key = `${dir.name}/${file.slice(0, -3)}`;
        if (!targets.has(key)) {
          rmSync(join(OUT_ROOT, dir.name, file));
          console.log(`[sync] 清除过期词条: ${key}`);
        }
      }
    }
  }

  for (const [key, content] of targets) {
    const outPath = join(OUT_ROOT, `${key}.md`);
    mkdirSync(join(OUT_ROOT, key.split('/')[0]), { recursive: true });
    writeFileSync(outPath, content);
  }

  for (const msg of warnings) console.warn(`[sync] ${msg}`);
  console.log(`[sync] 同步 ${targets.size} 个词条(${VAULT_DIR} → ${OUT_ROOT}/)`);
}

main();
