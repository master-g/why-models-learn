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
import { lintChineseCopywriting } from './lib/copywriting-lint.mjs';

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

// 生成 vault 内的导航索引 _index.md(下划线前缀,不参与同步;生成物,勿手改)。
// statusMap: slug → vault 内的 status(无文件 = 未写)。
function writeVaultIndex(statusMap) {
  const data = yaml.load(readFileSync('sections.yaml', 'utf8'));
  const parts = (data.parts || []).slice().sort((a, b) => a.order - b.order);
  const sections = (data.sections || []).slice().sort((a, b) => a.order - b.order);
  const byPart = new Map();
  for (const sec of sections) {
    const key = sec.part ?? '';
    if (!byPart.has(key)) byPart.set(key, []);
    byPart.get(key).push(sec);
  }

  const MARK = { complete: '✅', reference: '📚', active: '📝' };
  const lines = [
    '# nn-to-llm 索引',
    '',
    '> 生成物(`npm run sync` 顺带生成),勿手改。✅=已毕业 📝=草稿 ⬜=未写;点 ⬜ 的链接可直接创建笔记。',
    '',
  ];
  let total = 0, done = 0, draft = 0;
  for (const part of parts) {
    lines.push(`## ${part.name_zh}`);
    lines.push('');
    for (const sec of byPart.get(part.id) || []) {
      lines.push(`### ${sec.order}. ${sec.name_zh}`);
      for (const slug of sec.entries || []) {
        const st = statusMap.get(slug);
        total++;
        if (st === 'complete' || st === 'reference') done++;
        else if (st) draft++;
        lines.push(`- ${MARK[st] ?? '⬜'} [[${slug}]]`);
      }
      lines.push('');
    }
  }
  lines.push(`---`);
  lines.push(`进度:${done}/${total} 已毕业,${draft} 篇草稿`);
  lines.push('');
  writeFileSync(join(VAULT_DIR, '_index.md'), lines.join('\n'));
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
  const statusMap = new Map(); // slug -> vault status(供 _index.md)

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
    statusMap.set(slug, SYNCABLE_STATUS.has(data.status) ? data.status : 'active');

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
    const outContent = `${outFrontmatter.join('\n')}\n${outBody}`;

    // 文案 lint(沿用 algebrica 规则):只警告不改写,vault 是事实源,由作者在 vault 侧修。
    const lint = lintChineseCopywriting(outContent);
    for (const r of lint.reports) warn(`${slug}:${r.line} LINT ${r.message}`);
    for (const e of lint.errors) warn(`${slug}:${e.line} LINT-ERROR ${e.message}`);

    targets.set(`${section}/${slug}`, outContent);
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
  writeVaultIndex(statusMap);
  console.log(`[sync] 同步 ${targets.size} 个词条(${VAULT_DIR} → ${OUT_ROOT}/),已刷新 _index.md`);
}

main();
