/**
 * Chinese copywriting lint.
 *
 * Auto-fixes:
 *  - CJK ↔ latin/digit spacing (盘古空格)
 *  - half-width prose punctuation → full-width, dropping the following space
 *    (full-width punctuation never takes one). Punctuation immediately after
 *    a closing math `$` is NOT converted: it is usually an English sentence
 *    mark leaked out of a math span (e.g. "$A,$") and must be deleted or
 *    restructured by a human, so it is surfaced as an error instead.
 *
 * Errors (fail validation; not auto-fixed):
 *  - full-width punctuation 。，；：？！、 followed by an ASCII space mid-line
 *  - full-width ，；： immediately before a closing paren
 *  - half-width punctuation after a closing math `$`
 *  - sentence punctuation ,.;: trapped at the end of an inline math span
 *    (e.g. "$\mathbb{Z}:$") — display math is exempt
 *
 * Reports (not auto-fixed):
 *  - straight quotes instead of 「」
 *  - full-width digits
 */

const CJK_CHAR = '[\\u4e00-\\u9fff\\u3040-\\u309f\\u30a0-\\u30ff]';
const LATIN_DIGIT = '[A-Za-z0-9]';

export function lintChineseCopywriting(text) {
  const frontmatter = text.match(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/);
  const frontmatterEnd = frontmatter?.[0].length ?? 0;
  const lintableText = frontmatter
    ? frontmatter[0].replace(/[^\r\n]/g, ' ') + text.slice(frontmatterEnd)
    : text;

  // Blank out math spans and URLs with spaces so that line indices stay valid
  // in the original text while we lint only the prose.
  // 围栏代码块最先遮蔽:代码不是文案(algebrica 原版无此条——原文语料不含代码)。
  const blanked = lintableText
    .replace(/^`{3,}[\s\S]*?^`{3,}[ \t]*$/gm, (m) => m.replace(/[^\r\n]/g, ' '))
    .replace(/\$\$[\s\S]*?\$\$/g, (m) => m.replace(/[^\r\n]/g, ' '))
    .replace(/\$[^$\n]+\$/g, (m) => ' '.repeat(m.length))
    .replace(/https?:\/\/[^\s)]+/g, (m) => ' '.repeat(m.length))
    .replace(/\[\/?(?:shortcode|class)(?:=[^\]\r\n]+)?\]/g, (m) => ' '.repeat(m.length))
    .replace(/^[ \t]*\[[^\]\r\n]+\]:[ \t]*\S+[ \t]*$/gm, (m) => ' '.repeat(m.length))
    .replace(/^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/gm, (m) => ' '.repeat(m.length))
    .replace(/!?\[[^\]]*\]\(([^)]+)\)/g, (m, url) => {
      const idx = m.indexOf('(' + url);
      const before = m.slice(0, idx + 1);
      const after = m.slice(idx + 1 + url.length);
      return before + ' '.repeat(url.length) + after;
    });

  const punctuationFixedText = fixHalfWidthPunctuation(text, blanked);
  const fixed = fixSpacing(blanked);
  const reports = [];
  const errors = [];

  let m;

  // Full-width punctuation followed by an ASCII space and more text on the
  // same line. Scanned on the raw text (not the mask): blanked math would
  // fabricate phantom spaces after a sentence-final 。 directly followed by
  // math, e.g. "成立。$n$ 元组" is fine and must not be flagged. A trailing
  // double space at end of line is a Markdown hard break, not an error.
  const fullWidthSpace = /[。，；：？！、][ \t]+(?=\S)/g;
  while ((m = fullWidthSpace.exec(lintableText)) !== null) {
    errors.push({
      line: lineOf(text, m.index),
      message:
        `全角标点 ${m[0]} 后跟半角空格；若该标点是英文句读残留` +
        `（如 $A.$ 移出数学区所致），通常应整体删除`,
    });
  }

  // Full-width comma/semicolon/colon immediately before a closing paren is
  // never valid Chinese (。） can be legitimate for full-sentence asides).
  const punctBeforeParen = /[，；：][）)]/g;
  while ((m = punctBeforeParen.exec(lintableText)) !== null) {
    errors.push({
      line: lineOf(text, m.index),
      message: `全角标点 ${m[0][0]} 不应紧邻右括号`,
    });
  }

  // Half-width punctuation right after a closing math span, e.g. "$A$, $B$"
  // or "$A$. " — English sentence marks that belong outside (or nowhere).
  const halfWidthAfterMath = /\$[,.;:!?](?=[ \t]+\S|\$)/g;
  while ((m = halfWidthAfterMath.exec(lintableText)) !== null) {
    errors.push({
      line: lineOf(text, m.index),
      message:
        `数学区结束后残留半角标点 ${m[0][1]}；中文语境通常应删除该标点，枚举应改用顿号`,
    });
  }

  // Sentence punctuation trapped at the end of an inline math span, e.g.
  // "$\mathbb{Z}:$" or "$[(a,b)]=0;$" — English sentence marks the translator
  // left inside the math. Display math is exempt (trailing punctuation there
  // is common and usually intentional), and !/? are exempt (factorial "$n!$").
  const displayBlanked = lintableText.replace(/\$\$[\s\S]*?\$\$/g, (s) => ' '.repeat(s.length));
  const inlineMath = /\$[^$\n]+\$/g;
  while ((m = inlineMath.exec(displayBlanked)) !== null) {
    const content = m[0].slice(1, -1);
    if (/[,.;:]$/.test(content)) {
      errors.push({
        line: lineOf(text, m.index),
        message:
          `行内数学区以英文句读标点 ${content.slice(-1)} 结尾；` +
          `该标点属于句子而非公式，应移出数学区或删除`,
      });
    }
  }

  // Straight quotes in Chinese text.
  const straightQuotes = /"([^"]+)"|'([^']+)'/g;
  while ((m = straightQuotes.exec(fixed)) !== null) {
    reports.push({ line: lineOf(text, m.index), message: '建议改用「」引号' });
  }

  // Full-width digits.
  const fullWidthDigits = /[０-９]/g;
  while ((m = fullWidthDigits.exec(fixed)) !== null) {
    reports.push({ line: lineOf(text, m.index), message: `全角数字：${m[0]}` });
  }

  // Apply spacing fixes only to the body. Frontmatter is machine-readable data
  // and must remain byte-for-byte unchanged.
  const fixedText = frontmatter
    ? punctuationFixedText.slice(0, frontmatterEnd) +
      fixSpacing(punctuationFixedText.slice(frontmatterEnd))
    : fixSpacing(punctuationFixedText);
  return { text: fixedText, reports, errors };
}

function fixHalfWidthPunctuation(text, lintMask) {
  const replacements = [];
  const punctuation = /(?<![\w.])([,.;:!?](?=\s|$|[一-鿿]))/g;
  const fullWidth = {
    ',': '，',
    '.': '。',
    ';': '；',
    ':': '：',
    '!': '！',
    '?': '？',
  };
  let match;

  while ((match = punctuation.exec(lintMask)) !== null) {
    // Skip decimals like 3.14 and ellipses ...
    if (/\d\.\d/.test(lintMask.slice(Math.max(0, match.index - 1), match.index + 3))) {
      continue;
    }
    // Skip initials like "A. Carreira".
    if (
      match[0] === '.' &&
      /[A-Za-zÁÉÍÓÚáéíóúÄÖÜäöü]/.test(lintMask[match.index - 1] || '') &&
      lintMask[match.index + 1] === ' '
    ) {
      continue;
    }
    // Skip punctuation right after a closing math span ("$A$, ..."): an
    // English sentence mark leaked out of the math, which a blind conversion
    // would cement into broken prose. Surfaced as an error instead.
    if (text[match.index - 1] === '$') continue;
    // Consume horizontal spaces after the mark (a full-width mark never takes
    // a following space), unless they are trailing whitespace ending the line.
    let end = match.index + 1;
    while (end < text.length && (text[end] === ' ' || text[end] === '\t')) end++;
    const next = text[end];
    const deleteCount =
      next === undefined || next === '\n' || next === '\r' ? 1 : end - match.index;
    replacements.push({ index: match.index, deleteCount, value: fullWidth[match[0]] });
  }

  let fixed = text;
  for (const replacement of replacements.reverse()) {
    fixed =
      fixed.slice(0, replacement.index) +
      replacement.value +
      fixed.slice(replacement.index + replacement.deleteCount);
  }
  return fixed;
}

function fixSpacing(text) {
  // CJK followed by latin/digit
  const cjkBefore = new RegExp(`(${CJK_CHAR})(${LATIN_DIGIT})`, 'g');
  // latin/digit followed by CJK
  const cjkAfter = new RegExp(`(${LATIN_DIGIT})(${CJK_CHAR})`, 'g');
  return text
    .replace(cjkBefore, '$1 $2')
    .replace(cjkAfter, '$1 $2')
    .replace(/([^\s])\n/g, '$1\n') // keep line breaks
    .replace(/\n{3,}/g, '\n\n'); // collapse excessive blank lines
}

function lineOf(text, index) {
  let line = 1;
  for (let i = 0; i < index; i++) {
    if (text[i] === '\n') line++;
  }
  return line;
}
