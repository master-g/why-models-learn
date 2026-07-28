import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { lintChineseCopywriting } from '../../scripts/lib/copywriting-lint.mjs';

describe('lintChineseCopywriting', () => {
  it('auto-fixes CJK↔latin spacing', () => {
    const { text, reports } = lintChineseCopywriting('中文ABC中文');
    assert.equal(text, '中文 ABC 中文');
    assert.equal(reports.length, 0);
  });

  it('auto-fixes CJK↔digit spacing', () => {
    const { text, reports } = lintChineseCopywriting('中文123');
    assert.equal(text, '中文 123');
    assert.equal(reports.length, 0);
  });

  it('auto-fixes half-width punctuation in Chinese prose and drops the trailing space', () => {
    const input = '你好, 世界';
    const { text, reports, errors } = lintChineseCopywriting(input);
    assert.equal(text, '你好，世界');
    assert.ok(!reports.some((r) => /半角标点/.test(r.message)));
    assert.equal(errors.length, 0);
  });

  it('errors on full-width punctuation followed by an ASCII space mid-line', () => {
    const input = '记号 $x \\notin A$ 表示 $x$ 不是 $A$。 的元素。';
    const { text, errors } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.equal(errors.length, 1);
    assert.ok(errors[0].message.includes('全角标点'));
  });

  it('does not flag a Markdown hard break after full-width punctuation', () => {
    const input = '按如下方式计算：  \n\n$$\nx\n$$';
    const { errors } = lintChineseCopywriting(input);
    assert.equal(errors.length, 0);
  });

  it('does not flag sentence-final punctuation directly followed by math', () => {
    const input = '恒等式成立。$n$ 元组的情形同理。';
    const { errors } = lintChineseCopywriting(input);
    assert.equal(errors.length, 0);
  });

  it('errors on full-width comma before a closing paren', () => {
    const input = '对于每对整数 $a$ 和 $b$（$b \\neq 0$，），存在唯一的商。';
    const { errors } = lintChineseCopywriting(input);
    assert.equal(errors.length, 1);
    assert.ok(errors[0].message.includes('右括号'));
  });

  it('errors on half-width punctuation after a closing math span and leaves it unconverted', () => {
    const input = '用大写字母 $A$, $B$, $C$, 表示。';
    const { text, errors } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.equal(errors.length, 3);
    assert.ok(errors[0].message.includes('半角标点'));
  });

  it('errors on sentence punctuation trapped at the end of an inline math span', () => {
    const input = '仅用到 $\\mathbb{N}:$ 上的加法，以及 $[(a,b)]=0;$；';
    const { text, errors } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.equal(errors.length, 2);
    assert.ok(errors[0].message.includes('数学区以英文句读标点'));
  });

  it('exempts display math, factorials, and mid-math punctuation', () => {
    const input = '集合 $\\{1, 2\\}$ 与阶乘 $n!$：\n\n$$\nA = B \\iff C,\n$$\n\n区间 $[0,1]$ 同理。';
    const { errors } = lintChineseCopywriting(input);
    assert.equal(errors.length, 0);
  });

  it('reports straight quotes in Chinese text', () => {
    const input = '他说"你好"。';
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.ok(reports.some((r) => /建议改用/.test(r.message)));
  });

  it('reports full-width digits', () => {
    const { text, reports } = lintChineseCopywriting('数字１２３');
    assert.equal(text, '数字１２３');
    assert.ok(reports.some((r) => /全角数字/.test(r.message)));
  });

  it('does not mangle inline math, display math, URLs, or markdown links', () => {
    const input = '令 $a<b$ 且 $$\\sum_{i=1}^{n} i$$，访问 https://example.com/foo 或 [链接](https://example.com/bar)。';
    const { text, reports, errors } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.ok(!reports.some((r) => /半角标点|建议改用|全角数字/.test(r.message)));
    assert.equal(errors.length, 0);
  });

  it('treats escaped dollar signs as text, not math', () => {
    const input = '\\$1,500，另一半高于 \\$1,500';
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(text, '\\$1,500，另一半高于 \\$1,500');
    assert.ok(!reports.some((r) => /半角标点/.test(r.message)));
  });

  it('does not treat LaTeX prime as a straight quote', () => {
    const input = '导数 $f\'(x)$ 存在。';
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(text, '导数 $f\'(x)$ 存在。');
    assert.ok(!reports.some((r) => /建议改用/.test(r.message)));
  });

  it('ignores quoted attributes in shortcode tags', () => {
    const input = '区间如下：\n\n[shortcode="intervals"]\n| $a$ | $b$ |\n[/shortcode]';
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.ok(!reports.some((r) => /建议改用/.test(r.message)));
  });

  it('ignores quoted attributes in class wrapper tags', () => {
    const input = '[class="table-1 -right"]\n\n| 恒等式 | 结果 |\n|---|---|\n\n[/class]';
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.ok(!reports.some((r) => /建议改用/.test(r.message)));
  });

  it('ignores alignment colons in Markdown table delimiter rows', () => {
    const input = '| 数值 | 说明 |\n| :--: | :------ |\n| 1 | 示例 |';
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.ok(!reports.some((r) => /半角标点/.test(r.message)));
  });

  it('ignores the structural colon in Markdown reference definitions', () => {
    const input = '![图 3][示意图]\n\n[示意图]: /assets/trigonometry/svg/example.zh.svg';
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(text, input);
    assert.ok(!reports.some((r) => /半角标点/.test(r.message)));
  });

  it('preserves YAML frontmatter and ignores quoted metadata values', () => {
    const input = [
      '---',
      'title: 中文ABC',
      'translation:',
      '  updated: "2026-07-23T08:19:04.773Z"',
      '---',
      '正文ABC。',
    ].join('\n');
    const { text, reports } = lintChineseCopywriting(input);
    assert.equal(
      text,
      [
        '---',
        'title: 中文ABC',
        'translation:',
        '  updated: "2026-07-23T08:19:04.773Z"',
        '---',
        '正文 ABC。',
      ].join('\n'),
    );
    assert.ok(!reports.some((r) => /建议改用/.test(r.message)));
  });

  it('skips fenced code blocks and preserves line numbers after them', () => {
    const input = [
      '正文。',
      '```python',
      'print("a + b =", a + b)',
      '```',
      '这是 "引号" 行。',
    ].join('\n');
    const { reports } = lintChineseCopywriting(input);
    const quoteReports = reports.filter((r) => /建议改用/.test(r.message));
    assert.equal(quoteReports.length, 1);
    assert.equal(quoteReports[0].line, 5);
  });
});
