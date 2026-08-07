import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function article(section, slug) {
  return readFileSync(`content-zh/${section}/${slug}.md`, 'utf8');
}

function assertBefore(text, earlier, later) {
  const earlierIndex = text.indexOf(earlier);
  const laterIndex = text.indexOf(later);
  assert.notEqual(earlierIndex, -1, `缺少前置内容：${earlier}`);
  assert.notEqual(laterIndex, -1, `缺少后续内容：${later}`);
  assert.ok(earlierIndex < laterIndex, `“${earlier}”必须早于“${later}”`);
}

describe('math first-pass reading contracts', () => {
  it('starts random variables from a concrete result table before formal probability spaces', () => {
    const text = article('probability', 'random-variables');
    assertBefore(text, '## 先把实验结果记成数字', String.raw`(\Omega,\mathcal F,P)`);
    assert.match(text, /> \[!marginnote\] 第一遍读法/);
  });

  it('teaches concrete norms before the abstract axioms', () => {
    const text = article('linear-algebra', 'norms');
    assertBefore(text, '## 先会计算三种常见范数', '## 三条公理');
  });

  it('defines the dot product before gradient uses it for a direction', () => {
    const text = article('calculus', 'gradient');
    assertBefore(text, '点积的计算规则', String.raw`\nabla f(a,b)\cdot\boldsymbol u`);
  });

  it('marks the first-pass boundary before convergence theory', () => {
    const text = article('optimization-theory', 'gradient-descent-theory');
    assertBefore(text, '## 第一遍到这里需要掌握什么', '## L-光滑性控制二阶余项');
    assert.match(text, /> \[!marginnote\] 第二遍内容/);
  });

  it('gives a direct classification example before the KL decomposition', () => {
    const text = article('information-theory', 'cross-entropy');
    assertBefore(text, '## 第一遍：先计算一次分类损失', '## 形式参考：交叉熵与 KL 散度');
  });
});
