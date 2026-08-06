import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { renderPageMarkdown } from '../../src/pages/_render-page.mjs';

describe('renderPageMarkdown article structure', () => {
  it('wraps every level-two heading and its content in an article section', async () => {
    const html = await renderPageMarkdown(`## Introduction

First paragraph.

## Properties

Second paragraph.
`);

    const sections = html.match(/<section class="article-section">/g) || [];
    assert.equal(sections.length, 2);
    assert.match(
      html,
      /<section class="article-section"><h2[^>]*>Introduction<\/h2>\s*<p>First paragraph\.<\/p>\s*<\/section>/,
    );
    assert.match(
      html,
      /<section class="article-section"><h2[^>]*>Properties<\/h2>\s*<p>Second paragraph\.<\/p>\s*<\/section>/,
    );
  });

  it('keeps the trailing MathJax companion style outside the themed sections', async () => {
    const html = await renderPageMarkdown(`## Formula

$x + y$
`);

    assert.match(html, /<\/section>\s*<style>/);
    assert.match(html, /<\/style>$/);
    assert.equal((html.match(/<section class="article-section">/g) || []).length, 1);
  });

  it('marks paragraph-only inline MathJax output for display-style centering', async () => {
    const html = await renderPageMarkdown(`## Formula

$$x + y$$

The value $x + y$ stays inline.
`);

    assert.match(
      html,
      /<p class="standalone-math"><mjx-container class="MathJax" jax="SVG">/,
    );
    assert.doesNotMatch(html, /<p class="standalone-math">The value/);
  });
});
