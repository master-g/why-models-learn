import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const baseLayout = readFileSync('src/layouts/BaseLayout.astro', 'utf8');
const homePage = readFileSync('src/pages/index.astro', 'utf8');
const articlePage = readFileSync('src/pages/[slug].astro', 'utf8');
const overrides = readFileSync('public/styles/zh-overrides.css', 'utf8');

describe('visual theme contracts', () => {
  it('keeps the desktop header spacer used by the upstream breadcrumb layout', () => {
    assert.match(baseLayout, /<header id="header" class="header">/);
    assert.doesNotMatch(baseLayout, /<header id="header" class="header only-mobile">/);
  });

  it('keeps linked module index headings on the upstream ink color', () => {
    assert.match(homePage, /class="module-index-heading__title"/);
    assert.match(
      overrides,
      /\.module-index-heading__title a\s*\{[^}]*color:\s*#312f2f;/s,
    );
  });

  it('centers standalone Markdown illustrations inside article sections', () => {
    assert.match(
      overrides,
      /\.post-section p > img:only-child\s*\{[^}]*display:\s*block;[^}]*margin-inline:\s*auto;/s,
    );
  });

  it('centers paragraph-only MathJax formulas without changing inline math', () => {
    assert.match(
      overrides,
      /\.post-section p\.standalone-math > mjx-container\s*\{[^}]*display:\s*block;[^}]*text-align:\s*center;/s,
    );
    assert.match(
      overrides,
      /@media screen and \(max-width: 480px\)\s*\{[^}]*\.post-section mjx-container\[display="true"\] > svg,[^}]*\.post-section p\.standalone-math > mjx-container > svg\s*\{[^}]*max-width:\s*100%;[^}]*height:\s*auto;/s,
    );
    assert.match(
      overrides,
      /@media screen and \(max-width: 480px\)\s*\{[\s\S]*?\.post-section mjx-container\[display="true"\]\[width="full"\] > svg\s*\{[^}]*min-width:\s*0\s*!important;/s,
    );
    assert.match(
      overrides,
      /\.post-section mjx-container\[display="true"\]\[width="full"\]\s*\{[^}]*width:\s*100%\s*!important;[^}]*min-width:\s*0\s*!important;/s,
    );
  });

  it('contains long inline MathJax expressions inside mobile list items', () => {
    assert.match(
      overrides,
      /\.post-section li > mjx-container:not\(\[display="true"\]\)\s*\{[^}]*display:\s*inline-block;[^}]*max-width:\s*100%;[^}]*overflow-x:\s*auto;/s,
    );
    assert.match(
      overrides,
      /\.post-section li:has\(> mjx-container:not\(\[display="true"\]\)\)\s*\{[^}]*text-align:\s*left;/s,
    );
    assert.match(
      overrides,
      /\.post-section li > mjx-container:not\(\[display="true"\]\) > svg\s*\{[^}]*max-width:\s*100%;[^}]*height:\s*auto;/s,
    );
  });

  it('contains wide article tables without widening the mobile page', () => {
    assert.match(
      overrides,
      /\.post-section table\s*\{[^}]*display:\s*block;[^}]*max-width:\s*100%;[^}]*overflow-x:\s*auto;/s,
    );
    assert.match(
      overrides,
      /\.post-section \.table-1,[^}]*\.post-section \.table-sign\s*\{[^}]*overflow-x:\s*auto;/s,
    );
    assert.match(
      overrides,
      /\.post-section \.table-1 > table,[^}]*\.post-section \.table-sign > table\s*\{[^}]*width:\s*max-content;[^}]*min-width:\s*100%;/s,
    );
  });

  it('renders the article footer with the site license note', () => {
    assert.match(articlePage, /<footer class="post-footer-attribution">/);
  });
});
