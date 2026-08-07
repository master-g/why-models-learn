import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { getLearningPaths } from '../../src/lib/learning-paths.mjs';
import {
  getArticleNavigation,
  getNextAvailableSlug,
  getSectionPathContext,
} from '../../src/lib/article-navigation.mjs';

describe('article navigation', () => {
  it('skips unfinished entries and links to the next available article', () => {
    const next = getNextAvailableSlug(
      ['written', 'todo-a', 'todo-b', 'next-written'],
      'written',
      new Set(['written', 'next-written']),
    );
    assert.equal(next, 'next-written');
  });

  it('returns no link when the remaining entries are unfinished', () => {
    const next = getNextAvailableSlug(
      ['written', 'todo-a', 'todo-b'],
      'written',
      new Set(['written']),
    );
    assert.equal(next, undefined);
  });

  it('returns no link when the current slug is outside the section', () => {
    const next = getNextAvailableSlug(['written'], 'missing', new Set(['written']));
    assert.equal(next, undefined);
  });

  it('uses the mainline across a stage boundary without query state', () => {
    const paths = getLearningPaths();
    const available = new Set([...paths.entryIndex.values()].filter((entry) => entry.available).map((entry) => entry.slug));
    const navigation = getArticleNavigation(paths, {
      slug: 'gradient-descent-theory',
      sectionEntries: paths.sections.get('optimization-theory').entries,
      availableSlugs: available,
    });

    assert.equal(navigation.mode, 'mainline');
    assert.equal(navigation.stageId, 'math-core');
    assert.equal(navigation.nextSlug, 'what-is-learning');
    assert.equal(navigation.previousSlug, 'optimization-problems');
    assert.equal(navigation.hrefQuery, undefined);
  });

  it('keeps optional architecture navigation separate and returns to Transformer', () => {
    const paths = getLearningPaths();
    const available = new Set([...paths.entryIndex.values()].filter((entry) => entry.available).map((entry) => entry.slug));
    const navigation = getArticleNavigation(paths, {
      slug: 'cnn',
      sectionEntries: paths.sections.get('cnn').entries,
      availableSlugs: available,
    });

    assert.equal(navigation.mode, 'optional-branch');
    assert.equal(navigation.branchId, 'classic-architectures');
    assert.equal(navigation.returnSlug, 'tokenization');
    assert.equal(navigation.nextSlug, 'why-convolution');
  });

  it('keeps reference articles on chapter navigation and exposes all backfill returns', () => {
    const paths = getLearningPaths();
    const available = new Set([...paths.entryIndex.values()].filter((entry) => entry.available).map((entry) => entry.slug));
    const reference = getArticleNavigation(paths, {
      slug: 'universal-approximation-formal',
      sectionEntries: paths.sections.get('approximation-theory').entries,
      availableSlugs: available,
    });
    const backfill = getArticleNavigation(paths, {
      slug: 'jacobian',
      sectionEntries: paths.sections.get('calculus').entries,
      availableSlugs: available,
    });

    assert.equal(reference.mode, 'catalog-only');
    assert.equal(reference.nextSlug, undefined);
    assert.deepEqual(backfill.backfillReturns.map((stage) => stage.id), ['neural-networks']);
  });

  it('exposes a category role only when a section has a configured path', () => {
    const paths = getLearningPaths();
    const backprop = getSectionPathContext(paths, 'backpropagation');
    const glossary = getSectionPathContext(paths, 'approximation-theory');

    assert.equal(backprop.mainlineStage.id, 'neural-networks');
    assert.deepEqual(backprop.backfillGroups.map((group) => group.id), ['backprop-training']);
    assert.equal(glossary.mainlineStage, undefined);
    assert.deepEqual(glossary.backfillGroups, []);
  });

  it('groups every math chapter by the reader action without changing outline order', () => {
    const paths = getLearningPaths();
    const probability = getSectionPathContext(paths, 'probability');

    assert.deepEqual(
      probability.mathGroups.map((group) => ({
        layer: group.layer,
        actionLabel: group.actionLabel,
        slugs: group.entries.map((entry) => entry.slug),
      })),
      [
        {
          layer: 'core',
          actionLabel: '现在读',
          slugs: ['random-variables', 'expectation', 'variance-and-covariance'],
        },
        {
          layer: 'backfill',
          actionLabel: '按需回补',
          slugs: [
            'gaussian-distribution',
            'joint-distributions',
            'marginal-and-conditional',
            'bayes-theorem',
            'independence',
            'covariance-matrix',
            'maximum-likelihood',
            'maximum-a-posteriori',
          ],
        },
        {
          layer: 'reference',
          actionLabel: '形式参考',
          slugs: [
            'probability-spaces',
            'discrete-distributions',
            'continuous-distributions',
            'law-of-large-numbers',
            'central-limit-theorem',
            'sampling',
            'change-of-variables',
            'exponential-family',
            'concentration-inequalities',
          ],
        },
      ],
    );
    assert.deepEqual(
      probability.mathGroups[2].entries.map((entry) => entry.indexInSection),
      [0, 2, 3, 12, 13, 14, 15, 18, 19],
    );
  });

  it('does not invent math groups for a non-math chapter', () => {
    const paths = getLearningPaths();
    const backprop = getSectionPathContext(paths, 'backpropagation');

    assert.deepEqual(backprop.mathGroups, []);
  });
});
