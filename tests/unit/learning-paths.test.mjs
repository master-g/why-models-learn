import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import yaml from 'js-yaml';
import {
  getLearningPaths,
  validateLearningPathData,
} from '../../src/lib/learning-paths.mjs';
import { getKnownAbsent, getSections } from '../../src/lib/sections.mjs';

const source = yaml.load(readFileSync('learning-paths.yaml', 'utf8'));

function clone(value) {
  return structuredClone(value);
}

describe('learning path contract', () => {
  it('expands the five math groups to 17 core entries in declared order', () => {
    const paths = getLearningPaths();
    const coreEntries = paths.math.layers.core.entries;
    const core = coreEntries.map((entry) => entry.slug);
    const firstStage = paths.path.mainline.stages[0];

    assert.equal(core.length, 17);
    assert.deepEqual(firstStage.entries.map((entry) => entry.slug), core);
    assert.deepEqual(firstStage.mathGroups, [
      'linear-algebra',
      'calculus',
      'probability',
      'information-theory',
      'optimization',
    ]);
    assert.equal(new Set(core).size, core.length);
    assert.equal(coreEntries.every((entry) => entry.why_now_zh && entry.reading_goal_zh), true);
  });

  it('keeps the 17/56/21 math partition complete and mutually exclusive', () => {
    const paths = getLearningPaths();
    assert.deepEqual(paths.math.counts, { core: 17, backfill: 56, reference: 21, total: 94 });
    assert.equal(paths.math.allSlugs.length, 94);
    assert.equal(new Set(paths.math.allSlugs).size, 94);
  });

  it('keeps the mainline independent from classic architectures and advanced theory', () => {
    const paths = getLearningPaths();
    const mainline = new Set(paths.path.mainline.slugs);
    const optional = paths.path.optionalBranches.find((branch) => branch.id === 'classic-architectures');
    const reference = paths.path.referenceTracks.find((track) => track.id === 'advanced-theory');

    assert.equal(mainline.has('cnn'), false);
    assert.equal(mainline.has('universal-approximation-formal'), false);
    assert.equal(optional.entries.some((entry) => entry.slug === 'cnn'), true);
    assert.equal(reference.entries.some((entry) => entry.slug === 'universal-approximation-formal'), true);
    assert.equal(reference.entries.some((entry) => entry.available), false);
  });

  it('creates stable previous/next links for mainline entries', () => {
    const paths = getLearningPaths();
    const entries = paths.path.mainline.entries;
    assert.equal(entries[0].previousSlug, undefined);
    assert.equal(entries.at(-1).nextSlug, undefined);
    assert.equal(entries[0].nextSlug, entries[1].slug);
    assert.equal(entries[1].previousSlug, entries[0].slug);
    assert.equal(entries.at(-1).available, true);
  });

  it('reports missing, duplicate, and unknown math references', () => {
    const base = { sections: getSections(), knownAbsent: getKnownAbsent() };

    const missing = clone(source);
    missing.math_layers.core.groups[0].entries.pop();
    assert.throws(
      () => validateLearningPathData({ ...base, data: missing }),
      /vectors|数学分区遗漏/,
    );

    const duplicate = clone(source);
    duplicate.math_layers.backfill[0].entries.push('vectors');
    assert.throws(
      () => validateLearningPathData({ ...base, data: duplicate }),
      /vectors|重复/,
    );

    const unknown = clone(source);
    unknown.math_layers.reference.entries.push('not-an-outline-slug');
    assert.throws(
      () => validateLearningPathData({ ...base, data: unknown }),
      /not-an-outline-slug|未知数学 slug/,
    );
  });

  it('reports unknown section selectors and missing mainline articles', () => {
    const base = { sections: getSections(), knownAbsent: getKnownAbsent() };

    const unknownSection = clone(source);
    unknownSection.paths['first-pass'].stages[1].sections.push('missing-section');
    assert.throws(
      () => validateLearningPathData({ ...base, data: unknownSection }),
      /missing-section|未知 section/,
    );

    const absentMainline = clone(source);
    absentMainline.paths['first-pass'].stages[1].sections[0] = 'approximation-theory';
    assert.throws(
      () => validateLearningPathData({ ...base, data: absentMainline }),
      /universal-approximation-formal|主线正文缺失/,
    );
  });
});
