import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { getNextAvailableSlug } from '../../src/lib/article-navigation.mjs';

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
});
