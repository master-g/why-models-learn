import type { APIRoute } from 'astro';
import { getSections } from '../lib/sections.mjs';
import { getEntries } from './_entries.mjs';

interface SearchEntry {
  title_zh: string | null;
  title_en: string;
  tags: string[];
  keywords_zh: string[];
  url: string;
  section: string;
}

export const GET: APIRoute = async () => {
  const entries = await getEntries();
  const sections = getSections();

  // title_en 用 slug 兜底:英文检索词(如 "backpropagation")命中 slug。
  const result: SearchEntry[] = entries.map((entry) => {
    const [section, slug] = entry.id.split('/');
    return {
      title_zh: entry.data.title,
      title_en: slug.replace(/-/g, ' '),
      tags: entry.data.tags ?? [],
      keywords_zh: [],
      url: `/${slug}/`,
      section,
    };
  });

  for (const section of sections) {
    result.push({
      title_zh: section.name_zh,
      title_en: section.name_en,
      tags: [],
      keywords_zh: [],
      url: `/category/${section.dir}/`,
      section: section.dir,
    });
  }

  return new Response(JSON.stringify(result), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
};
