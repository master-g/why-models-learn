export function getNextAvailableSlug(sectionEntries, currentSlug, availableSlugs) {
  const currentIndex = sectionEntries.indexOf(currentSlug);
  if (currentIndex < 0) return undefined;

  for (const candidate of sectionEntries.slice(currentIndex + 1)) {
    if (availableSlugs.has(candidate)) return candidate;
  }
  return undefined;
}
