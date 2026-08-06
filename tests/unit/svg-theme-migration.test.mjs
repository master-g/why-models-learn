import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
	analyzeSvgMigration,
	applySvgMigration,
	structuralProjection,
} from "../../scripts/lib/svg-theme-migration.mjs";
import { validateSvgTheme } from "../../scripts/lib/svg-theme-contract.mjs";

const LEGACY_STANDARD_SVG = `
<svg width="200" height="120" viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10"><path d="M0 0L10 5L0 10Z" fill="#312f2f"/></marker>
  </defs>
  <rect width="200" height="120" fill="#faf9f5"/>
  <line x1="10" y1="100" x2="180" y2="100" stroke="#b8b2a8" marker-end="url(#arrow)"/>
  <path d="M10 90L100 20" fill="none" stroke="#cc785c"/>
  <circle cx="10" cy="90" r="3" fill="#312f2f"/>
  <text x="20" y="20" fill="#6c6a64">标签</text>
</svg>`;

const LEGACY_AMBIGUOUS_SVG = `
<svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
  <text x="5" y="20" fill="#123456">标签</text>
  <circle cx="60" cy="40" r="12" fill="#123456"/>
</svg>`;

describe("SVG theme migration", () => {
	it("classifies a standard paper-ink SVG as safe to migrate", () => {
		const report = analyzeSvgMigration(LEGACY_STANDARD_SVG, { asset: "fixture.svg" });

		assert.equal(report.classification, "standard");
		assert.deepEqual(report.reasons, []);
		assert.equal(report.mapping["#312f2f"], "svg-ink");
		assert.equal(report.mapping["#b8b2a8"], "svg-axis");
	});

	it("routes an ambiguous color to the specialized queue", () => {
		const report = analyzeSvgMigration(LEGACY_AMBIGUOUS_SVG, { asset: "ambiguous.svg" });

		assert.equal(report.classification, "specialized");
		assert.ok(report.reasons.some((reason) => reason.code === "AMBIGUOUS_COLOR_ROLE"));
		assert.equal(report.approved, false);
	});

	it("preserves the structural projection when applying a safe mapping", () => {
		const result = applySvgMigration(LEGACY_STANDARD_SVG, { asset: "fixture.svg" });

		assert.equal(result.analysis.classification, "standard");
		assert.deepEqual(
			structuralProjection(LEGACY_STANDARD_SVG),
			structuralProjection(result.source),
		);
		assert.equal(validateSvgTheme(result.source, { asset: "fixture.svg" }).errors.length, 0);
		assert.match(result.source, /data-svg-theme="paper-ink-v1"/);
		assert.match(result.source, /\.svg-ink-fill\s*\{/);
	});

	it("refuses to apply a specialized mapping", () => {
		assert.throws(
			() => applySvgMigration(LEGACY_AMBIGUOUS_SVG, { asset: "ambiguous.svg" }),
			/不能自动迁移|specialized/,
		);
	});

	it("keeps CLI analysis read-only and requires --apply for writes", () => {
		const root = mkdtempSync(join(tmpdir(), "why-models-svg-migration-"));
		const file = join(root, "fixture.svg");
		try {
			writeFileSync(file, LEGACY_STANDARD_SVG);
			const before = readFileSync(file, "utf8");
			execFileSync(process.execPath, ["scripts/migrate-svg-theme.mjs", file], { encoding: "utf8" });
			assert.equal(readFileSync(file, "utf8"), before);
			execFileSync(process.execPath, ["scripts/migrate-svg-theme.mjs", "--apply", file], { encoding: "utf8" });
			assert.notEqual(readFileSync(file, "utf8"), before);
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});
});
