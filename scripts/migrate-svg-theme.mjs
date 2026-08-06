#!/usr/bin/env node
/**
 * SVG 主题迁移命令。
 *
 * 默认只分析。只有显式传入 --apply,且所有文件都属于标准组时才写入。
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
	analyzeSvgMigration,
	applySvgMigration,
} from "./lib/svg-theme-migration.mjs";

function usage() {
	console.error("用法: node scripts/migrate-svg-theme.mjs [--apply] <svg-path> [...]");
}

function main() {
	const args = process.argv.slice(2);
	const apply = args.includes("--apply");
	const files = args.filter((arg) => arg !== "--apply");
	if (files.length === 0) {
		usage();
		process.exitCode = 2;
		return;
	}
	const inputs = files.map((file) => ({ file, absolute: resolve(file), source: existsSync(resolve(file)) ? readFileSync(resolve(file), "utf8") : null }));
	const missing = inputs.filter(({ source }) => source === null);
	if (missing.length > 0) {
		for (const { file } of missing) console.error(`[svg-migrate] 文件不存在: ${file}`);
		process.exitCode = 1;
		return;
	}
	const reports = inputs.map(({ file, source }) => analyzeSvgMigration(source, { asset: file }));
	for (const report of reports) {
		console.log(`[svg-migrate] ${report.asset}: ${report.classification} (${report.paintCount} 个颜色属性)`);
		for (const item of report.reasons) console.log(`  - ${item.code}: ${item.message}`);
	}
	if (!apply) return;
	if (reports.some((report) => !report.approved)) {
		console.error("[svg-migrate] 存在非标准 SVG,未写入任何文件");
		process.exitCode = 1;
		return;
	}
	for (const { file, absolute, source } of inputs) {
		const result = applySvgMigration(source, { asset: file });
		writeFileSync(absolute, result.source);
		console.log(`[svg-migrate] 已应用: ${file}`);
	}
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
