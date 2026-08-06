/**
 * SVG 主题迁移工具的纯核心。
 *
 * 自动路径只接受已登记的标准纸墨颜色。任何渐变、样式块、未知颜色或
 * 语义混用都进入专用队列,不会生成候选改写。
 */
import { STANDARD_SVG_ROLES, normalizeHex, validateSvgTheme } from "./svg-theme-contract.mjs";

const PAINT_ATTRIBUTE = /\b(fill|stroke|stop-color|flood-color)\s*=\s*(["'])([^"']*)\2/gi;
const COLOR_VALUE = /#[0-9a-f]{3,8}\b/i;
const NON_PAINT_VALUE = /^(?:none|transparent|url\([^)]*\))$/i;
const TAG_NAME = /<([a-z][a-z0-9:-]*)\b([^>]*?)(\/?)>/gi;
const STANDARD_COLOR_TO_ROLE = new Map();

for (const [role, definition] of Object.entries(STANDARD_SVG_ROLES)) {
	STANDARD_COLOR_TO_ROLE.set(definition.light, role);
	STANDARD_COLOR_TO_ROLE.set(definition.dark, role);
}

function reason(code, message, extra = {}) {
	return { code, message, ...extra };
}

function tagAt(source, index) {
	const prefix = source.slice(Math.max(0, index - 240), index);
	return prefix.match(/<([a-z][a-z0-9:-]*)\b[^>]*$/i)?.[1]?.toLowerCase() || "unknown";
}

function colorContext(source, index, property) {
	const tag = tagAt(source, index);
	if (tag === "text") return "text";
	if (property === "stroke" || tag === "line" || tag === "path" || tag === "polyline") return "graphic";
	return "fill";
}

function collectPaints(source) {
	const paints = [];
	for (const match of source.matchAll(PAINT_ATTRIBUTE)) {
		const property = match[1].toLowerCase();
		const value = match[3].trim();
		if (NON_PAINT_VALUE.test(value)) continue;
		const color = value.match(COLOR_VALUE)?.[0];
		paints.push({
			property,
			value,
			color: color ? normalizeHex(color) : null,
			context: colorContext(source, match.index, property),
			index: match.index,
		});
	}
	return paints;
}

function hasStyleColors(source) {
	return /<style\b[\s\S]*?<\/style>/i.test(source);
}

function hasGradient(source) {
	return /<(?:linear|radial)Gradient\b/i.test(source);
}

export function structuralProjection(source) {
	const withoutComments = String(source)
		.replace(/<!--[\s\S]*?-->/g, "")
		.replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, "");
	const projection = [];
	let cursor = 0;
	for (const match of withoutComments.matchAll(TAG_NAME)) {
		const text = withoutComments.slice(cursor, match.index).replace(/\s+/g, " ").trim();
		if (text) projection.push(`text:${text}`);
		const tag = match[1].toLowerCase();
		const attributes = [...match[2].matchAll(/([a-z_:][-a-z0-9_.:]*)\s*=\s*(["'])([^"']*)\2/gi)]
			.filter(([, name]) => !["class", "style", "fill", "stroke", "color", "stop-color", "flood-color", "data-svg-theme"].includes(name.toLowerCase()))
			.map(([, name, , value]) => `${name.toLowerCase()}=${value}`)
			.sort();
		projection.push(`<${tag}${attributes.length ? ` ${attributes.join(" ")}` : ""}${match[3] ? "/" : ""}>`);
		cursor = match.index + match[0].length;
	}
	const tail = withoutComments.slice(cursor).replace(/\s+/g, " ").trim();
	if (tail) projection.push(`text:${tail}`);
	return projection.join("\n");
}

export function analyzeSvgMigration(source, { asset = "<inline-svg>" } = {}) {
	const text = String(source);
	const reasons = [];
	const mapping = {};
	const paints = collectPaints(text);
	const byColor = new Map();

	if (!/<svg\b/i.test(text) || !/<\/svg>\s*$/i.test(text.trim())) {
		reasons.push(reason("INVALID_SVG_ROOT", "SVG 根元素不完整"));
	}
	if (paints.length === 0) reasons.push(reason("NO_PAINTS", "没有找到可迁移的颜色属性"));
	if (hasStyleColors(text)) reasons.push(reason("STYLED_COLOR_SOURCE", "颜色位于 style 块中,需要逐图处理"));
	if (hasGradient(text)) reasons.push(reason("GRADIENT_SOURCE", "包含渐变,需要逐图定义专用角色"));

	for (const paint of paints) {
		if (!paint.color) {
			reasons.push(reason("UNSUPPORTED_PAINT_VALUE", `不支持的绘制值 ${paint.value}`, { property: paint.property }));
			continue;
		}
		if (STANDARD_COLOR_TO_ROLE.has(paint.color)) {
			mapping[paint.color] = STANDARD_COLOR_TO_ROLE.get(paint.color);
			continue;
		}
		if (!byColor.has(paint.color)) byColor.set(paint.color, new Set());
		byColor.get(paint.color).add(paint.context);
	}
	for (const [color, contexts] of byColor) {
		if (contexts.size > 1) {
			reasons.push(reason("AMBIGUOUS_COLOR_ROLE", `颜色 ${color} 同时出现在 ${[...contexts].join("、")} 语义中`, { color, contexts: [...contexts] }));
		} else {
			reasons.push(reason("UNREGISTERED_COLOR", `颜色 ${color} 没有已确认的标准角色`, { color, contexts: [...contexts] }));
		}
	}

	const classification = reasons.length === 0 ? "standard" : "specialized";
	return {
		asset,
		classification,
		approved: classification === "standard",
		mapping,
		reasons,
		paintCount: paints.length,
	};
}

function classFor(role, property) {
	return `${role}-${property}`;
}

function addClasses(attributes, classes) {
	const unique = [...new Set(classes)];
	if (unique.length === 0) return attributes;
	const classAttribute = attributes.match(/\bclass\s*=\s*(["'])([^"']*)\1/i);
	if (classAttribute) {
		const existing = classAttribute[2].trim();
		const value = [...new Set(`${existing} ${unique.join(" ")}`.trim().split(/\s+/))].join(" ");
		return attributes.replace(classAttribute[0], `class=${classAttribute[1]}${value}${classAttribute[1]}`);
	}
	return `${attributes} class="${unique.join(" ")}"`;
}

function styleBlock(classes) {
	const entries = [...classes].sort().map((className) => {
		const property = className.endsWith("-stroke") ? "stroke" : "fill";
		const role = className.replace(/-(?:fill|stroke)$/, "");
		const definition = STANDARD_SVG_ROLES[role];
		return { className, property, definition };
	});
	const lines = ["<style>"];
	for (const { className, property, definition } of entries) {
		lines.push(`  .${className} { ${property}: ${definition.light}; }`);
	}
	lines.push("  @media (prefers-color-scheme: dark) {");
	for (const { className, property, definition } of entries) {
		lines.push(`    .${className} { ${property}: ${definition.dark}; }`);
	}
	lines.push("  }");
	lines.push("</style>");
	return lines.join("\n");
}

function addThemeMarker(source) {
	return source.replace(/<svg\b([^>]*)>/i, (match, attributes) => {
		if (/\bdata-svg-theme\s*=/.test(attributes)) return match;
		return `<svg data-svg-theme="paper-ink-v1"${attributes}>`;
	});
}

function rewriteStandardPaints(source, mapping) {
	const classes = new Set();
	const rewritten = source.replace(TAG_NAME, (match, tag, attributes, selfClose) => {
		if (tag.toLowerCase() === "svg" || tag.toLowerCase() === "style") return match;
		const localClasses = [];
		const nextAttributes = attributes.replace(PAINT_ATTRIBUTE, (paint, property, quote, value) => {
			const normalized = normalizeHex(value);
			const role = normalized ? mapping[normalized] : null;
			if (!role || !["fill", "stroke"].includes(property.toLowerCase())) return paint;
			const className = classFor(role, property.toLowerCase());
			localClasses.push(className);
			classes.add(className);
			return "";
		});
		return `<${tag}${addClasses(nextAttributes, localClasses)}${selfClose}>`;
	});
	return { source: rewritten, classes };
}

function insertStyle(source, classes) {
	const block = styleBlock(classes);
	return source.replace(/(<svg\b[^>]*>)/i, `$1\n${block}`);
}

export function applySvgMigration(source, { asset = "<inline-svg>" } = {}) {
	const analysis = analyzeSvgMigration(source, { asset });
	if (!analysis.approved) {
		const details = analysis.reasons.map(({ code, message }) => `${code}: ${message}`).join("; ");
		throw new Error(`SVG ${asset} 不能自动迁移: ${details}`);
	}
	const rewritten = rewriteStandardPaints(String(source), analysis.mapping);
	const migrated = insertStyle(addThemeMarker(rewritten.source), rewritten.classes);
	if (structuralProjection(source) !== structuralProjection(migrated)) {
		throw new Error(`SVG ${asset} 迁移后结构投影发生变化`);
	}
	const contract = validateSvgTheme(migrated, { asset });
	if (contract.errors.length > 0) {
		throw new Error(`SVG ${asset} 迁移结果未通过主题契约: ${contract.errors.map(({ code }) => code).join(", ")}`);
	}
	return { source: migrated, analysis, changed: migrated !== source };
}
