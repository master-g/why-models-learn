import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

const pilots = [
	{
		path: "content-zh/linear-algebra/lengths-and-distances.md",
		footnote: "lengths-norm-equivalence",
		label: "有限维提醒",
	},
	{
		path: "content-zh/training-nn/mnist-mlp-training-loop.md",
		footnote: "mnist-test-boundary",
		label: "指标口径",
	},
];

describe("旁注试点内容合同", () => {
	for (const pilot of pilots) {
		it(`${pilot.path} 同时包含唯一编号旁注与无编号边注`, () => {
			const source = readFileSync(pilot.path, "utf8");
			const reference = `[^${pilot.footnote}]`;

			assert.equal(source.split(reference).length - 1, 2);
			assert.match(source, new RegExp(`^\\[\\^${pilot.footnote}\\]: .+`, "m"));
			assert.match(source, new RegExp(`^> \\[!marginnote\\] ${pilot.label}$`, "m"));
		});
	}
});
