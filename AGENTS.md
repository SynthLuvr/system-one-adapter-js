# AGENTS.md

Instructions for AI coding agents working in this repository.

## Quick Start

``` bash
pnpm install
pnpm build    # type-check
pnpm test     # run tests
pnpm lint     # lint all files
```

## Required Workflow

Always run these before considering work complete:

``` bash
pnpm build && pnpm lint && pnpm test
```

All three must pass with zero errors.

## Toolchain

Lint, format, and environment checks come from
[ts-canon](https://github.com/SynthLuvr/ts-canon) — one devDependency
that bundles biome, oxlint (tsgolint), the ast-grep rules,
convert-to-arrow, jscpd, and the pandoc/peer-deps/audit helpers.

- Run tooling through `pnpm <script>` (`pnpm lint`, `pnpm format`), not
  by invoking binaries in `node_modules/.bin` directly.
- `ts-canon lint` / `ts-canon format` accept path arguments and `--fast`
  (skips `pnpm audit` and jscpd).
- `pnpm exec ts-canon doctor` diagnoses toolchain/environment problems.
- Markdown files must be byte-identical to `pandoc --eol=lf -t gfm`
  output; fix drift with `pnpm format`.
- Tool, rule, and preset changes belong in ts-canon — bump its version
  in `package.json` to pick them up. Do not add per-step tool scripts
  back here.

## Coding Conventions (Enforced)

These are **not** preferences — the toolchain will fail if you violate
them:

### No function declarations

Use `const` + arrow function instead.

``` ts
// ❌ Wrong
function foo() {}

// ✅ Right
const foo = (): void => {};
```

**Exception: function overloads.** TypeScript overloads *require*
`function` declarations — each signature reuses the same name, which is
impossible with `const`. Overload sets are permitted, and the
`no-function-declaration` rule skips any `function_declaration` that
immediately follows an overload signature:

``` ts
// ✅ Allowed — overloads must use function declarations
function toLabel(value: string): string;
function toLabel(value: number): string;
function toLabel(value: string | number): string {
  return `label: ${value}`;
}
```

### No inline exports

Write `export` as a separate statement.

``` ts
// ❌ Wrong
export const foo = 1;

// ✅ Right
const foo = 1;
export { foo };
```

### Single-statement braces are stripped

`if`/`for`/`while` with a single body statement should not have braces.

``` ts
// ❌ Wrong
if (done) {
  return;
}

// ✅ Right
if (done)
  return;
```

### Formatting

- Double quotes
- 2-space indentation
- 80-character line width
- Trailing commas
- Semicolons
- Arrow function parentheses always

## Formatting

If the linter complains about formatting, run:

``` bash
pnpm format
```

This runs five steps in order (via `ts-canon format`):

1.  `convert-to-arrow` — rewrites `function` declarations to arrow
    consts
2.  strip-braces — removes unnecessary braces from single-statement
    blocks
3.  `biome format` — formats all files
4.  `biome check` — applies lint auto-fixes
5.  `pandoc` — normalizes markdown to GFM

## Project Structure

- Source code lives in `src/`
- Tests live in `src/tests/` (filenames end in `.test.ts`)
- TypeScript is type-check only (`noEmit: true`) — bundled tooling runs
  TypeScript sources via tsx inside ts-canon
- ESM only (`"type": "module"`)
