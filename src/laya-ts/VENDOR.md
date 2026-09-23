# Vendored: laya-ts

This directory is a vendored copy of
[`laya-ts`](https://github.com/NandhaKishorM/laya/tree/main/laya-ts),
the TypeScript inference port of the
[laya](https://github.com/NandhaKishorM/laya) System 1 decision engine,
merged upstream in
[NandhaKishorM/laya#243](https://github.com/NandhaKishorM/laya/pull/243).

- Source repository: `NandhaKishorM/laya`, directory `laya-ts`
- Vendored commit: `ec8409e542941bb4bb649d5fec00d4cec96ae024` (the PR
  \#243 merge commit, `laya-ts` 0.1.0)
- License: Apache-2.0 (see `LICENSE`; upstream `LICENSE`, copied
  verbatim)
- The package is not published to npm yet, so it is vendored instead of
  depended on. When an npm release appears, delete this directory and
  depend on the published package instead.
- `scripts/export_onnx.py` (repo root) is vendored from the same commit
  for the one-time checkpoint export the engine needs.

## What is excluded from the copy

Upstream’s `package.json`, `tsconfig.json`, vitest tests, examples, and
README are not carried: this repo compiles the sources through its own
toolchain and the wrapper in `src/providers/laya.ts` is the supported
public surface. Everything else is byte-identical to upstream, except
the `convert-to-arrow` codemod is pre-applied (function declarations
became arrow consts) so `pnpm format` stays idempotent over the copy.

## Toolchain exclusions

The nested `.gitignore` marks this directory ignored so ast-grep,
oxlint, jscpd, and biome (through `vcs.useIgnoreFile`) skip it: the
sources keep upstream’s style, which intentionally does not follow this
repository’s enforced conventions. Git keeps tracking the files because
they were already committed; add future files with `git add -f`. Vitest
coverage excludes the directory in `vitest.config.ts` for the same
reason.

## Resync procedure

``` bash
commit=ec8409e542941bb4bb649d5fec00d4cec96ae024
curl -sL "https://github.com/NandhaKishorM/laya/archive/$commit.tar.gz" |
  tar xz -C /tmp
cp /tmp/laya-*/laya-ts/src/*.ts src/laya-ts/
node node_modules/.pnpm/convert-to-arrow@*/node_modules/convert-to-arrow/dist/cli.js src/laya-ts
git -C src/laya-ts add -f .
```

Then re-run `pnpm build && pnpm lint && pnpm test`, and update the
vendored commit in this file.
