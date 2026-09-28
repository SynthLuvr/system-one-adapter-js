# Changelog

## Unreleased

Ported from upstream `system-one-adapter-python` v0.2.1 (`e1d4cc9`).

### Bug Fixes

- the OpenAI chat provider now rejects completions whose `finish_reason`
  is anything other than `"stop"` or `null` (for example `length` or
  `content_filter`) with a non-retryable `TypeSafeError` (“did not
  complete: …”), instead of silently feeding possibly truncated or empty
  text into decoding, where it burned the corrective malformed-output
  retry budget and surfaced a misleading schema error
- the Anthropic provider likewise rejects any `stop_reason` other than
  `"end_turn"`, `"stop_sequence"`, or `null` (for example `refusal` or
  `model_context_window_exceeded`), keeping the dedicated truncation
  message for `max_tokens`
- the OpenAI Responses provider now scans `response.output` for refusal
  content parts and raises “was a refusal: …” even when a valid-looking
  message is also present
- like upstream, these provider-declared non-answers raise plain
  `TypeSafeError`s that never consume the transient-retry or
  corrective-retry budgets, while `error.debug.llm_attempts` keeps the
  recorded request, full provider response, and finish reason

### Features

- OpenAI-compatible endpoints may omit token usage: missing or null
  usage, or usage with missing/null count fields, no longer fails the
  evaluation. `ProviderResult` counts are `number | null`;
  `AdapterUsage` overrides the SDK’s non-nullable `Usage` with nullable
  `input_tokens`/`output_tokens` and nullable cumulative
  `input_tokens_total`/`output_tokens_total`, which are `null` once any
  attempt omitted a count. Reported counts, including zero, are
  preserved. The claude-code provider reports absent CLI usage as `null`
  instead of coercing it to zero (laya’s deliberate zeros are unchanged)

## v0.5.0 (2026-09-23)

### Breaking Changes

- the laya provider no longer spawns python: it runs the laya engine
  in-process through the upstream `laya-ts` TypeScript port, installed
  as a GitHub dependency from the `SynthLuvr/laya` fork (tag
  `laya-ts-v0.1.0`), which commits the compiled `dist/` because upstream
  does not publish the package to npm; it drives the exported
  `encoder.onnx` + `head.onnx` weights with onnxruntime-node; the
  `python` option, the `LAYA_PYTHON` environment variable, and the
  `PythonRunner`, `PythonResult`, `defaultRunner`, and `LAYA_SCRIPT`
  exports are gone, replaced by the `models`, `device`, `numThreads`,
  and `session` options on `LayaOptions` — and because the engine reads
  ONNX exports instead of safetensors checkpoints, the weights must be
  exported once per checkpoint with the export script shipped inside the
  `laya-ts` package (or served from a Hugging Face repo that hosts
  them); locations come from the `models` option or `LAYA_MODEL_DIR`,
  defaulting to the official `convaiinnovations/laya` bundle layout

### Features

- `LayaOptions.device` selects the ONNX execution device (`cpu` or
  `cuda`, with an automatic CPU fallback), `numThreads` caps ONNX
  Runtime intra-op parallelism, and `session` injects a prebuilt ONNX
  session shim for tests and custom runtimes; the `router` model now
  routes states to the english or multilingual checkpoint in-process
  with laya’s script/stopword language detection, loading checkpoints
  lazily and unloading them on `close`

## v0.4.0 (2026-09-21)

### Features

- new `claude_code` provider evaluates through the Claude Code CLI: each
  request spawns
  `MAX_THINKING_TOKENS=0 claude -p --output-format json --model … --tools "" --no-session-persistence`,
  passing the answer schema to `--json-schema` in structured mode and
  the conversation on stdin; CLI API failures map to the SDK error
  classes with their HTTP status so the retry policy applies, hung
  processes abort with `APITimeoutError`, and input tokens include the
  CLI’s cache-write and cache-read tokens
- new `laya` provider runs the local System 1 decision engine
  ([laya](https://github.com/NandhaKishorM/laya), `pip install laya`)
  through a one-shot python process per request: named with
  `provider: "laya"` plus one of the `router`, `english`,
  `multilingual`, or `typed-decisions` checkpoints (or constructed
  directly as `LayaProvider`), it answers choice, score, and noul
  questions natively with calibrated probabilities, so token counts stay
  zero and only latency is metered; the python interpreter defaults to
  `python3` and is overridable via `LAYA_PYTHON` or the `python` option
- provider requests now carry the validated questions, state, and answer
  mode in an optional `typed` field on `ProviderRequestOptions`, so
  providers that answer typed questions directly no longer need to
  reverse-engineer them from the prompt or JSON schema
- the laya provider’s tests run the real engine end to end — real python
  one-shot, real `convaiinnovations/laya` checkpoints from the Hugging
  Face hub — instead of stubbing the runner or the package; a repo-local
  venv under `node_modules/.cache/laya-venv` (or any `LAYA_PYTHON`
  interpreter) hosts it, and nothing is ever skipped: without an
  interpreter the engine-backed tests fail with setup instructions,
  while CI provisions a cached CPU-only venv (and checkpoints) so every
  pull request runs the full suite

## v0.3.0 (2026-09-20)

### Breaking Changes

- the package has been migrated from Python to TypeScript and is now
  published as `system-one-adapter` on npm; the Python package is
  discontinued
- the Python sync client is gone: JavaScript is asynchronous, so
  `SystemOneAdapterClient.systemOne` is `async` and there is no separate
  async client
- `systemOne` takes one request object
  (`{ state, questions, provider, model, retry }`) like the JavaScript
  SDK, instead of positional arguments
- responses are plain objects with `toJSON()` instead of Pydantic models
  with `model_dump()`

### Features

- backed by `@typesafe-ai/sdk` question factories and error classes
- provider error translation maps OpenAI/Anthropic SDK failures onto the
  TypeSafe error classes, preserving status, body, and cancellation
- own retry loop implementing the SDK `RetryPolicy` semantics (status
  sets, timeout/connection flags, exponential backoff with jitter,
  `Retry-After`)
- `await using` disposal via `Symbol.asyncDispose`
- all runtime type validation now runs through `arktype`: question
  collections, model output, provider response payloads, and constructor
  options are validated with arktype schemas whose inferences replace
  the previously hand-written guards
- provider request parameters are built against the OpenAI and Anthropic
  SDK parameter types directly, removing the unsafe casts that
  previously bridged them
- validated model answers flow into responses as typed SDK answer
  shapes, so responses no longer rely on double casts; a single
  documented assertion projects the validated answers onto the caller’s
  inferred question types
- test helpers validate recorded JSON via arktype as well, removing
  `as never` and `as unknown as` assertions from the suite

## v0.2.0 (2026-09-18)

### Breaking Changes

- ser/de library has been changed from `msgspec` to `pydantic` as
  `typesafe-sdk` did in `v0.7.0`

### Features

- support `typesafe-sdk>=0.7.0`

## v0.1.5 (2026-09-18)

### Bug fixes

- constrained `typesafe-sdk` version to `>=0.6.0,<0.7.0`

## v0.1.4 (2026-09-16)

### Bug fixes

- reuse providers and close owned SDK clients. Thanks
  [@AbdelStark](https://github.com/AbdelStark)!

## v0.1.3 (2026-09-15)

Initial release.
