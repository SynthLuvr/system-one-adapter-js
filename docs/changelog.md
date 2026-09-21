# Changelog

## Unreleased

### Features

- new `claude_code` provider evaluates through the Claude Code CLI: each
  request spawns
  `MAX_THINKING_TOKENS=0 claude -p --output-format json --model … --tools "" --no-session-persistence`,
  passing the answer schema to `--json-schema` in structured mode and
  the conversation on stdin; CLI API failures map to the SDK error
  classes with their HTTP status so the retry policy applies, hung
  processes abort with `APITimeoutError`, and input tokens include the
  CLI’s cache-write and cache-read tokens
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
