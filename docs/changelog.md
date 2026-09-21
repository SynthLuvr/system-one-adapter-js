# Changelog

## Unreleased

### Features

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
