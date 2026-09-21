# System One Adapter

A drop-in replacement for `@typesafe-ai/sdk`’s `systemOne` evaluation
API, backed by LLM APIs instead of TypeSafe.

Useful for comparing TypeSafe against an LLM on cost/speed/intelligence.

## Install

``` bash
npm install system-one-adapter
```

Both provider SDKs ship as dependencies: OpenAI-compatible endpoints and
native Anthropic.

## Usage

Unlike `TypeSafeClient`, the client is configured with how the LLM
should answer, and each call names a `provider` alongside the `model`:

``` ts
import { noul } from "@typesafe-ai/sdk";
import { SystemOneAdapterClient } from "system-one-adapter";

const client = new SystemOneAdapterClient({
  structuredOutputs: true, // use the provider's native structured-output mode
  llmAnswerMode: "probabilities", // or "discrete"
  normalizeProbabilities: true,
});

const response = await client.systemOne({
  state: "This book was a delight to read.",
  questions: { positive: noul("The book review is positive.") },
  provider: "openai", // "openai", "anthropic", or "laya"
  model: "gpt-4o-mini",
});
```

`provider` and `model` may also be set on the constructor as defaults.
`provider` is required unless `model` is a provider instance (e.g. a
custom OpenAI-compatible endpoint):

``` ts
import { OpenAIProvider } from "system-one-adapter";

const response = await client.systemOne({
  state,
  questions,
  model: new OpenAIProvider("grok-4", { baseUrl: "https://api.x.ai/v1" }),
});
```

Release the client with `await client.close()`, or use it with
`await using` (Node 24+). The adapter closes providers it creates;
provider instances passed as `model` remain caller-owned.

OpenAI’s endpoint uses the Responses API, with strict JSON Schema for
structured output and JSON mode for prompted output. Custom endpoints
(including `OPENAI_BASE_URL`) default to Chat Completions. Pass
`api: "responses"` or `api: "chat_completions"` to `OpenAIProvider` to
select explicitly, for example when using an OpenAI proxy. Responses are
requested with `store: false`; corrective retries send the conversation
history with each request.

For larger Anthropic evaluations, configure the output token limit on
the provider (default: 4,096 tokens):

``` ts
import { AnthropicProvider } from "system-one-adapter";

const response = await client.systemOne({
  state,
  questions,
  model: new AnthropicProvider("claude-haiku-4-5", { maxTokens: 8192 }),
});
```

A response that reaches the limit throws a `TypeSafeError` with
instructions to increase `maxTokens` or request fewer questions; it does
not consume malformed-output retries.

### Local decision engine: laya

For evaluations that need no text generation at all, the adapter ships a
provider for [laya](https://github.com/NandhaKishorM/laya), a local
System 1 decision engine that answers typed questions (`choice`,
`score`, `noul`) with calibrated probabilities in a single forward pass:

``` ts
const client = new SystemOneAdapterClient({
  structuredOutputs: true, // ignored by laya; answers are always typed
  llmAnswerMode: "probabilities",
  provider: "laya",
  model: "router", // "router" | "english" | "multilingual" | "typed-decisions"
});
```

Or construct a `LayaProvider` directly to override the python
interpreter:

``` ts
import { LayaProvider } from "system-one-adapter";

const response = await client.systemOne({
  state,
  questions,
  model: new LayaProvider("router", { python: "/usr/bin/python3.12" }),
});
```

laya runs as a local python package (`pip install laya`); each request
spawns one short-lived `python3` process (override with the
`LAYA_PYTHON` environment variable or the `python` option). The first
run downloads the checkpoints from the Hugging Face hub. Because laya is
a local encoder, token counts stay zero: latency is reported while cost
columns stay excluded. Each provider request carries the validated
questions in `ProviderRequestOptions.typed`, so laya evaluates the same
typed questions an LLM provider is prompted with — including score
questions, which laya answers natively.

### Response

The response mirrors the SDK’s `SystemOneResult` — same `answers` and
typed views (`nouls`, `scores`, `choices`) — with two additions:

- `response.usage` adds `input_tokens_total` / `output_tokens_total`
  (across retries), `n_retries`, `n_retries_malformed_structure`, and
  `latency`.
- `response.debug` holds `llm_attempts`, `retry_reasons`, and
  probability-normalization diagnostics.

`llm_attempts` records every provider call in order, including transient
failures and malformed responses. Each entry contains a snapshot of
`messages`, `model_request_parameters` (`schema` and `structured`),
`llm_response`, and `debug_info` with the model, provider, and any
error. The built-in providers also include the exact SDK `request`
arguments, the full provider response in `llm_response`, and the API and
finish reason in `debug_info`. Custom providers return their text and
token counts in `llm_response`. Calls that fail before returning a model
response leave it as `null`. Terminal `TypeSafeError` exceptions expose
the same attempt history in `error.debug`.

To replay an attempt through the same configured provider:

``` ts
import { type Message, type Provider } from "system-one-adapter";

const attempt = response.debug.llm_attempts[0]!;
const result = await provider.request(
  attempt.messages.map((message) => ({ ...message }) as Message),
  attempt.model_request_parameters as Parameters<Provider["request"]>[1],
);
```

The response object is plain data; serialize it with
`JSON.stringify(response)` (it implements `toJSON`):

``` ts
console.log(JSON.stringify(response));
```

## Options

| Option | Meaning |
|----|----|
| `structuredOutputs` | Use the provider’s native structured output, else prompt for JSON and validate client-side (works with any chat model). |
| `llmAnswerMode` | `"probabilities"` (per-label distribution) or `"discrete"` (one value per question). |
| `normalizeProbabilities` | Rescale invalid LLM probability distributions to sum to 1. |
| `nRetryMalformedStructure` | Corrective retries when the model’s output fails schema validation. |
| `retry` | Partial `RetryPolicy` from `@typesafe-ai/sdk` for transient provider failures; unset fields use the SDK defaults with retries off. |

The transient retry count and time budget apply separately to each
provider request. Corrective requests share the evaluation’s
`nRetryMalformedStructure` allowance and preserve the earlier responses
and correction messages.

## Development

``` bash
pnpm install
pnpm build    # type-check
pnpm test     # run tests (coverage gated at 80%)
pnpm lint     # format + lint everything
```

See [AGENTS.md](AGENTS.md) for the toolchain and coding conventions.

## Testing

The suite is entirely integration tests: every evaluation runs the real
client, provider, and provider SDK, and [MSW](https://mswjs.io)
intercepts the outgoing HTTP traffic. No test mocks, spies on, or stubs
any function — provider behavior is exercised through real requests and
responses, so request construction, response parsing, error translation,
retries, and the debug traces are verified end to end. Unhandled
requests are rejected, so a test that triggers unintended network
traffic fails.

The laya tests go one step further and run the real decision engine:
every laya evaluation spawns the provider’s python one-shot for real and
answers with the actual `convaiinnovations/laya` checkpoints, which are
downloaded from the Hugging Face hub on the first run (expect the first
test to take a few minutes while they fetch). Point the suite at an
interpreter with laya installed — a repo-local venv is picked up
automatically (it lives under `node_modules/.cache` so the repo tooling
ignores it):

``` bash
uv venv node_modules/.cache/laya-venv
uv pip install --python node_modules/.cache/laya-venv laya \
  --torch-backend=cpu
```

`pip install laya` into any `python3` works too, as does exporting
`LAYA_PYTHON`. Without one, the engine-backed tests are skipped with
setup instructions.
