import { InternalServerError } from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import type { Provider } from "../providers/index.js";
import { OpenAIProvider } from "../providers/openai.js";
import { type AdapterDebug } from "../response.js";
import {
  ANSWER,
  anthropicEndpoint,
  jsonResponseError,
  openAIChatEndpoint,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  QUESTIONS,
  server,
} from "./msw.js";

/** An endpoint and provider pair that always fails with a 503. */
const failingProvider = (
  name: "openai" | "anthropic",
): {
  provider: Provider;
  requests: Record<string, unknown>[];
} => {
  const endpoint =
    name === "openai"
      ? openAIChatEndpoint(() => jsonResponseError(503))
      : anthropicEndpoint(() => jsonResponseError(503));
  server.use(endpoint.handler);
  const provider =
    name === "openai"
      ? new OpenAIProvider("test-model", {
          apiKey: "test-key",
          api: "chat_completions",
        })
      : new AnthropicProvider("test-model", { apiKey: "test-key" });
  return { provider, requests: endpoint.requests };
};

describe("provider retry budgets", () => {
  it.each([
    ["openai", 0],
    ["openai", 1],
    ["anthropic", 0],
    ["anthropic", 1],
  ] as const)(
    "%s retry policy controls HTTP attempts (budget %d)",
    async (name, retryBudget) => {
      const { provider, requests } = failingProvider(name);
      const client = new SystemOneAdapterClient({
        structuredOutputs: false,
        llmAnswerMode: "discrete",
        model: provider,
        retry: {
          maxRetries: retryBudget,
          backoffInitialMs: 0,
          backoffJitter: 0,
        },
      });

      const error = await client
        .systemOne({ state: "A delightful book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught as InternalServerError);

      expect(error).toBeInstanceOf(InternalServerError);
      const debug = (error as InternalServerError & { debug?: AdapterDebug })
        .debug;
      const attempts = debug?.llm_attempts ?? [];
      expect(attempts.length).toBe(retryBudget + 1);
      expect(attempts.map((attempt) => attempt.request)).toEqual(requests);
      for (const attempt of attempts) {
        expect(attempt.llm_response).toBeNull();
        expect(attempt.debug_info.error_type).toBe("InternalServerError");
        expect(attempt.debug_info.error).toContain("unavailable");
      }
      expect(() => JSON.stringify(debug)).not.toThrow();
      expect(requests.length).toBe(retryBudget + 1);
    },
  );
});

describe("retry-after coordination", () => {
  it.each([
    ["a millisecond header", { "retry-after-ms": "20" }],
    ["a zero-second header", { "retry-after": "0" }],
    ["a past-date header", { "retry-after": new Date(0).toUTCString() }],
    ["an unparseable header", { "retry-after": "soon" }],
    ["a negative millisecond header", { "retry-after-ms": "-5" }],
  ] as [string, Record<string, string>][])(
    "recovers after %s and succeeds",
    async (_name, headers) => {
      const endpoint = openAIResponsesEndpoint((_body, index) =>
        index === 0
          ? jsonResponseError(429, "slow down", headers)
          : openAIResponsesPayload(ANSWER({ positive: true })),
      );
      server.use(endpoint.handler);
      const response = await new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: new OpenAIProvider("test-model", { apiKey: "test-key" }),
        retry: { maxRetries: 1, backoffInitialMs: 1, backoffJitter: 0 },
      }).systemOne({ state: "A book.", questions: QUESTIONS });

      expect(response.nouls.positive?.noul).toBe(1);
      expect(response.usage.n_retries).toBe(1);
      expect(response.debug.retry_reasons[0][1]).toContain("slow down");
    },
  );

  it("ignores a retry-after larger than the policy cap", async () => {
    const endpoint = openAIResponsesEndpoint((_body, index) =>
      index === 0
        ? jsonResponseError(429, "slow down", { "retry-after-ms": "999999" })
        : openAIResponsesPayload(ANSWER({ positive: true })),
    );
    server.use(endpoint.handler);
    const startedAt = performance.now();
    const response = await new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
      model: new OpenAIProvider("test-model", { apiKey: "test-key" }),
      retry: {
        maxRetries: 1,
        backoffInitialMs: 1,
        backoffJitter: 0,
        maxRetryAfterMs: 5,
      },
    }).systemOne({ state: "A book.", questions: QUESTIONS });

    expect(performance.now() - startedAt).toBeLessThan(1000);
    expect(response.nouls.positive?.noul).toBe(1);
  });

  it("does not retry non-retryable statuses", async () => {
    const endpoint = openAIResponsesEndpoint(() =>
      jsonResponseError(400, "bad request"),
    );
    server.use(endpoint.handler);
    await expect(
      new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: new OpenAIProvider("test-model", { apiKey: "test-key" }),
        retry: { maxRetries: 3, backoffInitialMs: 0, backoffJitter: 0 },
      }).systemOne({ state: "A book.", questions: QUESTIONS }),
    ).rejects.toThrow(/bad request/);
    expect(endpoint.requests.length).toBe(1);
  });
});
