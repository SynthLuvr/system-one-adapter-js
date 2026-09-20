import { InternalServerError } from "@typesafe-ai/sdk";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import type { Provider } from "../providers/base.js";
import { OpenAIProvider } from "../providers/openai.js";
import { asRecord, asRecords, bodyText, debugOf } from "./testRecords.js";

const QUESTIONS = {
  positive: { type: "noul", instructions: "The review is positive." },
} as const;

afterEach(() => {
  vi.unstubAllEnvs();
});

const makeProvider = (
  name: "openai" | "anthropic",
): { provider: Provider; requests: Record<string, unknown>[] } => {
  vi.stubEnv(
    name === "openai" ? "OPENAI_API_KEY" : "ANTHROPIC_API_KEY",
    "test-key",
  );
  const requests: Record<string, unknown>[] = [];
  const fetch = async (
    _input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    requests.push(JSON.parse(bodyText(init)));
    return new Response(JSON.stringify({ error: { message: "unavailable" } }), {
      status: 503,
      headers: { "content-type": "application/json" },
    });
  };
  const provider =
    name === "openai"
      ? new OpenAIProvider("test-model", { api: "chat_completions", fetch })
      : new AnthropicProvider("test-model", { fetch });
  return { provider, requests };
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
      const { provider, requests } = makeProvider(name);
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
        .catch((caught: unknown) => caught);

      expect(error).toBeInstanceOf(InternalServerError);
      const debug = debugOf(error);
      const attempts = asRecords(debug.llm_attempts);
      expect(attempts.length).toBe(retryBudget + 1);
      expect(attempts.map((attempt) => attempt.request)).toEqual(requests);
      for (const attempt of attempts) {
        expect(attempt.llm_response).toBeNull();
        expect(asRecord(attempt.debug_info).error_type).toBe(
          "InternalServerError",
        );
        expect(String(asRecord(attempt.debug_info).error)).toContain(
          "unavailable",
        );
      }
      expect(() => JSON.stringify(debug)).not.toThrow();
      expect(requests.length).toBe(retryBudget + 1);
    },
  );
});
