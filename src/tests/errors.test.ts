import anthropic from "@anthropic-ai/sdk";
import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  APIUserAbortError,
  AuthenticationError,
  BadRequestError,
  InternalServerError,
  PermissionDeniedError,
  RateLimitError,
  TypeSafeError,
} from "@typesafe-ai/sdk";
import openai from "openai";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import {
  AnthropicProvider,
  translateAnthropicError,
} from "../providers/anthropic.js";
import type { Provider } from "../providers/index.js";
import { OpenAIProvider, translateOpenAIError } from "../providers/openai.js";
import {
  ANSWER,
  anthropicEndpoint,
  jsonResponseError,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  QUESTIONS,
  server,
} from "./msw.js";

const STATUS_CASES = [
  { status: 400, expected: BadRequestError },
  { status: 401, expected: AuthenticationError },
  { status: 403, expected: PermissionDeniedError },
  { status: 429, expected: RateLimitError },
  { status: 500, expected: InternalServerError },
  { status: 418, expected: APIError },
] as const;

const PROVIDERS = [
  {
    provider: "openai",
    build: (baseUrl?: string): Provider =>
      new OpenAIProvider(
        "test-model",
        baseUrl === undefined
          ? { apiKey: "test-key" }
          : { apiKey: "test-key", baseUrl, api: "chat_completions" },
      ),
    endpoint: openAIResponsesEndpoint,
    translate: translateOpenAIError,
    timeoutError: () => new openai.APIConnectionTimeoutError(),
    userAbortError: () => new openai.APIUserAbortError(),
    connectionError: () => new openai.APIConnectionError({ message: "boom" }),
    statusError: (status: number) =>
      new openai.APIError(
        status,
        { error: { message: "boom" } },
        "error",
        new Headers(),
      ),
  },
  {
    provider: "anthropic",
    build: (baseUrl?: string): Provider =>
      new AnthropicProvider(
        "test-model",
        baseUrl === undefined
          ? { apiKey: "test-key" }
          : { apiKey: "test-key", baseUrl },
      ),
    endpoint: anthropicEndpoint,
    translate: translateAnthropicError,
    timeoutError: () => new anthropic.APIConnectionTimeoutError(),
    userAbortError: () => new anthropic.APIUserAbortError(),
    connectionError: () =>
      new anthropic.APIConnectionError({ message: "boom" }),
    statusError: (status: number) =>
      new anthropic.APIError(
        status,
        { error: { message: "boom" } },
        "error",
        new Headers(),
      ),
  },
] as const;

describe("provider error translation over HTTP", () => {
  it.each(PROVIDERS)(
    "$provider maps HTTP status errors and preserves status and body",
    async ({ build, endpoint }) => {
      for (const { status, expected } of STATUS_CASES) {
        const recorded = endpoint(() => jsonResponseError(status, "boom"));
        server.use(recorded.handler);
        const error = await new SystemOneAdapterClient({
          structuredOutputs: true,
          llmAnswerMode: "discrete",
          model: build(),
        })
          .systemOne({ state: "A book.", questions: QUESTIONS })
          .catch((caught: unknown) => caught);

        expect(error, `status ${status}`).toBeInstanceOf(expected);
        expect(error).toBeInstanceOf(APIError);
        const apiError = error as APIError;
        expect(apiError.status).toBe(status);
        expect(JSON.stringify(apiError.body)).toContain("boom");
        expect(recorded.requests.length).toBe(1);
      }
    },
  );

  it.each(PROVIDERS)(
    "$provider maps transport failures to connection errors with a cause",
    async ({ build }) => {
      // An unresolvable origin fails before any handler could reply; no
      // request is intercepted, so the SDK sees a genuine transport failure.
      const error = await new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: build("https://unintercepted.invalid/v1"),
      })
        .systemOne({ state: "A book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught);

      expect(error).toBeInstanceOf(APIConnectionError);
      expect((error as Error).cause).toBeInstanceOf(Error);
    },
  );

  it("keeps provider errors distinguishable on the debug trace", async () => {
    const endpoint = openAIResponsesEndpoint((_body, index) =>
      index === 0
        ? jsonResponseError(503)
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
    const [reason] = response.debug.retry_reasons;
    expect(reason[0]).toBe("provider_error");
    expect(reason[1]).toContain("503");
  });
});

describe("provider error translators", () => {
  it.each(PROVIDERS)(
    "$provider maps its SDK error classes onto SDK errors",
    ({
      build,
      translate,
      timeoutError,
      userAbortError,
      connectionError,
      statusError,
    }) => {
      expect(translate(timeoutError())).toBeInstanceOf(APITimeoutError);
      expect(translate(userAbortError())).toBeInstanceOf(APIUserAbortError);
      expect(translate(connectionError())).toBeInstanceOf(APIConnectionError);
      expect(translate(statusError(429))).toBeInstanceOf(RateLimitError);
      expect((translate(statusError(429)) as APIError).status).toBe(429);
      expect(translate(new Error("weird"))).toBeInstanceOf(TypeSafeError);
      expect(translate(new Error("weird"))).not.toBeInstanceOf(APIError);
      const alreadyMapped = APIError.fromResponse(
        400,
        "already mapped",
        new Headers(),
      );
      expect(translate(alreadyMapped)).toBe(alreadyMapped);

      // The same mapping is exposed as a method on every provider instance.
      const provider = build();
      expect(provider.translateError(timeoutError())).toBeInstanceOf(
        APITimeoutError,
      );
      expect(provider.translateError(statusError(429))).toBeInstanceOf(
        RateLimitError,
      );
    },
  );
});
