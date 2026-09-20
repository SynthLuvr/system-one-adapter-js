import anthropic from "@anthropic-ai/sdk";
import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  AuthenticationError,
  BadRequestError,
  InternalServerError,
  PermissionDeniedError,
  RateLimitError,
  TypeSafeError,
} from "@typesafe-ai/sdk";
import openai from "openai";
import { describe, expect, it } from "vitest";
import { translateAnthropicError } from "../providers/anthropic.js";
import { translateOpenAIError } from "../providers/openai.js";
import {
  type RetryReason,
  resolveRetryPolicy,
  runWithRetries,
  translating,
} from "../utils/errorHandling.js";

process.env.OPENAI_API_KEY = "test";
process.env.ANTHROPIC_API_KEY = "test";

const STATUS_CASES = [
  { status: 400, expected: BadRequestError },
  { status: 401, expected: AuthenticationError },
  { status: 403, expected: PermissionDeniedError },
  { status: 429, expected: RateLimitError },
  { status: 500, expected: InternalServerError },
  { status: 418, expected: APIError },
] as const;

const openAIStatusError = (
  status: number,
  body: Record<string, unknown>,
): unknown => new openai.APIError(status, body, "error", new Headers());

const anthropicStatusError = (
  status: number,
  body: Record<string, unknown>,
): unknown => new anthropic.APIError(status, body, "error", new Headers());

const PROVIDERS = [
  {
    provider: "openai",
    translate: translateOpenAIError,
    statusError: openAIStatusError,
    timeoutError: () => new openai.APIConnectionTimeoutError(),
    connectionError: () => new openai.APIConnectionError({ message: "boom" }),
  },
  {
    provider: "anthropic",
    translate: translateAnthropicError,
    statusError: anthropicStatusError,
    timeoutError: () => new anthropic.APIConnectionTimeoutError(),
    connectionError: () =>
      new anthropic.APIConnectionError({ message: "boom" }),
  },
] as const;

describe("translateError", () => {
  it.each(PROVIDERS)(
    "$provider maps status errors and preserves status and body",
    ({ translate, statusError }) => {
      const body = { error: { message: "boom" } };
      for (const { status, expected } of STATUS_CASES) {
        const translated = translate(statusError(status, body));
        expect(translated, `status ${status}`).toBeInstanceOf(expected);
        expect(translated).toBeInstanceOf(APIError);
        if (!(translated instanceof APIError))
          throw new Error("expected an APIError");
        expect(translated.status).toBe(status);
        expect(translated.body).toEqual(body);
      }
    },
  );

  it.each(PROVIDERS)(
    "$provider maps timeout and connection errors",
    ({ translate, timeoutError, connectionError }) => {
      expect(translate(timeoutError())).toBeInstanceOf(APITimeoutError);
      expect(translate(connectionError())).toBeInstanceOf(APIConnectionError);
    },
  );

  it.each(PROVIDERS)(
    "$provider passes through SDK errors and wraps unknown errors",
    ({ translate }) => {
      const sdkError = APIError.fromResponse(
        400,
        "already mapped",
        new Headers(),
      );
      expect(translate(sdkError)).toBe(sdkError);
      const wrapped = translate(new Error("weird"));
      expect(wrapped).toBeInstanceOf(TypeSafeError);
      expect(wrapped).not.toBeInstanceOf(APIError);
    },
  );
});

describe("translating", () => {
  it("re-raises the translated error with the original as cause", async () => {
    const original = new openai.APIError(429, { m: 1 }, "error", new Headers());
    const fail = (): Promise<never> => Promise.reject(original);

    await expect(translating(fail, translateOpenAIError)).rejects.toMatchObject(
      { status: 429 },
    );

    let caught: unknown;
    try {
      await translating(fail, translateOpenAIError);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(RateLimitError);
    if (!(caught instanceof Error)) throw new Error("expected an Error");
    expect(caught.cause).toBe(original);
  });
});

describe("runWithRetries", () => {
  it("retries transient errors until success", async () => {
    let calls = 0;
    const failOnce = async (): Promise<string> => {
      calls += 1;
      if (calls === 1)
        throw translateOpenAIError(
          openAIStatusError(503, { m: "unavailable" }),
        );
      return "success";
    };

    const outcome = await runWithRetries(
      failOnce,
      resolveRetryPolicy({
        maxRetries: 1,
        backoffInitialMs: 1,
        backoffJitter: 0,
      }),
    );

    expect(outcome.result).toBe("success");
    expect(calls).toBe(2);
    expect(outcome.nRetries).toBe(1);
  });

  it("does not retry non-retryable errors", async () => {
    let calls = 0;
    const raiseBadRequest = async (): Promise<never> => {
      calls += 1;
      throw translateOpenAIError(openAIStatusError(400, { m: "bad" }));
    };

    await expect(
      runWithRetries(
        raiseBadRequest,
        resolveRetryPolicy({
          maxRetries: 2,
          backoffInitialMs: 1,
          backoffJitter: 0,
        }),
      ),
    ).rejects.toBeInstanceOf(BadRequestError);
    expect(calls).toBe(1);
  });

  it("exhausts retries and records reasons", async () => {
    const reasons: RetryReason[] = [];
    const alwaysFail = async (): Promise<never> => {
      throw translateOpenAIError(openAIStatusError(503, { m: "unavailable" }));
    };

    await expect(
      runWithRetries(
        alwaysFail,
        resolveRetryPolicy({
          maxRetries: 2,
          backoffInitialMs: 1,
          backoffJitter: 0,
        }),
        reasons,
      ),
    ).rejects.toBeInstanceOf(InternalServerError);

    expect(reasons.map((reason) => reason.category)).toEqual([
      "provider_error",
      "provider_error",
    ]);
  });
});
