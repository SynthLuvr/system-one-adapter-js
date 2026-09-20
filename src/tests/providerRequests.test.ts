import { TypeSafeError } from "@typesafe-ai/sdk";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import {
  AnthropicProvider,
  anthropicResult,
  requestParams,
} from "../providers/anthropic.js";
import type { Message } from "../providers/base.js";
import {
  chatResult,
  responseFormat,
  responsesResult,
} from "../providers/openai.js";
import {
  asRecord,
  asRecords,
  asString,
  bodyText,
  debugOf,
} from "./testRecords.js";

const SCHEMA = { type: "object", properties: { answers: { type: "object" } } };
const requestPath = (input: RequestInfo | URL): string =>
  typeof input === "string"
    ? input
    : input instanceof URL
      ? input.href
      : input.url;

const MESSAGES: Message[] = [
  { role: "system", content: "system prompt" },
  { role: "user", content: "the document" },
];
const QUESTIONS = {
  positive: { type: "noul", instructions: "The review is positive." },
} as const;

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("provider request building and parsing", () => {
  it("wraps the schema in a native chat response format", () => {
    expect(responseFormat({ schema: SCHEMA, structured: true })).toEqual({
      type: "json_schema",
      json_schema: { name: "evaluation", schema: SCHEMA, strict: true },
    });
  });

  it("sends no chat response format when prompted", () => {
    expect(
      responseFormat({ schema: SCHEMA, structured: false }),
    ).toBeUndefined();
  });

  it("reads chat completion content and usage", () => {
    const result = chatResult({
      choices: [
        { message: { content: '{"answers": {}}' }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 12, completion_tokens: 3 },
    });
    expect(result.text).toBe('{"answers": {}}');
    expect([result.inputTokens, result.outputTokens]).toEqual([12, 3]);
  });

  it("rejects responses payloads that do not match the expected shape", () => {
    expect(() =>
      responsesResult({
        status: 42,
        output_text: "x",
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
    ).toThrowError(/expected shape/);
  });

  it("rejects completed responses without valid usage", () => {
    expect(() =>
      responsesResult({ status: "completed", output_text: "x" }),
    ).toThrowError(/usage/);
  });

  it("rejects chat payloads that do not match the expected shape", () => {
    expect(() =>
      chatResult({ choices: [{ message: { content: "x" } }] }),
    ).toThrowError(/expected shape/);
  });

  it("rejects anthropic payloads that do not match the expected shape", () => {
    expect(() =>
      anthropicResult({
        content: [],
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
    ).toThrowError(/expected shape/);
  });

  it("puts the schema in the anthropic output config when structured", () => {
    const params = requestParams(
      "claude-haiku-4-5",
      MESSAGES,
      {
        schema: SCHEMA,
        structured: true,
      },
      4096,
    );
    expect(params.system).toBe("system prompt");
    expect(params.messages).toEqual([
      { role: "user", content: "the document" },
    ]);
    expect(params.max_tokens).toBeGreaterThan(0);
    expect(params.output_config).toEqual({
      format: { type: "json_schema", schema: SCHEMA },
    });
  });

  it("omits the anthropic output config when prompted", () => {
    const params = requestParams(
      "claude-haiku-4-5",
      MESSAGES,
      {
        schema: SCHEMA,
        structured: false,
      },
      4096,
    );
    expect("output_config" in params).toBe(false);
  });

  it("joins anthropic text blocks and reads usage", () => {
    const result = anthropicResult({
      stop_reason: "end_turn",
      content: [
        { type: "text", text: '{"answers":' },
        { type: "thinking", text: "ignored" },
        { type: "text", text: " {}}" },
      ],
      usage: { input_tokens: 20, output_tokens: 5 },
    });
    expect(result.text).toBe('{"answers": {}}');
    expect([result.inputTokens, result.outputTokens]).toEqual([20, 5]);
  });

  it.each([false, true])(
    "rejects truncated anthropic output without consuming retries (structured: %s)",
    async (structured) => {
      for (const truncated of [false, true]) {
        vi.stubEnv("ANTHROPIC_API_KEY", "test-key");
        const requests: Record<string, unknown>[] = [];
        const fetch = async (
          input: RequestInfo | URL,
          init?: RequestInit,
        ): Promise<Response> => {
          expect(new URL(requestPath(input)).pathname).toBe("/v1/messages");
          requests.push(JSON.parse(bodyText(init)));
          return new Response(
            JSON.stringify({
              id: "msg-test",
              type: "message",
              role: "assistant",
              model: "claude-haiku-4-5",
              stop_reason: truncated ? "max_tokens" : "end_turn",
              stop_sequence: null,
              // Even syntactically valid JSON must not hide a truncated generation.
              content: [
                { type: "text", text: '{"answers":{"positive":true}}' },
              ],
              usage: { input_tokens: 20, output_tokens: truncated ? 8192 : 10 },
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          );
        };
        const provider = new AnthropicProvider("claude-haiku-4-5", {
          maxTokens: 8192,
          fetch,
        });
        const client = new SystemOneAdapterClient({
          structuredOutputs: structured,
          llmAnswerMode: "discrete",
          nRetryMalformedStructure: 2,
          retry: { maxRetries: 2, backoffInitialMs: 0, backoffJitter: 0 },
          model: provider,
        });
        let debug: Record<string, unknown> | undefined;

        if (truncated) {
          const error = await client
            .systemOne({ state: "A delightful book.", questions: QUESTIONS })
            .catch((caught: unknown) => caught);
          expect(error).toBeInstanceOf(TypeSafeError);
          if (!(error instanceof TypeSafeError))
            throw new Error("expected a TypeSafeError");
          expect(error.message).toMatch(/truncated.*Increase max_tokens/);
          debug = debugOf(error);
        } else {
          const response = await client.systemOne({
            state: "A delightful book.",
            questions: QUESTIONS,
          });
          expect(response.nouls.positive?.noul).toBe(1);
          debug = response.debug;
        }
        expect(requests.length).toBe(1);
        expect(requests[0].max_tokens).toBe(8192);
        if (debug === undefined) throw new Error("debug missing");
        const attempts = asRecords(debug.llm_attempts);
        expect(attempts.length).toBe(1);
        expect(attempts[0].request).toEqual(requests[0]);
        const response = asRecord(attempts[0].llm_response);
        expect(asString(asRecords(response.content)[0].text)).toBe(
          '{"answers":{"positive":true}}',
        );
        const debugInfo = asRecord(attempts[0].debug_info);
        expect(debugInfo.finish_reason).toBe(
          truncated ? "max_tokens" : "end_turn",
        );
        expect("error" in debugInfo).toBe(truncated);
      }
    },
  );

  it.each([0, -1])("rejects a non-positive output limit (%d)", (maxTokens) => {
    expect(
      () => new AnthropicProvider("claude-haiku-4-5", { maxTokens }),
    ).toThrow("max_tokens must be > 0");
  });
});
