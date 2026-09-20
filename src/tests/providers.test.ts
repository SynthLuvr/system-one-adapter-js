import { TypeSafeError } from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import { type OpenAIApi, OpenAIProvider } from "../providers/openai.js";
import {
  anthropicEndpoint,
  anthropicPayload,
  openAIChatEndpoint,
  openAIChatPayload,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  server,
} from "./msw.js";

const QUESTIONS = {
  positive: { type: "noul", instructions: "The review is positive." },
} as const;
const ANSWER = (answers: Record<string, unknown>): string =>
  JSON.stringify({ answers });

const noop = (): void => undefined;

/** An OpenAI provider for one transport, built like production code would. */
const openAIProvider = (
  options: { baseUrl?: string; api?: OpenAIApi } = {},
): OpenAIProvider =>
  new OpenAIProvider("test-model", { apiKey: "test-key", ...options });

describe("openai transports", () => {
  it.each([
    ["default endpoint", undefined, undefined, "responses"],
    ["custom endpoint", "https://compatible.test/v1", undefined, "chat"],
    ["explicit responses", "https://proxy.test/v1", "responses", "responses"],
    ["explicit chat", undefined, "chat_completions", "chat"],
  ] as [string, string | undefined, OpenAIApi | undefined, string][])(
    "carries evaluations end to end through %s",
    async (_name, baseUrl, api, transport) => {
      const responses = openAIResponsesEndpoint(() =>
        openAIResponsesPayload(ANSWER({ positive: true })),
      );
      const chat = openAIChatEndpoint(() =>
        openAIChatPayload(ANSWER({ positive: true })),
      );
      server.use(responses.handler, chat.handler);
      const client = new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: openAIProvider({ baseUrl, api }),
      });

      const response = await client.systemOne({
        state: "A delightful book.",
        questions: QUESTIONS,
      });

      expect(response.nouls.positive?.noul).toBe(1);
      const served =
        transport === "responses" ? responses.requests : chat.requests;
      expect(served.length).toBe(1);
      expect(response.debug.llm_attempts[0].debug_info.api).toBe(
        transport === "responses" ? "responses" : "chat_completions",
      );
    },
  );

  it.each([false, true])(
    "builds structured chat requests only when asked (structured: %s)",
    async (structured) => {
      const chat = openAIChatEndpoint(() =>
        openAIChatPayload(ANSWER({ positive: true })),
      );
      server.use(chat.handler);
      await new SystemOneAdapterClient({
        structuredOutputs: structured,
        llmAnswerMode: "discrete",
        model: openAIProvider({ baseUrl: "https://compatible.test/v1" }),
      }).systemOne({ state: "A book.", questions: QUESTIONS });

      const body = chat.requests[0];
      const messages = body.messages as { role: string }[];
      expect(body.model).toBe("test-model");
      expect(messages[0].role).toBe("system");
      expect(body.stream).toBeUndefined();
      if (structured) {
        expect(
          (body.response_format as { json_schema: { strict: boolean } })
            .json_schema.strict,
        ).toBe(true);
        expect(
          (body.response_format as { json_schema: { name: string } })
            .json_schema.name,
        ).toBe("evaluation");
      } else expect(body.response_format).toBeUndefined();
    },
  );

  it.each([false, true])(
    "builds structured responses requests only when asked (structured: %s)",
    async (structured) => {
      const responses = openAIResponsesEndpoint(() =>
        openAIResponsesPayload(ANSWER({ positive: true })),
      );
      server.use(responses.handler);
      await new SystemOneAdapterClient({
        structuredOutputs: structured,
        llmAnswerMode: "discrete",
        model: openAIProvider(),
      }).systemOne({ state: "A book.", questions: QUESTIONS });

      const body = responses.requests[0];
      expect(body.store).toBe(false);
      expect("previous_response_id" in body).toBe(false);
      const input = body.input as { role: string; content: string }[];
      const instructions = body.instructions as string | undefined;
      expect(instructions ?? input[0].content).toMatch(
        /^Evaluate every question/,
      );
      if (!structured) expect(JSON.stringify(body.input)).toContain("JSON");
      const outputFormat = (
        body.text as { format: { type: string; strict?: boolean } }
      ).format;
      expect(outputFormat.type).toBe(
        structured ? "json_schema" : "json_object",
      );
      if (structured) {
        expect(outputFormat.strict).toBe(true);
        const schema = (
          body.text as {
            format: {
              schema: {
                $defs: Record<string, { properties: Record<string, unknown> }>;
              };
            };
          }
        ).format.schema;
        expect("positive" in schema.$defs.TypeSafeAnswers.properties).toBe(
          true,
        );
      }
      expect(input[0].role).toBe(structured ? "user" : "system");
    },
  );

  it.each([
    [
      "a failed response reports its error message",
      {
        status: "failed",
        error: { code: "server_error", message: "generation failed" },
      },
      "generation failed",
    ],
    [
      "an incomplete response reports its reason",
      {
        status: "incomplete",
        incomplete_details: { reason: "max_output_tokens" },
      },
      "max_output_tokens",
    ],
    [
      "a failed response without a message reports its status",
      { status: "failed", error: {} },
      "failed",
    ],
  ] as [string, Record<string, unknown>, string][])(
    "does not treat %s as an answer",
    async (_name, overrides, reason) => {
      const responses = openAIResponsesEndpoint(() =>
        openAIResponsesPayload(ANSWER({ positive: true }), overrides),
      );
      server.use(responses.handler);
      const client = new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: openAIProvider(),
      });

      const error = (await client
        .systemOne({ state: "A book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught)) as TypeSafeError & {
        debug?: { llm_attempts: { debug_info: Record<string, unknown> }[] };
      };

      expect(error).toBeInstanceOf(TypeSafeError);
      expect(error.message).toContain(reason);
      expect(error.debug?.llm_attempts.length).toBe(1);
      expect(error.debug?.llm_attempts[0].debug_info.error).toContain(reason);
    },
  );

  it("reads chat content and usage, treating null content as empty", async () => {
    const chat = openAIChatEndpoint((_body, index) =>
      index === 0
        ? openAIChatPayload(null)
        : openAIChatPayload(ANSWER({ positive: true })),
    );
    server.use(chat.handler);
    const client = new SystemOneAdapterClient({
      structuredOutputs: false,
      llmAnswerMode: "discrete",
      nRetryMalformedStructure: 1,
      model: openAIProvider({ baseUrl: "https://compatible.test/v1" }),
    });

    const response = await client.systemOne({
      state: "A book.",
      questions: QUESTIONS,
    });

    const attempts = response.debug.llm_attempts;
    expect(attempts.length).toBe(2);
    expect(
      (attempts[0].llm_response as { choices: { message: unknown }[] })
        .choices[0].message,
    ).toMatchObject({ content: null });
    expect(attempts[0].debug_info.finish_reason).toBe("stop");
    expect(response.usage.input_tokens).toBe(12);
    expect(response.usage.output_tokens).toBe(7);
  });

  it("defaults a custom endpoint from the environment to chat", async () => {
    const previous = process.env.OPENAI_BASE_URL;
    process.env.OPENAI_BASE_URL = "https://compatible.test/v1";
    const chat = openAIChatEndpoint(() =>
      openAIChatPayload(ANSWER({ positive: true })),
    );
    server.use(chat.handler);
    try {
      const provider = new OpenAIProvider("test-model", {
        apiKey: "test-key",
      });
      expect(provider.api).toBe("chat_completions");
      const response = await new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: provider,
      }).systemOne({ state: "A book.", questions: QUESTIONS });
      expect(response.nouls.positive?.noul).toBe(1);
      expect(chat.requests.length).toBe(1);
    } finally {
      if (previous === undefined) delete process.env.OPENAI_BASE_URL;
      else process.env.OPENAI_BASE_URL = previous;
    }
  });

  it("rejects an unknown transport selection", () => {
    expect(() => openAIProvider({ api: "bogus" as OpenAIApi })).toThrow(
      "api must be 'responses' or 'chat_completions'",
    );
  });

  it.each(["completed", "incomplete", "failed"])(
    "isolates concurrent attempts and preserves failed responses (%s)",
    async (firstStatus) => {
      let arrived = 0;
      let release: () => void = noop;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const endpoint = openAIResponsesEndpoint(async (body) => {
        arrived += 1;
        if (arrived >= 2) release();
        await gate;
        const document = (body.input as { content: string }[])[0].content;
        const status = document.includes("first document")
          ? firstStatus
          : "completed";
        return openAIResponsesPayload(ANSWER({ positive: true }), {
          status,
          error:
            status === "failed"
              ? { code: "server_error", message: "generation failed" }
              : null,
          incomplete_details:
            status === "incomplete" ? { reason: "max_output_tokens" } : null,
        });
      });
      server.use(endpoint.handler);
      const provider = openAIProvider();
      const client = new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: provider,
      });

      const evaluate = async (
        document: string,
      ): Promise<Record<string, unknown>> => {
        try {
          return (
            await client.systemOne({ state: document, questions: QUESTIONS })
          ).debug as unknown as Record<string, unknown>;
        } catch (error) {
          return (error as { debug?: Record<string, unknown> }).debug ?? {};
        }
      };
      const [first, second] = await Promise.all([
        evaluate("first document"),
        evaluate("second document"),
      ]);
      for (const [debug, document, status] of [
        [first, "first document", firstStatus],
        [second, "second document", "completed"],
      ] as const) {
        const attempts = debug.llm_attempts as {
          messages: { content: string }[];
          request: { input: { content: string }[] };
          llm_response: { status: string };
          debug_info: { finish_reason: string; error?: string };
        }[];
        expect(attempts.length).toBe(1);
        const attempt = attempts[0];
        expect(attempt.messages[1].content).toContain(document);
        expect(attempt.request.input[0].content).toContain(document);
        expect(endpoint.requests).toEqual(
          expect.arrayContaining([attempt.request]),
        );
        expect(attempt.llm_response.status).toBe(status);
        expect(attempt.debug_info.finish_reason).toBe(status);
        expect("error" in attempt.debug_info).toBe(status !== "completed");
      }

      // A later direct provider call must not mutate either finished trace.
      const before = JSON.stringify([first, second]);
      const attempt = (
        second.llm_attempts as {
          messages: { role: string; content: string }[];
          model_request_parameters: Record<string, unknown>;
        }[]
      )[0];
      await provider.request(
        attempt.messages.map((message) => ({
          ...message,
          role: message.role as "system" | "user" | "assistant",
        })),
        attempt.model_request_parameters as never,
      );
      expect(JSON.stringify([first, second])).toBe(before);
    },
  );
});

describe("anthropic messages transport", () => {
  const anthropicProvider = (
    options: { maxTokens?: number } = {},
  ): AnthropicProvider =>
    new AnthropicProvider("claude-haiku-4-5", {
      apiKey: "test-key",
      ...options,
    });

  it.each([false, true])(
    "builds structured messages requests only when asked (structured: %s)",
    async (structured) => {
      const messages = anthropicEndpoint(() =>
        anthropicPayload(ANSWER({ positive: true })),
      );
      server.use(messages.handler);
      await new SystemOneAdapterClient({
        structuredOutputs: structured,
        llmAnswerMode: "discrete",
        model: anthropicProvider(),
      }).systemOne({ state: "A book.", questions: QUESTIONS });

      const body = messages.requests[0];
      expect(body.model).toBe("claude-haiku-4-5");
      expect(body.max_tokens).toBe(4096);
      expect(body.system).toMatch(/^Evaluate every question/);
      expect(body.messages).toEqual([
        { role: "user", content: expect.stringContaining("A book.") },
      ]);
      if (structured)
        expect(body.output_config).toEqual({
          format: {
            type: "json_schema",
            schema: expect.objectContaining({ type: "object" }),
          },
        });
      else expect("output_config" in body).toBe(false);
    },
  );

  it("honors a custom output limit and joins text blocks", async () => {
    const messages = anthropicEndpoint(() =>
      anthropicPayload('{"answers":', {
        content: [
          { type: "text", text: '{"answers":' },
          { type: "thinking", text: "ignored reasoning" },
          { type: "text", text: ' {"positive":true}}' },
        ],
      }),
    );
    server.use(messages.handler);
    const response = await new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
      model: anthropicProvider({ maxTokens: 8192 }),
    }).systemOne({ state: "A book.", questions: QUESTIONS });

    expect(messages.requests[0].max_tokens).toBe(8192);
    expect(response.nouls.positive?.noul).toBe(1);
    expect(response.usage.input_tokens).toBe(20);
    expect(response.usage.output_tokens).toBe(10);
  });

  it.each([false, true])(
    "rejects truncated output without consuming retries (structured: %s)",
    async (structured) => {
      const messages = anthropicEndpoint(() =>
        anthropicPayload(ANSWER({ positive: true }), {
          stop_reason: "max_tokens",
          usage: { input_tokens: 20, output_tokens: 8192 },
        }),
      );
      server.use(messages.handler);
      const client = new SystemOneAdapterClient({
        structuredOutputs: structured,
        llmAnswerMode: "discrete",
        nRetryMalformedStructure: 2,
        retry: { maxRetries: 2, backoffInitialMs: 0, backoffJitter: 0 },
        model: anthropicProvider({ maxTokens: 8192 }),
      });

      const error = (await client
        .systemOne({ state: "A book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught)) as TypeSafeError & {
        debug?: {
          llm_attempts: {
            request: Record<string, unknown>;
            llm_response: { content: { text: string }[] };
            debug_info: { finish_reason: string; error?: string };
          }[];
        };
      };

      expect(error).toBeInstanceOf(TypeSafeError);
      expect(error.message).toMatch(/truncated.*Increase max_tokens/);
      expect(messages.requests.length).toBe(1);
      const attempts = error.debug?.llm_attempts ?? [];
      expect(attempts.length).toBe(1);
      expect(attempts[0].request).toEqual(messages.requests[0]);
      // Even syntactically valid JSON must not hide a truncated generation.
      expect(attempts[0].llm_response.content[0].text).toBe(
        ANSWER({ positive: true }),
      );
      expect(attempts[0].debug_info.finish_reason).toBe("max_tokens");
      expect("error" in attempts[0].debug_info).toBe(true);
    },
  );

  it.each([0, -1])("rejects a non-positive output limit (%d)", (maxTokens) => {
    expect(() => anthropicProvider({ maxTokens })).toThrow(
      "max_tokens must be > 0",
    );
  });
});
