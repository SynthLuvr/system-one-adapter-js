import { TypeSafeError } from "@typesafe-ai/sdk";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import {
  type OpenAIApi,
  OpenAIProvider,
  responsesResult,
} from "../providers/openai.js";

const QUESTIONS = {
  positive: { type: "noul", instructions: "The review is positive." },
} as const;
const MALFORMED = '{"answers":';

const noop = (): void => undefined;

const requestPath = (input: RequestInfo | URL): string =>
  typeof input === "string"
    ? input
    : input instanceof URL
      ? input.href
      : input.url;

interface RecordedTransport {
  fetch: typeof globalThis.fetch;
  requests: Record<string, unknown>[];
}

/** A fetch that records JSON request bodies and replies per endpoint. */
const transport = (
  endpoint: string,
  texts: () => string,
): RecordedTransport => {
  const requests: Record<string, unknown>[] = [];
  const fetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url = new URL(requestPath(input));
    expect(url.pathname).toBe(endpoint);
    requests.push(JSON.parse((init?.body as string | undefined) ?? ""));
    const text = texts();
    const payload =
      endpoint === "/v1/responses"
        ? {
            id: "resp-test",
            object: "response",
            created_at: 0,
            status: "completed",
            model: "test-model",
            text: (requests[requests.length - 1] as { text?: unknown }).text,
            output: [
              {
                id: "reasoning-test",
                type: "reasoning",
                summary: [{ type: "summary_text", text: "Ignored summary." }],
              },
              {
                type: "message",
                role: "assistant",
                status: "completed",
                content: [{ type: "output_text", text, annotations: [] }],
              },
            ],
            usage: { input_tokens: 12, output_tokens: 7, total_tokens: 19 },
          }
        : {
            id: "chatcmpl-test",
            object: "chat.completion",
            created: 0,
            model: "test-model",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: text },
                finish_reason: "stop",
              },
            ],
            usage: {
              prompt_tokens: 12,
              completion_tokens: 7,
              total_tokens: 19,
            },
          };
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
  return { fetch, requests };
};

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("openai transports", () => {
  it.each([
    ["default endpoint", undefined, undefined, "/v1/responses"],
    [
      "custom endpoint",
      "https://compatible.test/v1",
      undefined,
      "/v1/chat/completions",
    ],
    [
      "explicit responses",
      "https://proxy.test/v1",
      "responses",
      "/v1/responses",
    ],
    ["explicit chat", undefined, "chat_completions", "/v1/chat/completions"],
  ] as [string, string | undefined, OpenAIApi | undefined, string][])(
    "preserves corrections and usage through %s",
    async (_name, baseUrl, api, endpoint) => {
      vi.stubEnv("OPENAI_API_KEY", "test-key");
      const recorded = transport(endpoint, () =>
        recorded.requests.length === 1
          ? MALFORMED
          : '{"answers":{"positive":true}}',
      );
      for (const structured of [false, true]) {
        recorded.requests.length = 0;
        const provider = new OpenAIProvider("test-model", {
          baseUrl,
          api,
          fetch: recorded.fetch,
        });
        const client = new SystemOneAdapterClient({
          structuredOutputs: structured,
          llmAnswerMode: "discrete",
          nRetryMalformedStructure: 1,
          model: provider,
        });
        const response = await client.systemOne({
          state: "A delightful book.",
          questions: QUESTIONS,
        });

        expect(response.nouls.positive?.noul).toBe(1);
        expect([
          response.usage.input_tokens,
          response.usage.output_tokens,
          response.usage.input_tokens_total,
          response.usage.output_tokens_total,
          response.usage.n_retries,
          response.usage.n_retries_malformed_structure,
        ]).toEqual([12, 7, 24, 14, 0, 1]);

        const attempts = response.debug.llm_attempts;
        expect(attempts.map((attempt) => attempt.request)).toEqual(
          recorded.requests,
        );
        expect(attempts.map((attempt) => attempt.messages.length)).toEqual([
          2, 4,
        ]);
        for (const [index, attempt] of attempts.entries()) {
          const raw = attempt.llm_response as Record<string, never>;
          if (endpoint === "/v1/responses") {
            const requestFormat = (
              attempt.request as { text: { format: Record<string, unknown> } }
            ).text.format;
            expect(
              (raw.text as unknown as { format: Record<string, unknown> })
                .format,
            ).toEqual(requestFormat);
          }
          const text =
            endpoint === "/v1/responses"
              ? (raw.output as { content: { text: string }[] }[])[1].content[0]
                  .text
              : (raw.choices as { message: { content: string } }[])[0].message
                  .content;
          expect(text).toBe(
            index === 0 ? MALFORMED : '{"answers":{"positive":true}}',
          );
          expect(attempt.debug_info.api).toBe(
            endpoint === "/v1/responses" ? "responses" : "chat_completions",
          );
        }
        expect(() => JSON.stringify(response.debug)).not.toThrow();

        for (const body of recorded.requests)
          if (endpoint === "/v1/responses") {
            expect(body.store).toBe(false);
            expect("previous_response_id" in body).toBe(false);
            const input = body.input as { role: string; content: string }[];
            const instructions = body.instructions as string | undefined;
            expect(instructions ?? input[0].content).toMatch(
              /^Evaluate every question/,
            );
            if (!structured)
              expect(JSON.stringify(body.input)).toContain("JSON");
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
                      $defs: Record<
                        string,
                        { properties: Record<string, unknown> }
                      >;
                    };
                  };
                }
              ).format.schema;
              expect(
                "positive" in schema.$defs.TypeSafeAnswers.properties,
              ).toBe(true);
            }
            expect(input[0].role).toBe(structured ? "user" : "system");
          } else {
            const messages = body.messages as { role: string }[];
            expect(messages[0].role).toBe("system");
            if (structured)
              expect(
                (body.response_format as { json_schema: { strict: boolean } })
                  .json_schema.strict,
              ).toBe(true);
            else expect(body.response_format).toBeUndefined();
          }

        const lastMessages = (
          recorded.requests[recorded.requests.length - 1] as {
            input?: { role: string; content: string }[];
            messages?: { role: string; content: string }[];
          }
        )[endpoint === "/v1/responses" ? "input" : "messages"] as {
          role: string;
          content: string;
        }[];
        expect(lastMessages[lastMessages.length - 2]).toEqual({
          role: "assistant",
          content: MALFORMED,
        });
        expect(lastMessages[lastMessages.length - 1].role).toBe("user");
        expect(lastMessages[lastMessages.length - 1].content).toContain(
          "previous response did not match",
        );
      }
    },
  );

  it.each(["failed", "incomplete"])(
    "does not treat %s responses as answers",
    (status) => {
      const response = {
        status,
        error: status === "failed" ? { message: "generation failed" } : null,
        incomplete_details:
          status === "incomplete" ? { reason: "max_output_tokens" } : null,
        output_text: '{"answers":{"positive":true}}',
      };
      const reason =
        status === "failed" ? "generation failed" : "max_output_tokens";
      expect(() => responsesResult(response as never)).toThrowError(
        `OpenAI response did not complete: ${reason}.`,
      );
    },
  );

  it.each(["completed", "incomplete", "failed"])(
    "isolates concurrent attempts and preserves failed responses (%s)",
    async (firstStatus) => {
      vi.stubEnv("OPENAI_API_KEY", "test-key");
      const requests: Record<string, unknown>[] = [];
      let release: () => void = noop;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const fetch = async (
        _input: RequestInfo | URL,
        init?: RequestInit,
      ): Promise<Response> => {
        const body = JSON.parse((init?.body as string | undefined) ?? "") as {
          input: { content: string }[];
        };
        requests.push(body);
        if (requests.length >= 2) release();
        await gate;
        const document = body.input[0].content;
        const status = document.includes("first document")
          ? firstStatus
          : "completed";
        return new Response(
          JSON.stringify({
            id: "resp-test",
            object: "response",
            created_at: 0,
            status,
            model: "test-model",
            error:
              status === "failed"
                ? { code: "server_error", message: "generation failed" }
                : null,
            incomplete_details:
              status === "incomplete" ? { reason: "max_output_tokens" } : null,
            output: [
              {
                type: "message",
                role: "assistant",
                status: "completed",
                content: [
                  {
                    type: "output_text",
                    text: '{"answers":{"positive":true}}',
                    annotations: [],
                  },
                ],
              },
            ],
            usage: { input_tokens: 12, output_tokens: 7, total_tokens: 19 },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      };
      const provider = new OpenAIProvider("test-model", {
        api: "responses",
        fetch,
      });
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
        expect(requests).toEqual(expect.arrayContaining([attempt.request]));
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

  it("defaults a custom endpoint from the environment to chat", () => {
    vi.stubEnv("OPENAI_API_KEY", "test-key");
    vi.stubEnv("OPENAI_BASE_URL", "https://compatible.test/v1");
    const provider = new OpenAIProvider("test-model");
    expect(provider.api).toBe("chat_completions");
    expect(provider).toBeInstanceOf(OpenAIProvider);
    expect(TypeSafeError).toBeDefined();
  });
});
