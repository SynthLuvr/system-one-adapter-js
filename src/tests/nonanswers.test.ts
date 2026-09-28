import { type Questions, TypeSafeError } from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import { type OpenAIApi, OpenAIProvider } from "../providers/openai.js";
import type { AdapterDebug, SystemOneResponse } from "../response.js";
import {
  ANSWER,
  anthropicEndpoint,
  anthropicPayload,
  openAIChatEndpoint,
  openAIChatPayload,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  QUESTIONS,
  server,
} from "./msw.js";

/** One evaluation's outcome: its response, or its terminal error. */
type Outcome<Q extends Questions> =
  | { response: SystemOneResponse<Q>; error?: undefined }
  | { response?: undefined; error: TypeSafeError & { debug?: AdapterDebug } };

/** Run one evaluation, capturing the response or the terminal error. */
const evaluate = async <Q extends Questions>(
  client: SystemOneAdapterClient,
  questions: Q,
): Promise<Outcome<Q>> => {
  try {
    return {
      response: await client.systemOne({
        state: "A delightful book.",
        questions,
      }),
    };
  } catch (caught) {
    return { error: caught as TypeSafeError & { debug?: AdapterDebug } };
  }
};

/** The empty debug payload of an outcome that unexpectedly carried none. */
const EMPTY_DEBUG: AdapterDebug = {
  max_error: 0,
  invalid_probs: 0,
  probability_errors: {},
  llm_attempts: [],
  retry_reasons: [],
};

/** The debug payload of one outcome, from its response or its error. */
const debugOf = (outcome: Outcome<Questions>): AdapterDebug =>
  outcome.response?.debug ?? outcome.error?.debug ?? EMPTY_DEBUG;

/** An evaluation client spending budgets a non-answer must not consume. */
const budgetedClient = (
  model: OpenAIProvider | AnthropicProvider,
  structured: boolean,
): SystemOneAdapterClient =>
  new SystemOneAdapterClient({
    structuredOutputs: structured,
    llmAnswerMode: "discrete",
    nRetryMalformedStructure: 2,
    retry: { maxRetries: 2, backoffInitialMs: 0, backoffJitter: 0 },
    model,
  });

describe("chat completion finish reasons", () => {
  it.each([
    "stop",
    null,
    "length",
    "content_filter",
    "tool_calls",
    "function_call",
    "unknown",
  ])(
    "treat finish_reason %s as an answer only when complete",
    async (reason) => {
      const content = ANSWER({ positive: true });
      const chat = openAIChatEndpoint(() =>
        openAIChatPayload(content, {
          choices: [
            {
              index: 0,
              message: { role: "assistant", content },
              finish_reason: reason,
            },
          ],
        }),
      );
      server.use(chat.handler);

      for (const structured of [false, true]) {
        const outcome = await evaluate(
          budgetedClient(
            new OpenAIProvider("test-model", {
              apiKey: "test-key",
              api: "chat_completions",
            }),
            structured,
          ),
          QUESTIONS,
        );
        const completed = reason === "stop" || reason === null;
        if (completed) expect(outcome.response?.nouls.positive?.noul).toBe(1);
        else
          expect(outcome.error?.message).toBe(
            `OpenAI chat completion did not complete: ${reason}.`,
          );

        const debug = debugOf(outcome);
        expect(chat.requests.length).toBe(structured ? 2 : 1);
        expect(debug.retry_reasons).toEqual([]);
        expect(debug.llm_attempts.length).toBe(1);
        const attempt = debug.llm_attempts[0];
        expect(attempt.request).toEqual(chat.requests.at(-1));
        const response = attempt.llm_response as {
          choices: {
            message: { content: string };
            finish_reason: string | null;
          }[];
        };
        expect(response.choices[0].message.content).toBe(content);
        expect(response.choices[0].finish_reason).toBe(reason);
        expect(attempt.debug_info.finish_reason).toBe(reason);
        expect("error" in attempt.debug_info).toBe(!completed);
        expect(() => JSON.stringify(debug)).not.toThrow();
      }
    },
  );
});

describe("openai missing token usage", () => {
  it.each(
    ["chat_completions", "responses"].flatMap((api) =>
      [
        "omitted",
        "null",
        "missing_input",
        "null_input",
        "missing_output",
        "null_output",
        "zero",
        "present",
      ].map((usageCase) => [api, usageCase] as const),
    ),
  )("%s still completes with %s usage", async (api, usageCase) => {
    const [inputField, outputField] =
      api === "responses"
        ? (["input_tokens", "output_tokens"] as const)
        : (["prompt_tokens", "completion_tokens"] as const);
    const usage: Record<string, number | null> = {
      [inputField]: 12,
      [outputField]: 7,
      total_tokens: 19,
    };
    if (usageCase === "missing_input") delete usage[inputField];
    if (usageCase === "missing_output") delete usage[outputField];
    if (usageCase === "null_input") usage[inputField] = null;
    if (usageCase === "null_output") usage[outputField] = null;
    if (usageCase === "zero") {
      usage[inputField] = 0;
      usage[outputField] = 0;
      usage.total_tokens = 0;
    }
    const provider = new OpenAIProvider("test-model", {
      apiKey: "test-key",
      ...(api === "responses" ? {} : { api: "chat_completions" as OpenAIApi }),
    });
    const payload =
      api === "responses"
        ? openAIResponsesPayload(
            ANSWER({ positive: true }),
            usageCase === "omitted"
              ? {}
              : { usage: usageCase === "null" ? null : usage },
          )
        : openAIChatPayload(
            ANSWER({ positive: true }),
            usageCase === "omitted"
              ? {}
              : { usage: usageCase === "null" ? null : usage },
          );
    // The payload helpers always carry a usage object; the omitted case
    // needs the key gone entirely.
    if (usageCase === "omitted") delete payload.usage;
    const endpoint =
      api === "responses"
        ? openAIResponsesEndpoint(() => payload)
        : openAIChatEndpoint(() => payload);
    server.use(endpoint.handler);
    const response = await budgetedClient(provider, true).systemOne({
      state: "A delightful book.",
      questions: QUESTIONS,
    });

    expect(response.nouls.positive?.noul).toBe(1);
    const expectedInput = [
      "omitted",
      "null",
      "missing_input",
      "null_input",
    ].includes(usageCase)
      ? null
      : usage[inputField];
    const expectedOutput = [
      "omitted",
      "null",
      "missing_output",
      "null_output",
    ].includes(usageCase)
      ? null
      : usage[outputField];
    expect(response.usage.input_tokens).toBe(expectedInput);
    expect(response.usage.input_tokens_total).toBe(expectedInput);
    expect(response.usage.output_tokens).toBe(expectedOutput);
    expect(response.usage.output_tokens_total).toBe(expectedOutput);
    expect(response.usage.n_retries).toBe(0);
    expect(response.usage.n_retries_malformed_structure).toBe(0);
    expect(response.toJSON().usage.input_tokens).toBe(expectedInput);
    expect(response.toJSON().usage.output_tokens).toBe(expectedOutput);
    expect(response.debug.retry_reasons).toEqual([]);
    expect(endpoint.requests.length).toBe(1);
    expect(response.debug.llm_attempts.length).toBe(1);
    const attempt = response.debug.llm_attempts[0];
    expect(attempt.request).toEqual(endpoint.requests[0]);
    const recorded = (attempt.llm_response as { usage: unknown }).usage;
    if (usageCase === "omitted" || usageCase === "null")
      expect(recorded).toBe(usageCase === "null" ? null : undefined);
    else
      expect(recorded).toMatchObject(
        Object.fromEntries(
          Object.entries(usage).filter(([, value]) => value !== null),
        ),
      );
    expect(attempt.debug_info.finish_reason).toBe(
      api === "responses" ? "completed" : "stop",
    );
    expect("error" in attempt.debug_info).toBe(false);
    expect(() => JSON.stringify(response.debug)).not.toThrow();
  });

  it("nulls cumulative totals once any attempt omits a count", async () => {
    const responses = openAIResponsesEndpoint((_body, index) =>
      index === 0
        ? openAIResponsesPayload("not json")
        : openAIResponsesPayload(ANSWER({ positive: true }), { usage: null }),
    );
    server.use(responses.handler);
    const response = await new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
      nRetryMalformedStructure: 1,
      model: new OpenAIProvider("test-model", { apiKey: "test-key" }),
    }).systemOne({ state: "A delightful book.", questions: QUESTIONS });

    expect(responses.requests.length).toBe(2);
    expect(response.usage.n_retries_malformed_structure).toBe(1);
    // The first attempt reported 12/7, but the final attempt reported
    // nothing, so both its counts and the cumulative totals are null.
    expect(response.usage.input_tokens).toBe(null);
    expect(response.usage.output_tokens).toBe(null);
    expect(response.usage.input_tokens_total).toBe(null);
    expect(response.usage.output_tokens_total).toBe(null);
  });
});

describe("anthropic stop reasons", () => {
  it.each([
    "end_turn",
    "stop_sequence",
    null,
    "refusal",
    "model_context_window_exceeded",
    "pause_turn",
    "tool_use",
    "unknown",
    "max_tokens",
  ])("treat stop_reason %s as an answer only when complete", async (reason) => {
    const content =
      reason === "refusal"
        ? []
        : [{ type: "text", text: ANSWER({ positive: true }) }];
    const messages = anthropicEndpoint(() =>
      anthropicPayload(ANSWER({ positive: true }), {
        stop_reason: reason,
        content,
      }),
    );
    server.use(messages.handler);
    const completed =
      reason === "end_turn" || reason === "stop_sequence" || reason === null;

    for (const structured of [false, true]) {
      const outcome = await evaluate(
        budgetedClient(
          new AnthropicProvider("claude-haiku-4-5", { apiKey: "test-key" }),
          structured,
        ),
        QUESTIONS,
      );
      if (completed) expect(outcome.response?.nouls.positive?.noul).toBe(1);
      else if (reason === "max_tokens")
        expect(outcome.error?.message).toMatch(
          /truncated.*Increase max_tokens/,
        );
      else
        expect(outcome.error?.message).toBe(
          `Anthropic response did not complete: ${reason}.`,
        );

      const debug = debugOf(outcome);
      expect(messages.requests.length).toBe(structured ? 2 : 1);
      expect(debug.retry_reasons).toEqual([]);
      expect(debug.llm_attempts.length).toBe(1);
      const attempt = debug.llm_attempts[0];
      expect(attempt.request).toEqual(messages.requests.at(-1));
      const response = attempt.llm_response as {
        content: unknown[];
        stop_reason: string | null;
      };
      expect(response.content.length).toBe(reason === "refusal" ? 0 : 1);
      expect(response.stop_reason).toBe(reason);
      expect(attempt.debug_info.finish_reason).toBe(reason);
      expect("error" in attempt.debug_info).toBe(!completed);
      expect(() => JSON.stringify(debug)).not.toThrow();
    }
  });
});

describe("openai responses refusals", () => {
  it.each([false, true])(
    "rejects a refusal even beside valid text (valid text: %s)",
    async (withValidText) => {
      const refusal = "Cannot evaluate this request.";
      const output: Record<string, unknown>[] = [
        { type: "reasoning", id: "reasoning-test", summary: [] },
      ];
      if (withValidText)
        output.push({
          id: "msg-text",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [
            {
              type: "output_text",
              text: ANSWER({ positive: true }),
              annotations: [],
            },
          ],
        });
      output.push({
        id: "msg-refusal",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "refusal", refusal }],
      });
      // Missing usage is allowed, but a refusal must still fail the
      // evaluation.
      const responses = openAIResponsesEndpoint(() =>
        openAIResponsesPayload(ANSWER({ positive: true }), {
          output,
          usage: null,
        }),
      );
      server.use(responses.handler);

      for (const structured of [false, true]) {
        const outcome = await evaluate(
          budgetedClient(
            new OpenAIProvider("test-model", { apiKey: "test-key" }),
            structured,
          ),
          QUESTIONS,
        );
        expect(outcome.error?.message).toBe(
          `OpenAI response was a refusal: ${refusal}`,
        );

        const debug = debugOf(outcome);
        expect(debug.retry_reasons).toEqual([]);
        expect(debug.llm_attempts.length).toBe(1);
        const attempt = debug.llm_attempts[0];
        expect(attempt.request).toEqual(responses.requests.at(-1));
        const lastMessage = (
          attempt.llm_response as {
            output: { content: { refusal?: string }[] }[];
          }
        ).output.at(-1);
        expect(lastMessage?.content[0]?.refusal).toBe(refusal);
        expect(attempt.debug_info.finish_reason).toBe("completed");
        expect(attempt.debug_info.error).toContain("refusal");
        expect(() => JSON.stringify(debug)).not.toThrow();
      }
      expect(responses.requests.length).toBe(2);
    },
  );
});
