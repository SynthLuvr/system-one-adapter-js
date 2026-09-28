import { type Questions, TypeSafeError } from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import type { LlmAttempt, Provider } from "../providers/base.js";
import { GeminiProvider } from "../providers/gemini.js";
import { OpenAIProvider } from "../providers/openai.js";
import type { AdapterDebug, SystemOneResponse } from "../response.js";
import {
  ANSWER,
  anthropicEndpoint,
  anthropicPayload,
  geminiEndpoint,
  geminiPayload,
  openAIChatEndpoint,
  openAIChatPayload,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  QUESTIONS,
  type RecordedEndpoint,
  server,
} from "./msw.js";

/** The document every evaluation in this suite evaluates. */
const STATE = "A delightful book.";

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
    return { response: await client.systemOne({ state: STATE, questions }) };
  } catch (caught) {
    return { error: caught as TypeSafeError & { debug?: AdapterDebug } };
  }
};

/** The debug payload of an outcome that unexpectedly carried none. */
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

/**
 * Assert the evaluation consumed no retry budgets and traced exactly one
 * attempt, recorded as the endpoint's latest request.
 */
const singleAttempt = (
  debug: AdapterDebug,
  endpoint: RecordedEndpoint,
): LlmAttempt => {
  expect(debug.retry_reasons).toEqual([]);
  expect(debug.llm_attempts.length).toBe(1);
  const attempt = debug.llm_attempts[0];
  expect(attempt.request).toEqual(endpoint.requests.at(-1));
  expect(() => JSON.stringify(debug)).not.toThrow();
  return attempt;
};

/** A Gemini provider against the recorded Interactions endpoint. */
const geminiProvider = (): GeminiProvider =>
  new GeminiProvider("gemini-3.8-flash", { apiKey: "test-key" });

/** A client whose retry budgets a non-answer must not consume. */
const budgetedClient = (
  model: Provider,
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
        openAIChatPayload("", {
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
      const provider = new OpenAIProvider("test-model", {
        apiKey: "test-key",
        api: "chat_completions",
      });

      for (const structured of [false, true]) {
        const outcome = await evaluate(
          budgetedClient(provider, structured),
          QUESTIONS,
        );
        const completed = reason === "stop" || reason === null;
        if (completed) expect(outcome.response?.nouls.positive?.noul).toBe(1);
        else
          expect(outcome.error?.message).toBe(
            `OpenAI chat completion did not complete: ${reason}.`,
          );
        expect(chat.requests.length).toBe(structured ? 2 : 1);

        const attempt = singleAttempt(debugOf(outcome), chat);
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
      }
    },
  );
});

describe("openai missing token usage", () => {
  /** The token-count fields of each API's usage object: input, then output. */
  const FIELDS = {
    chat_completions: ["prompt_tokens", "completion_tokens"],
    responses: ["input_tokens", "output_tokens"],
  } as const;

  it.each(
    (["chat_completions", "responses"] as const).flatMap((api) =>
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
    const [inputField, outputField] = FIELDS[api];
    const usage: Record<string, number | null> = {
      [inputField]: 12,
      [outputField]: 7,
    };
    if (usageCase === "missing_input") delete usage[inputField];
    if (usageCase === "missing_output") delete usage[outputField];
    if (usageCase === "null_input") usage[inputField] = null;
    if (usageCase === "null_output") usage[outputField] = null;
    if (usageCase === "zero") {
      usage[inputField] = 0;
      usage[outputField] = 0;
    }
    const payloadUsage =
      usageCase === "omitted" ? undefined : usageCase === "null" ? null : usage;
    const overrides = { usage: payloadUsage };
    const payload =
      api === "responses"
        ? openAIResponsesPayload(ANSWER({ positive: true }), overrides)
        : openAIChatPayload(ANSWER({ positive: true }), overrides);
    if (payloadUsage === undefined) delete payload.usage;
    const endpoint =
      api === "responses"
        ? openAIResponsesEndpoint(() => payload)
        : openAIChatEndpoint(() => payload);
    server.use(endpoint.handler);
    const response = await budgetedClient(
      new OpenAIProvider("test-model", { apiKey: "test-key", api }),
      true,
    ).systemOne({ state: STATE, questions: QUESTIONS });

    const expectedInput = payloadUsage?.[inputField] ?? null;
    const expectedOutput = payloadUsage?.[outputField] ?? null;
    expect(response.nouls.positive?.noul).toBe(1);
    expect(response.usage.input_tokens).toBe(expectedInput);
    expect(response.usage.input_tokens_total).toBe(expectedInput);
    expect(response.usage.output_tokens).toBe(expectedOutput);
    expect(response.usage.output_tokens_total).toBe(expectedOutput);
    expect(response.usage.n_retries).toBe(0);
    expect(response.usage.n_retries_malformed_structure).toBe(0);
    expect(response.toJSON().usage.input_tokens).toBe(expectedInput);
    expect(response.toJSON().usage.output_tokens).toBe(expectedOutput);
    expect(endpoint.requests.length).toBe(1);

    const attempt = singleAttempt(response.debug, endpoint);
    const recorded = (attempt.llm_response as { usage: unknown }).usage;
    if (payloadUsage == null) expect(recorded).toBe(payloadUsage);
    else
      expect(recorded).toMatchObject(
        Object.fromEntries(
          Object.entries(payloadUsage).filter(([, value]) => value !== null),
        ),
      );
    expect(attempt.debug_info.finish_reason).toBe(
      api === "responses" ? "completed" : "stop",
    );
    expect("error" in attempt.debug_info).toBe(false);
  });

  it("nulls cumulative totals once any attempt omits a count", async () => {
    const responses = openAIResponsesEndpoint((_body, index) =>
      index === 0
        ? openAIResponsesPayload("not json")
        : openAIResponsesPayload(ANSWER({ positive: true }), { usage: null }),
    );
    server.use(responses.handler);
    const response = await budgetedClient(
      new OpenAIProvider("test-model", { apiKey: "test-key" }),
      true,
    ).systemOne({ state: STATE, questions: QUESTIONS });

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
      anthropicPayload("", { stop_reason: reason, content }),
    );
    server.use(messages.handler);
    const provider = new AnthropicProvider("claude-haiku-4-5", {
      apiKey: "test-key",
    });
    const completed =
      reason === "end_turn" || reason === "stop_sequence" || reason === null;

    for (const structured of [false, true]) {
      const outcome = await evaluate(
        budgetedClient(provider, structured),
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
      expect(messages.requests.length).toBe(structured ? 2 : 1);

      const attempt = singleAttempt(debugOf(outcome), messages);
      const response = attempt.llm_response as {
        content: unknown[];
        stop_reason: string | null;
      };
      expect(response.content.length).toBe(reason === "refusal" ? 0 : 1);
      expect(response.stop_reason).toBe(reason);
      expect(attempt.debug_info.finish_reason).toBe(reason);
      expect("error" in attempt.debug_info).toBe(!completed);
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
      const provider = new OpenAIProvider("test-model", { apiKey: "test-key" });

      for (const structured of [false, true]) {
        const outcome = await evaluate(
          budgetedClient(provider, structured),
          QUESTIONS,
        );
        expect(outcome.error?.message).toBe(
          `OpenAI response was a refusal: ${refusal}`,
        );

        const attempt = singleAttempt(debugOf(outcome), responses);
        const lastMessage = (
          attempt.llm_response as {
            output: { content: { refusal?: string }[] }[];
          }
        ).output.at(-1);
        expect(lastMessage?.content[0]?.refusal).toBe(refusal);
        expect(attempt.debug_info.finish_reason).toBe("completed");
        expect(attempt.debug_info.error).toContain("refusal");
      }
      expect(responses.requests.length).toBe(2);
    },
  );
});

describe("gemini interaction statuses", () => {
  it.each([
    "completed",
    "in_progress",
    "requires_action",
    "failed",
    "cancelled",
    "incomplete",
    "budget_exceeded",
    "queued",
  ])("treat status %s as an answer only when completed", async (status) => {
    const interactions = geminiEndpoint(() =>
      geminiPayload(ANSWER({ positive: true }), { status }),
    );
    server.use(interactions.handler);
    const provider = geminiProvider();
    const completed = status === "completed";

    for (const structured of [false, true]) {
      const outcome = await evaluate(
        budgetedClient(provider, structured),
        QUESTIONS,
      );
      if (completed) expect(outcome.response?.nouls.positive?.noul).toBe(1);
      else
        expect(outcome.error?.message).toBe(
          `Gemini response did not complete: ${status}.`,
        );
      expect(interactions.requests.length).toBe(structured ? 2 : 1);

      const attempt = singleAttempt(debugOf(outcome), interactions);
      const response = attempt.llm_response as { status: string };
      expect(response.status).toBe(status);
      expect(attempt.debug_info.finish_reason).toBe(status);
      expect("error" in attempt.debug_info).toBe(!completed);
    }
  });

  it("reports an empty status as unknown", async () => {
    const interactions = geminiEndpoint(() =>
      geminiPayload(ANSWER({ positive: true }), { status: "" }),
    );
    server.use(interactions.handler);
    const outcome = await evaluate(
      budgetedClient(geminiProvider(), true),
      QUESTIONS,
    );

    expect(outcome.error?.message).toBe(
      "Gemini response did not complete: unknown.",
    );
    singleAttempt(debugOf(outcome), interactions);
  });

  it("reports recorded interaction errors as the failure reason", async () => {
    const errors = [{ code: "500", message: "generation failed" }];
    const interactions = geminiEndpoint(() =>
      geminiPayload(ANSWER({ positive: true }), { status: "failed", errors }),
    );
    server.use(interactions.handler);
    const outcome = await evaluate(
      budgetedClient(geminiProvider(), true),
      QUESTIONS,
    );

    expect(outcome.error?.message).toContain("did not complete");
    expect(outcome.error?.message).toContain("generation failed");
    const attempt = singleAttempt(debugOf(outcome), interactions);
    expect(attempt.llm_response).toMatchObject({ status: "failed", errors });
  });
});

describe("gemini missing token usage", () => {
  it.each(["omitted", "null", "missing_input", "missing_output"])(
    "rejects %s usage as an incomplete answer",
    async (usageCase) => {
      const usage: Record<string, number> = {
        total_input_tokens: 12,
        total_output_tokens: 7,
      };
      if (usageCase === "missing_input") delete usage.total_input_tokens;
      if (usageCase === "missing_output") delete usage.total_output_tokens;
      const payload = geminiPayload(ANSWER({ positive: true }));
      if (usageCase === "omitted") delete payload.usage;
      else payload.usage = usageCase === "null" ? null : usage;
      const interactions = geminiEndpoint(() => payload);
      server.use(interactions.handler);

      const outcome = await evaluate(
        budgetedClient(geminiProvider(), true),
        QUESTIONS,
      );

      expect(outcome.error?.message).toBe("Gemini response omitted usage.");
      expect(interactions.requests.length).toBe(1);
      singleAttempt(debugOf(outcome), interactions);
    },
  );

  it.each(["present", "zero"])(
    "preserves reported %s usage counts",
    async (usageCase) => {
      const usage = {
        total_input_tokens: usageCase === "zero" ? 0 : 12,
        total_output_tokens: usageCase === "zero" ? 0 : 7,
      };
      const interactions = geminiEndpoint(() =>
        geminiPayload(ANSWER({ positive: true }), { usage }),
      );
      server.use(interactions.handler);
      const response = await budgetedClient(geminiProvider(), true).systemOne({
        state: STATE,
        questions: QUESTIONS,
      });

      expect(response.nouls.positive?.noul).toBe(1);
      expect(response.usage.input_tokens).toBe(usage.total_input_tokens);
      expect(response.usage.output_tokens).toBe(usage.total_output_tokens);
      expect(response.usage.input_tokens_total).toBe(usage.total_input_tokens);
      expect(response.usage.output_tokens_total).toBe(
        usage.total_output_tokens,
      );
      expect(interactions.requests.length).toBe(1);
    },
  );
});
