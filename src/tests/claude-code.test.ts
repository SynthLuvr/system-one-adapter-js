import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  TypeSafeError,
} from "@typesafe-ai/sdk";
import { type MockBinBehaviour, type MockBinHandle, mockBin } from "type-a-bin";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { ClaudeCodeProvider } from "../providers/claude-code.js";
import { ANSWER, QUESTIONS } from "./msw.js";

/**
 * How long a scripted first response waits before printing, holding the
 * provider waiting while the test installs the next scripted mock for
 * the retry its failure is about to trigger.
 */
const SWAP_WINDOW_MS = 500;

/** One scripted CLI result payload, printed as a single stdout line. */
const resultPayload = (overrides: Record<string, unknown> = {}): string =>
  JSON.stringify({
    type: "result",
    subtype: "success",
    is_error: false,
    stop_reason: "end_turn",
    result: ANSWER({ positive: true }),
    usage: {
      input_tokens: 3,
      output_tokens: 4,
      cache_creation_input_tokens: 100,
      cache_read_input_tokens: 200,
    },
    ...overrides,
  });

/** A provider running the `claude` the active mock shadows on `PATH`. */
const claudeCodeProvider = (
  options: {
    env?: Record<string, string | undefined>;
    timeoutMs?: number;
  } = {},
): ClaudeCodeProvider => new ClaudeCodeProvider("claude-haiku-4-5", options);

/** The value recorded after `flag` in one mocked invocation. */
const valueAfter = (args: readonly string[], flag: string): string => {
  const index = args.indexOf(flag);
  if (index === -1) throw new Error(`flag not recorded: ${flag}`);
  return args[index + 1] ?? "";
};

/**
 * The scripted `claude` mocks of one request flow. Each `next` tears the
 * previous mock down and installs the next behaviour, so responses
 * arrive in the order they were scripted; `done` releases the last one.
 */
const claudeScript = (): {
  next: (behaviour: MockBinBehaviour) => Promise<MockBinHandle>;
  done: () => void;
} => {
  let handle: MockBinHandle | undefined;
  return {
    next: async (behaviour) => {
      handle?.();
      handle = await mockBin("claude", behaviour);
      return handle;
    },
    done: () => handle?.(),
  };
};

describe("claude code CLI transport", () => {
  it("carries evaluations end to end over a CLI process", async () => {
    const claude = await mockBin("claude", {
      stdout: resultPayload(),
      record: { stdin: true },
    });
    try {
      const response = await new SystemOneAdapterClient({
        structuredOutputs: false,
        llmAnswerMode: "discrete",
        model: claudeCodeProvider(),
      }).systemOne({
        state: "A delightful book.",
        questions: QUESTIONS,
      });

      expect(response.nouls.positive?.noul).toBe(1);
      // Input tokens include the CLI's cache write and read.
      expect(response.usage.input_tokens).toBe(303);
      expect(response.usage.output_tokens).toBe(4);
      const [call] = claude.calls;
      expect(call.args).toEqual([
        "-p",
        "--output-format",
        "json",
        "--model",
        "claude-haiku-4-5",
        "--tools",
        "",
        "--no-session-persistence",
        "--system-prompt",
        expect.stringMatching(/^Evaluate every question/),
      ]);
      expect(call.args.includes("--json-schema")).toBe(false);
      expect(call.stdin?.startsWith("User:\n")).toBe(true);
      expect(call.stdin).toContain("A delightful book.");
      expect(call.env.MAX_THINKING_TOKENS).toBe("0");
      const attempt = response.debug.llm_attempts[0];
      expect(attempt.debug_info.provider).toBe("ClaudeCodeProvider");
      expect(attempt.debug_info.api).toBe("claude_code");
      expect(attempt.debug_info.finish_reason).toBe("end_turn");
      expect(attempt.request).toMatchObject({
        command: "claude",
        env: { MAX_THINKING_TOKENS: "0" },
      });
      expect((attempt.request as { prompt: string }).prompt).toBe(call.stdin);
      expect((attempt.llm_response as { is_error: boolean }).is_error).toBe(
        false,
      );
    } finally {
      claude();
    }
  });

  it.each([false, true])(
    "passes the answer schema to the CLI only in structured mode (structured: %s)",
    async (structured) => {
      const claude = await mockBin("claude", { stdout: resultPayload() });
      try {
        await new SystemOneAdapterClient({
          structuredOutputs: structured,
          llmAnswerMode: "discrete",
          model: claudeCodeProvider(),
        }).systemOne({ state: "A book.", questions: QUESTIONS });
        const [call] = claude.calls;
        if (structured) {
          const schema = JSON.parse(valueAfter(call.args, "--json-schema"));
          expect(schema.$defs.TypeSafeAnswers.properties).toHaveProperty(
            "positive",
          );
        } else expect(call.args.includes("--json-schema")).toBe(false);
      } finally {
        claude();
      }
    },
  );

  it("renders the corrective conversation for a retried request", async () => {
    const claude = claudeScript();
    try {
      const first = await claude.next({
        stdout: resultPayload({ result: "{not json" }),
        delayMs: SWAP_WINDOW_MS,
        record: { stdin: true },
      });
      const response = new SystemOneAdapterClient({
        structuredOutputs: false,
        llmAnswerMode: "discrete",
        nRetryMalformedStructure: 1,
        model: claudeCodeProvider(),
      }).systemOne({ state: "A book.", questions: QUESTIONS });
      // The malformed answer only prints once the retry's mock is in
      // place, so the retried process cannot race the swap.
      await first.waitForCall();
      const second = await claude.next({
        stdout: resultPayload(),
        record: { stdin: true },
      });
      const answered = await response;

      expect(answered.nouls.positive?.noul).toBe(1);
      expect(answered.usage.n_retries_malformed_structure).toBe(1);
      expect(answered.usage.input_tokens_total).toBe(606);
      expect(first.calls[0]?.stdin?.includes("Assistant:")).toBe(false);
      expect(second.calls[0]?.stdin).toContain("Assistant:\n{not json");
      expect(second.calls[0]?.stdin).toContain(
        "did not match the required schema",
      );
    } finally {
      claude.done();
    }
  });

  it("retries a rate-limited CLI result through the retry policy", async () => {
    const claude = claudeScript();
    try {
      const first = await claude.next({
        stdout: resultPayload({
          is_error: true,
          api_error_status: 429,
          result: "rate limited",
          stop_reason: null,
        }),
        delayMs: SWAP_WINDOW_MS,
      });
      const response = new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        retry: { maxRetries: 1, backoffInitialMs: 0, backoffJitter: 0 },
        model: claudeCodeProvider(),
      }).systemOne({ state: "A book.", questions: QUESTIONS });
      await first.waitForCall();
      const second = await claude.next({ stdout: resultPayload() });
      const answered = await response;

      expect(answered.nouls.positive?.noul).toBe(1);
      expect(answered.usage.n_retries).toBe(1);
      expect(answered.debug.retry_reasons).toEqual([
        ["provider_error", expect.stringContaining("rate limited")],
      ]);
      expect(first.calls.length + second.calls.length).toBe(2);
    } finally {
      claude.done();
    }
  });

  it("maps an API error to its status without retrying it", async () => {
    const claude = await mockBin("claude", {
      stdout: resultPayload({
        is_error: true,
        api_error_status: 404,
        result: "no such model",
        stop_reason: null,
      }),
    });
    try {
      const error = (await new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        retry: { maxRetries: 2, backoffInitialMs: 0, backoffJitter: 0 },
        model: claudeCodeProvider(),
      })
        .systemOne({ state: "A book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught)) as APIError;

      expect(error).toBeInstanceOf(APIError);
      expect(error.status).toBe(404);
      expect(
        (
          error as unknown as {
            debug: { llm_attempts: { debug_info: { error: string } }[] };
          }
        ).debug.llm_attempts[0].debug_info.error,
      ).toContain("no such model");
      expect(claude.calls.length).toBe(1);
    } finally {
      claude();
    }
  });

  it("raises a TypeSafeError for CLI errors without an API status", async () => {
    const claude = await mockBin("claude", {
      stdout: resultPayload({
        is_error: true,
        subtype: "error_during_execution",
        result: "hooks failed",
      }),
    });
    try {
      const error = (await new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        retry: { maxRetries: 2, backoffInitialMs: 0, backoffJitter: 0 },
        model: claudeCodeProvider(),
      })
        .systemOne({ state: "A book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught)) as TypeSafeError;

      expect(error).toBeInstanceOf(TypeSafeError);
      expect(error).not.toBeInstanceOf(APIError);
      expect(error.message).toContain("hooks failed");
      expect(claude.calls.length).toBe(1);
    } finally {
      claude();
    }
  });

  it("retries a crashed CLI process as a connection error", async () => {
    const claude = claudeScript();
    try {
      const first = await claude.next({
        stderr: "boom",
        exitCode: 3,
        delayMs: SWAP_WINDOW_MS,
      });
      const response = new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        retry: { maxRetries: 1, backoffInitialMs: 0, backoffJitter: 0 },
        model: claudeCodeProvider(),
      }).systemOne({ state: "A book.", questions: QUESTIONS });
      await first.waitForCall();
      await claude.next({ stdout: resultPayload() });
      const answered = await response;

      expect(answered.nouls.positive?.noul).toBe(1);
      expect(answered.usage.n_retries).toBe(1);
      expect(answered.debug.retry_reasons[0]).toEqual([
        "provider_error",
        expect.stringContaining("boom"),
      ]);
    } finally {
      claude.done();
    }
  });

  it("raises a TypeSafeError when the CLI prints no JSON", async () => {
    const claude = await mockBin("claude", { stdout: "definitely not JSON" });
    try {
      const error = (await new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: claudeCodeProvider(),
      })
        .systemOne({ state: "A book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught)) as TypeSafeError;

      expect(error).toBeInstanceOf(TypeSafeError);
      expect(error.message).toContain("did not print a JSON result");
    } finally {
      claude();
    }
  });

  it("aborts a hung CLI process with a timeout error", async () => {
    const claude = await mockBin("claude", {
      stdout: resultPayload(),
      delayMs: 5000,
    });
    try {
      const error = (await new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "discrete",
        model: claudeCodeProvider({ timeoutMs: 50 }),
      })
        .systemOne({ state: "A book.", questions: QUESTIONS })
        .catch((caught: unknown) => caught)) as APITimeoutError;

      expect(error).toBeInstanceOf(APITimeoutError);
    } finally {
      claude();
    }
  });

  it("reports a missing CLI as a connection error", async () => {
    const provider = new ClaudeCodeProvider("claude-haiku-4-5", {
      command: "claude-missing-for-tests",
    });

    const error = (await provider
      .request([{ role: "user", content: "hi" }], {
        schema: {},
        structured: false,
      })
      .catch((caught: unknown) => caught)) as APIConnectionError;

    expect(error).toBeInstanceOf(APIConnectionError);
    expect(error.message).toContain("claude-missing-for-tests");
  });

  it("requests directly without a system prompt and releases cleanly", async () => {
    const claude = await mockBin("claude", {
      stdout: resultPayload(),
      record: { stdin: true },
    });
    try {
      const provider = claudeCodeProvider();
      const result = await provider.request(
        [{ role: "user", content: "hello" }],
        { schema: {}, structured: false },
      );

      expect(result.text).toBe('{"answers":{"positive":true}}');
      expect(result.inputTokens).toBe(303);
      expect(result.outputTokens).toBe(4);
      const [call] = claude.calls;
      expect(call.args.includes("--system-prompt")).toBe(false);
      expect(call.stdin).toBe("User:\nhello");
      expect(provider.translateError(new Error("boom"))).toBeInstanceOf(
        TypeSafeError,
      );
      provider.close();
    } finally {
      claude();
    }
  });

  it("validates options and applies env overrides and removals", async () => {
    expect(() => claudeCodeProvider({ timeoutMs: 0 })).toThrow(
      "timeout_ms must be > 0",
    );
    expect(new ClaudeCodeProvider("claude-haiku-4-5").command).toBe("claude");
    const claude = await mockBin("claude", { stdout: resultPayload() });
    try {
      const overridden = claudeCodeProvider({
        env: { MAX_THINKING_TOKENS: "4096" },
      });
      const removed = claudeCodeProvider({
        env: { MAX_THINKING_TOKENS: undefined, ANTHROPIC_API_KEY: undefined },
      });
      const options = { schema: {}, structured: false } as const;
      await overridden.request([{ role: "user", content: "hi" }], options);
      await removed.request([{ role: "user", content: "hi" }], options);

      const [overriding, removing] = claude.calls;
      expect(overriding.env.MAX_THINKING_TOKENS).toBe("4096");
      expect(removing.env.MAX_THINKING_TOKENS).toBeUndefined();
    } finally {
      claude();
    }
  });
});
