import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  TypeSafeError,
} from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { ClaudeCodeProvider } from "../providers/claude-code.js";
import { QUESTIONS } from "./msw.js";

/** The fake `claude` CLI, run through the current Node binary. */
const FIXTURE = new URL("./fake-claude.mjs", import.meta.url).pathname;

/** One recorded fake-CLI invocation. */
interface FakeCall {
  argv: string[];
  stdin: string;
  env: { MAX_THINKING_TOKENS?: string };
}

/** A throwaway directory holding the per-test invocation log. */
const workspace = async (): Promise<{
  log: string;
  calls: () => Promise<FakeCall[]>;
  cleanup: () => Promise<void>;
}> => {
  const log = join(await mkdtemp(join(tmpdir(), "claude-code-")), "calls.log");
  const calls = async (): Promise<FakeCall[]> => {
    const text = await readFile(log, "utf8").catch(() => "");
    return text
      .split("\n")
      .filter((line) => line !== "")
      .map((line) => JSON.parse(line) as FakeCall);
  };
  const cleanup = (): Promise<void> =>
    rm(join(log, ".."), { recursive: true, force: true });
  return { log, calls, cleanup };
};

/** Run `fn` with fake-CLI environment variables, restoring them after. */
const withFakeEnv = async <T>(
  env: Record<string, string>,
  fn: () => Promise<T>,
): Promise<T> => {
  const previous = Object.fromEntries(
    Object.keys(env).map((key) => [key, process.env[key]]),
  );
  Object.assign(process.env, env);
  try {
    return await fn();
  } finally {
    for (const [key, value] of Object.entries(previous))
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
  }
};

/** A provider whose CLI is the fixture running under the current Node. */
const claudeCodeProvider = (
  options: {
    env?: Record<string, string | undefined>;
    timeoutMs?: number;
  } = {},
): ClaudeCodeProvider =>
  new ClaudeCodeProvider("claude-haiku-4-5", {
    command: process.execPath,
    args: [FIXTURE],
    ...options,
  });

/** The value recorded after `flag` in one fake invocation. */
const valueAfter = (argv: string[], flag: string): string => {
  const index = argv.indexOf(flag);
  if (index === -1) throw new Error(`flag not recorded: ${flag}`);
  return argv[index + 1] ?? "";
};

describe("claude code CLI transport", () => {
  it("carries evaluations end to end over a CLI process", async () => {
    const space = await workspace();
    try {
      const response = await withFakeEnv({ FAKE_CLAUDE_LOG: space.log }, () =>
        new SystemOneAdapterClient({
          structuredOutputs: false,
          llmAnswerMode: "discrete",
          model: claudeCodeProvider(),
        }).systemOne({
          state: "A delightful book.",
          questions: QUESTIONS,
        }),
      );

      expect(response.nouls.positive?.noul).toBe(1);
      // Input tokens include the CLI's cache write and read.
      expect(response.usage.input_tokens).toBe(303);
      expect(response.usage.output_tokens).toBe(4);
      const [call] = await space.calls();
      expect(call.argv).toEqual([
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
      expect(call.argv.includes("--json-schema")).toBe(false);
      expect(call.stdin.startsWith("User:\n")).toBe(true);
      expect(call.stdin).toContain("A delightful book.");
      expect(call.env.MAX_THINKING_TOKENS).toBe("0");
      const attempt = response.debug.llm_attempts[0];
      expect(attempt.debug_info.provider).toBe("ClaudeCodeProvider");
      expect(attempt.debug_info.api).toBe("claude_code");
      expect(attempt.debug_info.finish_reason).toBe("end_turn");
      expect(attempt.request).toMatchObject({
        command: process.execPath,
        env: { MAX_THINKING_TOKENS: "0" },
      });
      expect((attempt.request as { prompt: string }).prompt).toBe(call.stdin);
      expect((attempt.llm_response as { is_error: boolean }).is_error).toBe(
        false,
      );
    } finally {
      await space.cleanup();
    }
  });

  it.each([false, true])(
    "passes the answer schema to the CLI only in structured mode (structured: %s)",
    async (structured) => {
      const space = await workspace();
      try {
        await withFakeEnv({ FAKE_CLAUDE_LOG: space.log }, () =>
          new SystemOneAdapterClient({
            structuredOutputs: structured,
            llmAnswerMode: "discrete",
            model: claudeCodeProvider(),
          }).systemOne({ state: "A book.", questions: QUESTIONS }),
        );
        const [call] = await space.calls();
        if (structured) {
          const schema = JSON.parse(valueAfter(call.argv, "--json-schema"));
          expect(schema.$defs.TypeSafeAnswers.properties).toHaveProperty(
            "positive",
          );
        } else expect(call.argv.includes("--json-schema")).toBe(false);
      } finally {
        await space.cleanup();
      }
    },
  );

  it("renders the corrective conversation for a retried request", async () => {
    const space = await workspace();
    try {
      const response = await withFakeEnv(
        {
          FAKE_CLAUDE_LOG: space.log,
          FAKE_CLAUDE_SCRIPT: JSON.stringify([{ result: "{not json" }, {}]),
        },
        () =>
          new SystemOneAdapterClient({
            structuredOutputs: false,
            llmAnswerMode: "discrete",
            nRetryMalformedStructure: 1,
            model: claudeCodeProvider(),
          }).systemOne({ state: "A book.", questions: QUESTIONS }),
      );

      expect(response.nouls.positive?.noul).toBe(1);
      expect(response.usage.n_retries_malformed_structure).toBe(1);
      expect(response.usage.input_tokens_total).toBe(606);
      const [first, second] = await space.calls();
      expect(first.stdin.includes("Assistant:")).toBe(false);
      expect(second.stdin).toContain("Assistant:\n{not json");
      expect(second.stdin).toContain("did not match the required schema");
    } finally {
      await space.cleanup();
    }
  });

  it("retries a rate-limited CLI result through the retry policy", async () => {
    const space = await workspace();
    try {
      const response = await withFakeEnv(
        {
          FAKE_CLAUDE_LOG: space.log,
          FAKE_CLAUDE_SCRIPT: JSON.stringify([
            {
              is_error: true,
              api_error_status: 429,
              result: "rate limited",
              stop_reason: null,
            },
            {},
          ]),
        },
        () =>
          new SystemOneAdapterClient({
            structuredOutputs: true,
            llmAnswerMode: "discrete",
            retry: { maxRetries: 1, backoffInitialMs: 0, backoffJitter: 0 },
            model: claudeCodeProvider(),
          }).systemOne({ state: "A book.", questions: QUESTIONS }),
      );

      expect(response.nouls.positive?.noul).toBe(1);
      expect(response.usage.n_retries).toBe(1);
      expect(response.debug.retry_reasons).toEqual([
        ["provider_error", expect.stringContaining("rate limited")],
      ]);
      expect((await space.calls()).length).toBe(2);
    } finally {
      await space.cleanup();
    }
  });

  it("maps an API error to its status without retrying it", async () => {
    const space = await workspace();
    try {
      const error = (await withFakeEnv(
        {
          FAKE_CLAUDE_LOG: space.log,
          FAKE_CLAUDE_SCRIPT: JSON.stringify([
            { is_error: true, api_error_status: 404, result: "no such model" },
          ]),
        },
        () =>
          new SystemOneAdapterClient({
            structuredOutputs: true,
            llmAnswerMode: "discrete",
            retry: { maxRetries: 2, backoffInitialMs: 0, backoffJitter: 0 },
            model: claudeCodeProvider(),
          })
            .systemOne({ state: "A book.", questions: QUESTIONS })
            .catch((caught: unknown) => caught),
      )) as APIError;

      expect(error).toBeInstanceOf(APIError);
      expect(error.status).toBe(404);
      expect(
        (
          error as unknown as {
            debug: { llm_attempts: { debug_info: { error: string } }[] };
          }
        ).debug.llm_attempts[0].debug_info.error,
      ).toContain("no such model");
      expect((await space.calls()).length).toBe(1);
    } finally {
      await space.cleanup();
    }
  });

  it("raises a TypeSafeError for CLI errors without an API status", async () => {
    const space = await workspace();
    try {
      const error = (await withFakeEnv(
        {
          FAKE_CLAUDE_LOG: space.log,
          FAKE_CLAUDE_SCRIPT: JSON.stringify([
            {
              is_error: true,
              subtype: "error_during_execution",
              result: "hooks failed",
            },
          ]),
        },
        () =>
          new SystemOneAdapterClient({
            structuredOutputs: true,
            llmAnswerMode: "discrete",
            retry: { maxRetries: 2, backoffInitialMs: 0, backoffJitter: 0 },
            model: claudeCodeProvider(),
          })
            .systemOne({ state: "A book.", questions: QUESTIONS })
            .catch((caught: unknown) => caught),
      )) as TypeSafeError;

      expect(error).toBeInstanceOf(TypeSafeError);
      expect(error).not.toBeInstanceOf(APIError);
      expect(error.message).toContain("hooks failed");
      expect((await space.calls()).length).toBe(1);
    } finally {
      await space.cleanup();
    }
  });

  it("retries a crashed CLI process as a connection error", async () => {
    const space = await workspace();
    try {
      const response = await withFakeEnv(
        {
          FAKE_CLAUDE_LOG: space.log,
          FAKE_CLAUDE_SCRIPT: JSON.stringify(["!exit:3:fake-claude: boom", {}]),
        },
        () =>
          new SystemOneAdapterClient({
            structuredOutputs: true,
            llmAnswerMode: "discrete",
            retry: { maxRetries: 1, backoffInitialMs: 0, backoffJitter: 0 },
            model: claudeCodeProvider(),
          }).systemOne({ state: "A book.", questions: QUESTIONS }),
      );

      expect(response.nouls.positive?.noul).toBe(1);
      expect(response.usage.n_retries).toBe(1);
      expect(response.debug.retry_reasons[0]).toEqual([
        "provider_error",
        expect.stringContaining("boom"),
      ]);
    } finally {
      await space.cleanup();
    }
  });

  it("raises a TypeSafeError when the CLI prints no JSON", async () => {
    const space = await workspace();
    try {
      const error = (await withFakeEnv(
        {
          FAKE_CLAUDE_LOG: space.log,
          FAKE_CLAUDE_SCRIPT: JSON.stringify(["!garbage"]),
        },
        () =>
          new SystemOneAdapterClient({
            structuredOutputs: true,
            llmAnswerMode: "discrete",
            model: claudeCodeProvider(),
          })
            .systemOne({ state: "A book.", questions: QUESTIONS })
            .catch((caught: unknown) => caught),
      )) as TypeSafeError;

      expect(error).toBeInstanceOf(TypeSafeError);
      expect(error.message).toContain("did not print a JSON result");
    } finally {
      await space.cleanup();
    }
  });

  it("aborts a hung CLI process with a timeout error", async () => {
    const space = await workspace();
    try {
      const error = (await withFakeEnv(
        {
          FAKE_CLAUDE_LOG: space.log,
          FAKE_CLAUDE_SCRIPT: JSON.stringify(["!sleep:5000"]),
        },
        () =>
          new SystemOneAdapterClient({
            structuredOutputs: true,
            llmAnswerMode: "discrete",
            model: claudeCodeProvider({ timeoutMs: 50 }),
          })
            .systemOne({ state: "A book.", questions: QUESTIONS })
            .catch((caught: unknown) => caught),
      )) as APITimeoutError;

      expect(error).toBeInstanceOf(APITimeoutError);
    } finally {
      await space.cleanup();
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
    const space = await workspace();
    try {
      const provider = claudeCodeProvider();
      const result = await withFakeEnv({ FAKE_CLAUDE_LOG: space.log }, () =>
        provider.request([{ role: "user", content: "hello" }], {
          schema: {},
          structured: false,
        }),
      );

      expect(result.text).toBe('{"answers":{"positive":true}}');
      expect(result.inputTokens).toBe(303);
      expect(result.outputTokens).toBe(4);
      const [call] = await space.calls();
      expect(call.argv.includes("--system-prompt")).toBe(false);
      expect(call.stdin).toBe("User:\nhello");
      expect(provider.translateError(new Error("boom"))).toBeInstanceOf(
        TypeSafeError,
      );
      provider.close();
    } finally {
      await space.cleanup();
    }
  });

  it("validates options and applies env overrides and removals", async () => {
    expect(() => claudeCodeProvider({ timeoutMs: 0 })).toThrow(
      "timeout_ms must be > 0",
    );
    expect(new ClaudeCodeProvider("claude-haiku-4-5").command).toBe("claude");
    const space = await workspace();
    try {
      const overridden = claudeCodeProvider({
        env: { MAX_THINKING_TOKENS: "4096" },
      });
      const removed = claudeCodeProvider({
        env: { MAX_THINKING_TOKENS: undefined, ANTHROPIC_API_KEY: undefined },
      });
      await withFakeEnv({ FAKE_CLAUDE_LOG: space.log }, async () => {
        await overridden.request([{ role: "user", content: "hi" }], {
          schema: {},
          structured: false,
        });
        await removed.request([{ role: "user", content: "hi" }], {
          schema: {},
          structured: false,
        });
      });

      const [overriding, removing] = await space.calls();
      expect(overriding.env.MAX_THINKING_TOKENS).toBe("4096");
      expect(removing.env.MAX_THINKING_TOKENS).toBeUndefined();
    } finally {
      await space.cleanup();
    }
  });
});
