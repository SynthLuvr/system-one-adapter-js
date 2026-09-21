import { type ExecFileException, execFile } from "node:child_process";
import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  TypeSafeError,
} from "@typesafe-ai/sdk";
import { scope, type } from "arktype";
import { describeError, translating } from "../utils/errorHandling.js";
import {
  type ClosableProvider,
  conversation,
  type Message,
  type ProviderRequestOptions,
  type ProviderResult,
  parsePayload,
  recordRequest,
  recordResponse,
  systemPrompt,
} from "./base.js";

/** Default environment overrides for cheap, deterministic evaluations. */
const DEFAULT_ENV = { MAX_THINKING_TOKENS: "0" };

/** Largest stream the provider reads from one CLI process. */
const MAX_BUFFER_BYTES = 16 * 1024 * 1024;

/** Options for constructing a Claude Code CLI provider. */
interface ClaudeCodeProviderOptions {
  /** CLI executable; defaults to `claude` resolved from `PATH`. */
  command?: string;
  /** Extra CLI arguments, placed before the provider's own flags. */
  args?: string[];
  /**
   * Environment overrides applied over the calling process on top of the
   * `MAX_THINKING_TOKENS=0` default; a value of `undefined` removes the
   * variable, for example an inherited `ANTHROPIC_API_KEY`.
   */
  env?: Record<string, string | undefined>;
  /** Abort requests that run longer than this many milliseconds. */
  timeoutMs?: number;
}

/** Runtime validation of the print-mode result payload the CLI prints. */
const payloadTypes = scope({
  Usage: {
    "input_tokens?": "number",
    "output_tokens?": "number",
    "cache_creation_input_tokens?": "number",
    "cache_read_input_tokens?": "number",
  },
  ResultPayload: {
    type: "'result'",
    subtype: "string",
    is_error: "boolean",
    "result?": "string|null",
    "stop_reason?": "string|null",
    "api_error_status?": "number|null",
    "usage?": "Usage|null",
  },
}).export();

/** The validated shape of one CLI result payload. */
type ResultPayload = typeof payloadTypes.ResultPayload.infer;

/** One completed CLI process run. */
interface CliRun {
  stdout: string;
  stderr: string;
}

/** The execFile failure of one CLI process, with its captured stderr. */
class CliProcessError extends Error {
  readonly exitCode: string | number | undefined;
  readonly timedOut: boolean;
  readonly stderr: string;

  constructor(error: ExecFileException, stderr: string) {
    super(error.message, { cause: error });
    this.exitCode = error.code;
    this.timedOut = error.killed === true;
    this.stderr = stderr;
  }
}

/** The first line of `text`, shortened for error messages. */
const excerpt = (text: string): string =>
  (text.trim().split("\n")[0] ?? "").slice(0, 200);

/** Render non-system messages as labeled turns for one CLI prompt. */
const renderConversation = (messages: readonly Message[]): string =>
  conversation(messages)
    .map(
      (message) =>
        `${message.role === "user" ? "User" : "Assistant"}:\n${message.content}`,
    )
    .join("\n\n");

/** The print-mode CLI arguments for one evaluation request. */
const cliArgs = (
  modelName: string,
  messages: readonly Message[],
  options: ProviderRequestOptions,
): string[] => {
  const args = [
    "-p",
    "--output-format",
    "json",
    "--model",
    modelName,
    "--tools",
    "",
    "--no-session-persistence",
  ];
  const prompt = systemPrompt(messages);
  if (prompt !== "") args.push("--system-prompt", prompt);
  if (options.structured)
    args.push("--json-schema", JSON.stringify(options.schema));
  return args;
};

/** Map an `is_error` result onto the SDK error it represents. */
const resultError = (payload: ResultPayload): TypeSafeError => {
  const message = payload.result ?? `result subtype: ${payload.subtype}`;
  if (payload.api_error_status == null) return new TypeSafeError(message);
  return APIError.fromResponse(
    payload.api_error_status,
    { error: { message } },
    new Headers(),
  );
};

/** Parse one print-mode result payload, counting input across all caches. */
const claudeCodeResult = (stdout: string): ProviderResult => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout);
  } catch {
    throw new TypeSafeError(
      "The Claude Code CLI did not print a JSON result. Check that it is " +
        "installed and authenticated, and inspect the request recorded in " +
        "debug.llm_attempts.",
    );
  }
  const payload = parsePayload(
    () => payloadTypes.ResultPayload(parsed),
    "Claude Code result",
  );
  recordResponse(parsed, { finishReason: payload.stop_reason ?? null });
  if (payload.is_error) throw resultError(payload);
  const usage = payload.usage;
  return {
    text: payload.result ?? "",
    // The CLI bills its own system prompt as a cache write or read, so the
    // honest input figure adds those tokens to the request's input tokens.
    inputTokens:
      (usage?.input_tokens ?? 0) +
      (usage?.cache_creation_input_tokens ?? 0) +
      (usage?.cache_read_input_tokens ?? 0),
    outputTokens: usage?.output_tokens ?? 0,
  };
};

/** Run the CLI once with `prompt` on stdin. */
const runCli = (
  command: string,
  args: readonly string[],
  env: NodeJS.ProcessEnv,
  prompt: string,
  timeoutMs: number | undefined,
): Promise<CliRun> =>
  new Promise((resolve, reject) => {
    const child = execFile(
      command,
      args,
      { env, timeout: timeoutMs, maxBuffer: MAX_BUFFER_BYTES },
      (error, stdout, stderr) => {
        if (error === null) resolve({ stdout, stderr });
        else reject(new CliProcessError(error, stderr));
      },
    );
    // A CLI that exits before draining stdin (unknown flags, missing config)
    // raises EPIPE on this stream; the exit above is the actual failure.
    child.stdin?.on("error", () => undefined);
    child.stdin?.end(prompt);
  });

/** Map one failed CLI process onto the SDK error it represents. */
const cliRunError = (
  error: CliProcessError,
  timeoutMs: number | undefined,
): TypeSafeError => {
  if (error.timedOut)
    return new APITimeoutError(timeoutMs ?? 0, { cause: error.cause });
  const detail = excerpt(error.stderr) || error.message;
  return new APIConnectionError(
    error.exitCode === undefined
      ? `Claude Code CLI could not run: ${detail}`
      : `Claude Code CLI exited with code ${String(error.exitCode)}: ${detail}`,
    { cause: error.cause },
  );
};

/** Map an unexpected provider exception onto an SDK error. */
const toProviderError = (error: unknown): TypeSafeError =>
  error instanceof TypeSafeError
    ? error
    : new TypeSafeError(describeError(error));

/** The environment one CLI process runs in, and what traces may record. */
interface ResolvedEnvironment {
  /** The child process environment. */
  child: NodeJS.ProcessEnv;
  /** The variables the provider set, recorded on debug traces. */
  applied: Record<string, string>;
}

/** Apply the defaults plus caller overrides over the calling process. */
const resolveEnvironment = (
  overrides: Record<string, string | undefined>,
): ResolvedEnvironment => {
  const child: NodeJS.ProcessEnv = { ...process.env };
  const applied: Record<string, string> = {};
  for (const [key, value] of Object.entries({
    ...DEFAULT_ENV,
    ...overrides,
  }))
    if (value === undefined) delete child[key];
    else {
      child[key] = value;
      applied[key] = value;
    }
  return { child, applied };
};

/** Run evaluations through the Claude Code CLI in print mode. */
class ClaudeCodeProvider implements ClosableProvider {
  readonly modelName: string;
  readonly command: string;
  readonly extraArgs: string[];
  readonly env: Record<string, string | undefined>;
  readonly timeoutMs: number | undefined;

  constructor(modelName: string, options: ClaudeCodeProviderOptions = {}) {
    if (options.timeoutMs !== undefined) {
      const timeout = type("number > 0")(options.timeoutMs);
      if (timeout instanceof type.errors)
        throw new Error("timeout_ms must be > 0");
    }
    this.modelName = modelName;
    this.command = options.command ?? "claude";
    this.extraArgs = options.args ?? [];
    this.env = options.env ?? {};
    this.timeoutMs = options.timeoutMs;
  }

  /** No-op: each request owns its own CLI process. */
  close(): void {
    // Nothing is held between requests.
  }

  /** Run one CLI evaluation and return its payload and usage. */
  async request(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    return translating(async () => {
      const args = [
        ...this.extraArgs,
        ...cliArgs(this.modelName, messages, options),
      ];
      const prompt = renderConversation(messages);
      const { child, applied } = resolveEnvironment(this.env);
      // Only the variables the provider sets are recorded: the inherited
      // environment may hold credentials that do not belong in traces.
      recordRequest(
        { command: this.command, args, env: applied, prompt },
        { api: "claude_code" },
      );
      let run: CliRun;
      try {
        run = await runCli(this.command, args, child, prompt, this.timeoutMs);
      } catch (error) {
        if (error instanceof CliProcessError)
          throw cliRunError(error, this.timeoutMs);
        throw error;
      }
      return claudeCodeResult(run.stdout);
    }, toProviderError);
  }

  /** Map an exception from this provider onto an SDK error. */
  translateError(error: unknown): TypeSafeError {
    return toProviderError(error);
  }
}

export { ClaudeCodeProvider, type ClaudeCodeProviderOptions };
