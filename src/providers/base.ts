import { AsyncLocalStorage } from "node:async_hooks";
import type { TypeSafeError } from "@typesafe-ai/sdk";

/** The provider an owned model name is built from. */
type ProviderName = "openai" | "anthropic";

/** One chat message in provider-neutral form. */
interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

/** The raw JSON payload a model returned and the tokens it cost. */
interface ProviderResult {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** Options for one provider request. */
interface ProviderRequestOptions {
  /** JSON schema for the model's answer. */
  schema: Record<string, unknown>;
  /** Whether to use native structured output. */
  structured: boolean;
}

/** Perform one asynchronous model request in native or prompted output mode. */
interface Provider {
  /** Model name reported on responses. */
  readonly modelName: string;
  /** Perform one model request and return its raw payload and usage. */
  request(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult>;
  /** Map an exception from this provider's SDK to an SDK error. */
  translateError(error: unknown): TypeSafeError;
}

/** A provider that owns resources a caller can release with `close`. */
interface ClosableProvider extends Provider {
  /** Release resources owned by the provider. */
  close(): void | Promise<void>;
}

/** One captured provider call, including calls that raise before returning. */
interface LlmAttempt {
  messages: { role: string; content: string }[];
  model_request_parameters: { schema: unknown; structured: boolean };
  llm_response: unknown;
  debug_info: {
    model_name: string;
    provider: string;
    error?: string;
    error_type?: string;
    api?: string;
    finish_reason?: string | null;
  };
  request?: unknown;
}

/** Per-call context so providers enrich the trace without protocol changes. */
const attemptStorage = new AsyncLocalStorage<LlmAttempt>();

/** Render messages into the role/content dictionaries the chat APIs expect. */
const renderMessages = (
  messages: readonly Message[],
): { role: string; content: string }[] =>
  messages.map((message) => ({ role: message.role, content: message.content }));

/** The joined system-prompt text of a conversation. */
const systemPrompt = (messages: readonly Message[]): string =>
  messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n\n");

/** The non-system messages, for APIs that take the system prompt separately. */
const conversation = (messages: readonly Message[]) =>
  renderMessages(messages.filter((message) => message.role !== "system"));

/** Deep-copy a plain SDK response object for the attempt trace. */
const snapshotResponse = (response: unknown): unknown => {
  if (typeof (response as { toJSON?: unknown })?.toJSON === "function")
    return structuredClone((response as { toJSON: () => unknown }).toJSON());
  return structuredClone(response);
};

/** Record a thrown error on its attempt trace. */
const recordFailure = (attempt: LlmAttempt, error: unknown): void => {
  attempt.debug_info.error =
    error instanceof Error ? error.message : String(error);
  attempt.debug_info.error_type =
    error instanceof Error ? error.name : typeof error;
};

/** Snapshot one provider call, including calls that raise before returning. */
const captureAttempt = async (
  attempts: LlmAttempt[],
  provider: Provider,
  messages: readonly Message[],
  options: ProviderRequestOptions,
  fn: () => Promise<ProviderResult>,
): Promise<{ attempt: LlmAttempt; result: ProviderResult }> => {
  const attempt: LlmAttempt = {
    messages: renderMessages(messages),
    model_request_parameters: {
      schema: structuredClone(options.schema),
      structured: options.structured,
    },
    llm_response: null,
    debug_info: {
      model_name: provider.modelName,
      provider: provider.constructor.name,
    },
  };
  attempts.push(attempt);
  return attemptStorage.run(attempt, async () => {
    try {
      return { attempt, result: await fn() };
    } catch (error) {
      recordFailure(attempt, error);
      throw error;
    }
  });
};

/** Capture the built-in provider's SDK arguments before sending the request. */
const recordRequest = (request: unknown, { api }: { api: string }): void => {
  const attempt = attemptStorage.getStore();
  if (attempt !== undefined) {
    attempt.request = structuredClone(request);
    attempt.debug_info.api = api;
  }
};

/** Capture the SDK response before parsing can reject incomplete output. */
const recordResponse = (
  response: unknown,
  { finishReason }: { finishReason: string | null | undefined },
): void => {
  const attempt = attemptStorage.getStore();
  if (attempt !== undefined) {
    attempt.llm_response = snapshotResponse(response);
    attempt.debug_info.finish_reason = finishReason ?? null;
  }
};

export {
  type ClosableProvider,
  captureAttempt,
  conversation,
  type LlmAttempt,
  type Message,
  type Provider,
  type ProviderName,
  type ProviderRequestOptions,
  type ProviderResult,
  recordRequest,
  recordResponse,
  renderMessages,
  systemPrompt,
};
