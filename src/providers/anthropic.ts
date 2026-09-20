import Anthropic, {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  APIUserAbortError,
} from "@anthropic-ai/sdk";
import {
  TypeSafeError,
  APIUserAbortError as TypeSafeUserAbortError,
} from "@typesafe-ai/sdk";
import {
  describeError,
  toConnectionError,
  toStatusError,
  toTimeoutError,
  translating,
} from "../utils/errorHandling.js";
import {
  type Message,
  type Provider,
  type ProviderRequestOptions,
  type ProviderResult,
  recordRequest,
  recordResponse,
} from "./base.js";

/** Default maximum output tokens per Anthropic request. */
const DEFAULT_MAX_TOKENS = 4096;

/** Options for constructing an Anthropic provider. */
interface AnthropicProviderOptions {
  /** Maximum output tokens per request. */
  maxTokens?: number;
  /** Endpoint credential; defaults to the SDK's resolution. */
  apiKey?: string;
  /** Endpoint URL; defaults to the SDK's resolution. */
  baseUrl?: string;
  /** Custom fetch implementation, for tests and transports. */
  fetch?: typeof globalThis.fetch;
}

/** Map an Anthropic SDK exception to an SDK error. */
const translateError = (error: unknown): TypeSafeError => {
  if (error instanceof TypeSafeError) return error;
  if (error instanceof APIConnectionTimeoutError) return toTimeoutError(error);
  if (error instanceof APIUserAbortError)
    return new TypeSafeUserAbortError(undefined, { cause: error });
  if (error instanceof APIConnectionError) return toConnectionError(error);
  if (error instanceof APIError && typeof error.status === "number")
    return toStatusError(error);
  return new TypeSafeError(describeError(error));
};

/** The request parameters for one Messages API call. */
const requestKwargs = (
  modelName: string,
  messages: readonly Message[],
  schema: Record<string, unknown>,
  { structured, maxTokens }: { structured: boolean; maxTokens: number },
): Record<string, unknown> => {
  const system = messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n\n");
  const conversation = messages
    .filter((message) => message.role !== "system")
    .map((message) => ({ role: message.role, content: message.content }));
  const kwargs: Record<string, unknown> = {
    model: modelName,
    max_tokens: maxTokens,
    system,
    messages: conversation,
  };
  if (structured)
    kwargs.output_config = { format: { type: "json_schema", schema } };
  return kwargs;
};

/** Parse one Messages API payload, rejecting truncated output. */
const result = (response: {
  stop_reason: string | null;
  content: { type: string; text?: string }[];
  usage: { input_tokens: number; output_tokens: number };
}): ProviderResult => {
  recordResponse(response, { finishReason: response.stop_reason });
  if (response.stop_reason === "max_tokens")
    throw new TypeSafeError(
      "Anthropic response was truncated at the output token limit. " +
        "Increase max_tokens on AnthropicProvider, or request fewer questions.",
    );
  const text = response.content
    .filter((block) => block.type === "text")
    .map((block) => block.text ?? "")
    .join("");
  return {
    text,
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
  };
};

/** Call the native Anthropic Messages API. */
class AnthropicProvider implements Provider {
  readonly modelName: string;
  readonly maxTokens: number;
  readonly client: Anthropic;

  constructor(modelName: string, options: AnthropicProviderOptions = {}) {
    const maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    if (maxTokens <= 0) throw new Error("max_tokens must be > 0");
    this.modelName = modelName;
    this.maxTokens = maxTokens;
    this.client = new Anthropic({
      maxRetries: 0,
      apiKey: options.apiKey,
      baseURL: options.baseUrl,
      fetch: options.fetch,
    });
  }

  /** No-op: this SDK version owns no connection pool to release. */
  close(): void {
    // Nothing to release; the SDK uses the global fetch agent.
  }

  /** Perform one Messages request and return its raw payload and usage. */
  async request(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    return translating(async () => {
      const kwargs = requestKwargs(this.modelName, messages, options.schema, {
        structured: options.structured,
        maxTokens: this.maxTokens,
      });
      recordRequest(kwargs, { api: "messages" });
      const response = await this.client.messages.create(
        kwargs as unknown as Parameters<typeof this.client.messages.create>[0],
      );
      return result(
        response as unknown as {
          stop_reason: string | null;
          content: { type: string; text?: string }[];
          usage: { input_tokens: number; output_tokens: number };
        },
      );
    }, translateError);
  }

  /** Map an Anthropic SDK exception to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    return translateError(error);
  }
}

export {
  AnthropicProvider,
  type AnthropicProviderOptions,
  DEFAULT_MAX_TOKENS,
  requestKwargs,
  result as anthropicResult,
  translateError as translateAnthropicError,
};
