import Anthropic, {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  APIUserAbortError,
} from "@anthropic-ai/sdk";
import { TypeSafeError } from "@typesafe-ai/sdk";
import {
  providerErrorTranslator,
  translating,
} from "../utils/errorHandling.js";
import {
  conversation,
  type Message,
  type Provider,
  type ProviderRequestOptions,
  type ProviderResult,
  recordRequest,
  recordResponse,
  systemPrompt,
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

/** Map an Anthropic SDK error to an SDK error. */
const translateError = providerErrorTranslator({
  timeout: APIConnectionTimeoutError,
  userAbort: APIUserAbortError,
  connection: APIConnectionError,
  apiError: APIError,
});

/** One Messages API payload in the shape the provider reads. */
interface MessagesPayload {
  stop_reason: string | null;
  content: { type: string; text?: string }[];
  usage: { input_tokens: number; output_tokens: number };
}

/** Parse one Messages API payload, rejecting truncated output. */
const anthropicResult = (response: MessagesPayload): ProviderResult => {
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

/** The request parameters for one Messages API call. */
const requestParams = (
  modelName: string,
  messages: readonly Message[],
  options: ProviderRequestOptions,
  maxTokens: number,
): Record<string, unknown> => {
  const params: Record<string, unknown> = {
    model: modelName,
    max_tokens: maxTokens,
    system: systemPrompt(messages),
    messages: conversation(messages),
  };
  if (options.structured)
    params.output_config = {
      format: { type: "json_schema", schema: options.schema },
    };
  return params;
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
      const params = requestParams(
        this.modelName,
        messages,
        options,
        this.maxTokens,
      );
      recordRequest(params, { api: "messages" });
      const response = await this.client.messages.create(
        params as unknown as Parameters<typeof this.client.messages.create>[0],
      );
      return anthropicResult(response as unknown as MessagesPayload);
    }, translateError);
  }

  /** Map an Anthropic SDK error to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    return translateError(error);
  }
}

export {
  AnthropicProvider,
  anthropicResult,
  requestParams,
  translateError as translateAnthropicError,
};
