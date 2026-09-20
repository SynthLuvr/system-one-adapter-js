import Anthropic, {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  APIUserAbortError,
} from "@anthropic-ai/sdk";
import type { MessageCreateParamsNonStreaming } from "@anthropic-ai/sdk/resources/messages/messages";
import { TypeSafeError } from "@typesafe-ai/sdk";
import { scope, type } from "arktype";
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

/** arktype type of a positive output-token limit. */
const MaxTokens = type("number > 0");

/** Runtime validation of the Messages API payloads this provider reads. */
const payloadTypes = scope({
  ContentBlock: {
    type: "string",
    "text?": "string",
  },
  Usage: {
    input_tokens: "number",
    output_tokens: "number",
  },
  MessagesPayload: {
    stop_reason: "string|null",
    content: "ContentBlock[]",
    usage: "Usage",
  },
}).export();

/** Parse one Messages API payload, rejecting truncated output. */
const anthropicResult = (response: unknown): ProviderResult => {
  const payload = payloadTypes.MessagesPayload(response);
  if (payload instanceof type.errors)
    throw new TypeSafeError(
      `Anthropic response did not match the expected shape:\n${payload.summary}`,
    );
  recordResponse(response, { finishReason: payload.stop_reason });
  if (payload.stop_reason === "max_tokens")
    throw new TypeSafeError(
      "Anthropic response was truncated at the output token limit. " +
        "Increase max_tokens on AnthropicProvider, or request fewer questions.",
    );
  const text = payload.content
    .filter((block) => block.type === "text")
    .map((block) => block.text ?? "")
    .join("");
  return {
    text,
    inputTokens: payload.usage.input_tokens,
    outputTokens: payload.usage.output_tokens,
  };
};

/** The request parameters for one Messages API call. */
const requestParams = (
  modelName: string,
  messages: readonly Message[],
  options: ProviderRequestOptions,
  maxTokens: number,
): MessageCreateParamsNonStreaming => {
  const params: MessageCreateParamsNonStreaming = {
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
    const maxTokens = MaxTokens(options.maxTokens ?? DEFAULT_MAX_TOKENS);
    if (maxTokens instanceof type.errors)
      throw new Error("max_tokens must be > 0");
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
      const response = await this.client.messages.create(params);
      return anthropicResult(response);
    }, translateError);
  }

  /** Map an exception from this provider's SDK to an SDK error. */
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
