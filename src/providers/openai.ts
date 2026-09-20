import { TypeSafeError } from "@typesafe-ai/sdk";
import { scope, type } from "arktype";
import OpenAI, {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  APIUserAbortError,
} from "openai";
import type { ChatCompletionCreateParamsNonStreaming } from "openai/resources/chat/completions";
import type {
  ResponseCreateParamsNonStreaming,
  ResponseFormatTextConfig,
} from "openai/resources/responses/responses.js";
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
  parsePayload,
  recordRequest,
  recordResponse,
  renderMessages,
  systemPrompt,
} from "./base.js";

/** Which OpenAI-compatible API a provider calls. */
type OpenAIApi = "responses" | "chat_completions";

/** Options for constructing an OpenAI provider. */
interface OpenAIProviderOptions {
  /** OpenAI-compatible endpoint URL; defaults to the SDK's resolution. */
  baseUrl?: string;
  /** Endpoint credential; defaults to the SDK's resolution. */
  apiKey?: string;
  /** `"responses"` or `"chat_completions"`; defaults by endpoint host. */
  api?: OpenAIApi;
  /** Custom fetch implementation, for tests and transports. */
  fetch?: typeof globalThis.fetch;
}

/** Map an OpenAI SDK error to an SDK error. */
const translateError = providerErrorTranslator({
  timeout: APIConnectionTimeoutError,
  userAbort: APIUserAbortError,
  connection: APIConnectionError,
  apiError: APIError,
});

/** Runtime validation of the API payloads this provider reads. */
const payloadTypes = scope({
  ResponsesError: {
    "message?": "string",
  },
  IncompleteDetails: {
    "reason?": "string",
  },
  ResponsesUsage: {
    input_tokens: "number",
    output_tokens: "number",
  },
  ResponsesPayload: {
    status: "string",
    "error?": "ResponsesError|null",
    "incomplete_details?": "IncompleteDetails|null",
    output_text: "string",
    "usage?": "ResponsesUsage",
  },
  ChatChoice: {
    message: {
      content: "string|null",
    },
    finish_reason: "string|null",
  },
  ChatUsage: {
    prompt_tokens: "number",
    completion_tokens: "number",
  },
  ChatPayload: {
    choices: "ChatChoice[]",
    usage: "ChatUsage",
  },
}).export();

/** The `response_format` for Chat Completions, or none when prompted. */
const responseFormat = (
  options: ProviderRequestOptions,
): ChatCompletionCreateParamsNonStreaming["response_format"] | undefined => {
  if (!options.structured) return undefined;
  return {
    type: "json_schema",
    json_schema: { name: "evaluation", schema: options.schema, strict: true },
  };
};

/** The request parameters for one Responses API call. */
const responsesRequest = (
  modelName: string,
  messages: readonly Message[],
  options: ProviderRequestOptions,
): ResponseCreateParamsNonStreaming => {
  const format: ResponseFormatTextConfig = options.structured
    ? {
        type: "json_schema",
        name: "evaluation",
        schema: options.schema,
        strict: true,
      }
    : { type: "json_object" };
  const params: ResponseCreateParamsNonStreaming = {
    model: modelName,
    input: renderMessages(messages),
    text: { format },
    store: false,
  };
  if (options.structured) {
    params.instructions = systemPrompt(messages);
    params.input = conversation(messages);
  }
  // JSON mode requires a JSON instruction in `input`; the separate
  // `instructions` field does not satisfy the API's check, so prompted mode
  // keeps system messages inside `input`.
  return params;
};

/** Parse one Responses API payload, rejecting unfinished responses. */
const responsesResult = (response: unknown): ProviderResult => {
  const payload = parsePayload(
    () => payloadTypes.ResponsesPayload(response),
    "OpenAI response",
  );
  recordResponse(response, { finishReason: payload.status });
  if (payload.status !== "completed") {
    let reason: string = payload.status;
    if (payload.error != null) reason = payload.error.message ?? reason;
    else if (payload.incomplete_details != null)
      reason = payload.incomplete_details.reason ?? reason;
    throw new TypeSafeError(`OpenAI response did not complete: ${reason}.`);
  }
  const usage = parsePayload(
    () => payloadTypes.ResponsesUsage(payload.usage),
    "OpenAI response usage",
  );
  return {
    text: payload.output_text,
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
  };
};

/** Parse one Chat Completions payload. */
const chatResult = (response: unknown): ProviderResult => {
  const payload = parsePayload(
    () => payloadTypes.ChatPayload(response),
    "OpenAI chat completion",
  );
  const [choice] = payload.choices;
  recordResponse(response, { finishReason: choice.finish_reason });
  return {
    text: choice.message.content ?? "",
    inputTokens: payload.usage.prompt_tokens,
    outputTokens: payload.usage.completion_tokens,
  };
};

/** Resolve the API option, defaulting by endpoint host. */
const resolveApi = (
  option: OpenAIApi | undefined,
  baseURL: string,
): OpenAIApi => {
  if (option !== undefined) {
    const api = type("'responses'|'chat_completions'")(option);
    if (api instanceof type.errors)
      throw new Error("api must be 'responses' or 'chat_completions'");
    return api;
  }
  return new URL(baseURL).host === "api.openai.com"
    ? "responses"
    : "chat_completions";
};

/** Call the OpenAI Responses API or an OpenAI-compatible chat API. */
class OpenAIProvider implements Provider {
  readonly modelName: string;
  readonly api: OpenAIApi;
  readonly client: OpenAI;

  constructor(modelName: string, options: OpenAIProviderOptions = {}) {
    this.modelName = modelName;
    this.client = new OpenAI({
      baseURL: options.baseUrl,
      apiKey: options.apiKey,
      maxRetries: 0,
      fetch: options.fetch,
    });
    this.api = resolveApi(options.api, this.client.baseURL);
  }

  /** No-op: this SDK version owns no connection pool to release. */
  close(): void {
    // Nothing to release; the SDK uses the global fetch agent.
  }

  /** Perform one request and return its raw payload and usage. */
  async request(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    return translating(
      () =>
        this.api === "responses"
          ? this.#requestResponses(messages, options)
          : this.#requestChat(messages, options),
      translateError,
    );
  }

  async #requestResponses(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    const params = responsesRequest(this.modelName, messages, options);
    recordRequest(params, { api: this.api });
    const response = await this.client.responses.create(params);
    return responsesResult(response);
  }

  async #requestChat(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    const params: ChatCompletionCreateParamsNonStreaming = {
      model: this.modelName,
      messages: renderMessages(messages),
      response_format: responseFormat(options),
    };
    recordRequest(params, { api: this.api });
    const response = await this.client.chat.completions.create(params);
    return chatResult(response);
  }

  /** Map an OpenAI SDK error to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    return translateError(error);
  }
}

export {
  chatResult,
  type OpenAIApi,
  OpenAIProvider,
  responseFormat,
  responsesResult,
  translateError as translateOpenAIError,
};
