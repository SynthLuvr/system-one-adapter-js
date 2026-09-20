import {
  TypeSafeError,
  APIUserAbortError as TypeSafeUserAbortError,
} from "@typesafe-ai/sdk";
import OpenAI, {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  APIUserAbortError,
} from "openai";
import type { ChatCompletionCreateParamsNonStreaming } from "openai/resources/chat/completions";
import type { ResponseCreateParamsNonStreaming } from "openai/resources/responses/responses.js";
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
  renderMessages,
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

/** Map an OpenAI SDK exception to an SDK error. */
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

/** The `response_format` for Chat Completions, or none when prompted. */
const responseFormat = (
  schema: Record<string, unknown>,
  { structured }: { structured: boolean },
): ChatCompletionCreateParamsNonStreaming["response_format"] => {
  if (!structured) return undefined;
  return {
    type: "json_schema",
    json_schema: { name: "evaluation", schema, strict: true },
  };
};

/** The request parameters for one Responses API call. */
const responsesRequest = (
  modelName: string,
  messages: readonly Message[],
  schema: Record<string, unknown>,
  { structured }: { structured: boolean },
): Record<string, unknown> => {
  const outputFormat = structured
    ? { type: "json_schema", name: "evaluation", schema, strict: true }
    : { type: "json_object" };
  const kwargs: Record<string, unknown> = {
    model: modelName,
    input: renderMessages(messages),
    text: { format: outputFormat },
    store: false,
  };
  if (structured) {
    kwargs.instructions = messages
      .filter((message) => message.role === "system")
      .map((message) => message.content)
      .join("\n\n");
    kwargs.input = renderMessages(
      messages.filter((message) => message.role !== "system"),
    );
  }
  // JSON mode requires a JSON instruction in `input`; the separate
  // `instructions` field does not satisfy the API's check, so prompted mode
  // keeps system messages.
  return kwargs;
};

/** One Responses API payload in the shape the provider reads. */
interface ResponsesPayload {
  status: string;
  error?: { message?: string } | null;
  incomplete_details?: { reason?: string } | null;
  output_text: string;
  usage: { input_tokens: number; output_tokens: number };
}

/** Parse one Responses API payload, rejecting unfinished responses. */
const responsesResult = (response: ResponsesPayload): ProviderResult => {
  recordResponse(response, { finishReason: response.status });
  if (response.status !== "completed") {
    let reason: string = response.status;
    if (response.error !== null && response.error !== undefined)
      reason = response.error.message ?? reason;
    else if (
      response.incomplete_details !== null &&
      response.incomplete_details !== undefined
    )
      reason = response.incomplete_details.reason ?? reason;
    throw new TypeSafeError(`OpenAI response did not complete: ${reason}.`);
  }
  return {
    text: response.output_text,
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
  };
};

/** One Chat Completions payload in the shape the provider reads. */
interface ChatCompletionPayload {
  choices: {
    message: { content: string | null };
    finish_reason: string | null;
  }[];
  usage: { prompt_tokens: number; completion_tokens: number };
}

/** Parse one Chat Completions payload. */
const chatResult = (response: ChatCompletionPayload): ProviderResult => {
  recordResponse(response, { finishReason: response.choices[0].finish_reason });
  return {
    text: response.choices[0].message.content ?? "",
    inputTokens: response.usage.prompt_tokens,
    outputTokens: response.usage.completion_tokens,
  };
};

/** Call the OpenAI Responses API or an OpenAI-compatible chat API. */
class OpenAIProvider implements Provider {
  readonly modelName: string;
  readonly api: OpenAIApi;
  readonly client: OpenAI;

  constructor(modelName: string, options: OpenAIProviderOptions = {}) {
    if (
      options.api !== undefined &&
      options.api !== "responses" &&
      options.api !== "chat_completions"
    )
      throw new Error("api must be 'responses' or 'chat_completions'");
    this.modelName = modelName;
    this.client = new OpenAI({
      baseURL: options.baseUrl,
      apiKey: options.apiKey,
      maxRetries: 0,
      fetch: options.fetch,
    });
    this.api =
      options.api ??
      (new URL(this.client.baseURL).host === "api.openai.com"
        ? "responses"
        : "chat_completions");
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
    return translating(async () => {
      if (this.api === "responses") {
        const kwargs = responsesRequest(
          this.modelName,
          messages,
          options.schema,
          { structured: options.structured },
        );
        recordRequest(kwargs, { api: this.api });
        const response = await this.client.responses.create(
          kwargs as unknown as ResponseCreateParamsNonStreaming,
        );
        return responsesResult(response as unknown as ResponsesPayload);
      }
      const kwargs: Record<string, unknown> = {
        model: this.modelName,
        messages: renderMessages(messages),
        response_format: responseFormat(options.schema, {
          structured: options.structured,
        }),
      };
      recordRequest(kwargs, { api: this.api });
      const response = await this.client.chat.completions.create(
        kwargs as unknown as ChatCompletionCreateParamsNonStreaming,
      );
      return chatResult(response as unknown as ChatCompletionPayload);
    }, translateError);
  }

  /** Map an OpenAI SDK exception to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    return translateError(error);
  }
}

export {
  type ChatCompletionPayload,
  chatResult,
  type OpenAIApi,
  OpenAIProvider,
  type OpenAIProviderOptions,
  type ResponsesPayload,
  responseFormat,
  responsesRequest,
  responsesResult,
  translateError as translateOpenAIError,
};
