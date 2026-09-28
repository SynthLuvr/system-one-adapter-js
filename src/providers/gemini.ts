import { GoogleGenAI } from "@google/genai";
import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  APIUserAbortError,
  TypeSafeError,
} from "@typesafe-ai/sdk";
import { scope } from "arktype";
import { describeError, translating } from "../utils/errorHandling.js";
import {
  conversation,
  type Message,
  type Provider,
  type ProviderRequestOptions,
  type ProviderResult,
  parsePayload,
  recordRequest,
  recordResponse,
  systemPrompt,
} from "./base.js";

/** Options for constructing a Gemini provider. */
interface GeminiProviderOptions {
  /**
   * Endpoint credential; defaults to the SDK's own resolution, including
   * `GEMINI_API_KEY` and `GOOGLE_API_KEY`.
   */
  apiKey?: string;
  /** Endpoint URL; defaults to the SDK's resolution. */
  baseUrl?: string;
  /** Custom fetch implementation, for tests and transports. */
  fetch?: typeof globalThis.fetch;
}

/**
 * One conversation turn in the Interactions API's step format. The union keeps
 * each turn assignable to the SDK's own step types, which discriminate on the
 * literal `type`.
 */
type InteractionStep =
  | { type: "user_input"; content: { type: "text"; text: string }[] }
  | { type: "model_output"; content: { type: "text"; text: string }[] };

/** The Interactions API request parameters for one model interaction. */
interface InteractionParams {
  model: string;
  input: InteractionStep[];
  store: false;
  system_instruction?: string;
  response_format?: {
    type: "text";
    mime_type: "application/json";
    schema: Record<string, unknown>;
  };
}

/**
 * Map a Gemini SDK error to an SDK error. `@google/genai` does not export the
 * Interactions API's error classes, so transport failures are recognized by
 * the class name the SDK sets.
 */
const translateError = (error: unknown): TypeSafeError => {
  if (error instanceof TypeSafeError) return error;
  const name = error instanceof Error ? error.name : "";
  if (name === "APIConnectionTimeoutError")
    return new APITimeoutError(0, { cause: error });
  if (name === "APIUserAbortError")
    return new APIUserAbortError(undefined, { cause: error });
  if (name === "APIConnectionError")
    return new APIConnectionError(describeError(error), { cause: error });
  const {
    status,
    error: body,
    headers,
  } = error as {
    status?: unknown;
    error?: unknown;
    headers?: Headers;
  };
  if (typeof status === "number")
    return APIError.fromResponse(status, body, headers ?? new Headers());
  return new TypeSafeError(describeError(error));
};

/** Runtime validation of the Interactions API payloads this provider reads. */
const payloadTypes = scope({
  InteractionError: {
    "code?": "string",
    "message?": "string",
  },
  Usage: {
    "total_input_tokens?": "number",
    "total_output_tokens?": "number",
  },
  InteractionPayload: {
    status: "string",
    "output_text?": "string|null",
    "errors?": "InteractionError[]|null",
    "usage?": "Usage|null",
  },
}).export();

/**
 * The SDK attaches a non-cloneable `sdkHttpResponse`, holding the raw
 * `Response`, to every interaction; drop it before the trace snapshot.
 */
const withoutSdkHttpResponse = (response: unknown): unknown => {
  if (typeof response !== "object" || response === null) return response;
  const { sdkHttpResponse, ...rest } = response as Record<string, unknown>;
  return sdkHttpResponse === undefined ? response : rest;
};

/** Parse one Interactions API payload, rejecting unfinished interactions. */
const geminiResult = (response: unknown): ProviderResult => {
  const payload = parsePayload(
    () => payloadTypes.InteractionPayload(response),
    "Gemini response",
  );
  recordResponse(withoutSdkHttpResponse(response), {
    finishReason: payload.status,
  });
  if (payload.status !== "completed") {
    const errors = payload.errors ?? [];
    const reason =
      errors.length > 0 ? JSON.stringify(errors) : payload.status || "unknown";
    throw new TypeSafeError(`Gemini response did not complete: ${reason}.`);
  }
  const { total_input_tokens: inputTokens, total_output_tokens: outputTokens } =
    payload.usage ?? {};
  if (inputTokens === undefined || outputTokens === undefined)
    throw new TypeSafeError("Gemini response omitted usage.");
  return { text: payload.output_text ?? "", inputTokens, outputTokens };
};

/** The request parameters for one Interactions API call. */
const interactionParams = (
  modelName: string,
  messages: readonly Message[],
  options: ProviderRequestOptions,
): InteractionParams => {
  const params: InteractionParams = {
    model: modelName,
    input: conversation(messages).map((message) => ({
      type: message.role === "assistant" ? "model_output" : "user_input",
      content: [{ type: "text", text: message.content }],
    })),
    store: false,
  };
  const system = systemPrompt(messages);
  if (system !== "") params.system_instruction = system;
  if (options.structured)
    params.response_format = {
      type: "text",
      mime_type: "application/json",
      schema: options.schema,
    };
  return params;
};

/** Call the native Gemini Interactions API. */
class GeminiProvider implements Provider {
  readonly modelName: string;
  readonly client: GoogleGenAI;

  constructor(modelName: string, options: GeminiProviderOptions = {}) {
    this.modelName = modelName;
    this.client = new GoogleGenAI({
      apiKey: options.apiKey,
      httpOptions: { baseUrl: options.baseUrl, fetch: options.fetch },
    });
  }

  close(): void {
    // No-op: this SDK version owns no connection pool to release.
  }

  /** Perform one Interactions request and return its parsed text and usage. */
  async request(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    return translating(async () => {
      const params = interactionParams(this.modelName, messages, options);
      recordRequest(params, { api: "interactions" });
      // The Interactions client retries up to four times by default and
      // exposes no constructor-level switch, so every call disables its
      // retry policy and the adapter's RetryPolicy owns all attempts.
      const response = await this.client.interactions.create(params, {
        retries: { strategy: "none" },
      });
      return geminiResult(response);
    }, translateError);
  }

  /** Map an exception from this provider's SDK to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    return translateError(error);
  }
}

export {
  GeminiProvider,
  type GeminiProviderOptions,
  geminiResult,
  interactionParams,
  translateError as translateGeminiError,
};
