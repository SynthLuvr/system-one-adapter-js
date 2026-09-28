import { describe, expect, it } from "vitest";
import {
  AnthropicProvider,
  APIError,
  buildProvider,
  ClaudeCodeProvider,
  choice,
  GeminiProvider,
  LAYA_MODELS,
  LayaProvider,
  Message,
  noul,
  OpenAIProvider,
  ProviderResult,
  SystemOneAdapterClient,
  score,
  TypeSafeError,
} from "../index.js";
import {
  geminiEndpoint,
  geminiPayload,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  server,
} from "./msw.js";

// Provider SDKs demand a key when the client is built, before any HTTP happens.
process.env.OPENAI_API_KEY ??= "public-api-test";
process.env.ANTHROPIC_API_KEY ??= "public-api-test";

describe("public API", () => {
  it("exports the client, providers, and question factories", () => {
    expect(SystemOneAdapterClient).toBeTypeOf("function");
    expect(OpenAIProvider).toBeTypeOf("function");
    expect(AnthropicProvider).toBeTypeOf("function");
    expect(LayaProvider).toBeTypeOf("function");
    expect(buildProvider("openai", "test-model")).toBeInstanceOf(
      OpenAIProvider,
    );
    expect(buildProvider("anthropic", "test-model")).toBeInstanceOf(
      AnthropicProvider,
    );
    expect(buildProvider("gemini", "gemini-3.8-flash")).toBeInstanceOf(
      GeminiProvider,
    );
    expect(buildProvider("claude_code", "test-model")).toBeInstanceOf(
      ClaudeCodeProvider,
    );
    expect(buildProvider("laya", "router")).toBeInstanceOf(LayaProvider);
    expect(LAYA_MODELS).toEqual([
      "router",
      "english",
      "multilingual",
      "typed-decisions",
    ]);
    expect(noul("Yes or no?")).toEqual({
      type: "noul",
      instructions: "Yes or no?",
      criteria: undefined,
    });
    expect(score("Rating.", ["Bad.", "Good."])).toEqual({
      type: "score",
      instructions: "Rating.",
      criteria: ["Bad.", "Good."],
    });
    expect(choice("Genre.", { fiction: null })).toEqual({
      type: "choice",
      instructions: "Genre.",
      criteria: { fiction: null },
    });
    expect(new TypeSafeError("boom")).toBeInstanceOf(Error);
    expect(new APIError(418, {}, new Headers())).toBeInstanceOf(TypeSafeError);
    const message: Message = { role: "user", content: "hello" };
    const result: ProviderResult = {
      text: "{}",
      inputTokens: 1,
      outputTokens: 2,
    };
    expect(message.role).toBe("user");
    expect(result.inputTokens).toBe(1);
  });

  it("exposes a provider seam compatible with the client", async () => {
    const endpoint = openAIResponsesEndpoint(() =>
      openAIResponsesPayload('{"answers":{"positive":0.75}}'),
    );
    server.use(endpoint.handler);
    const provider = buildProvider("openai", "test-model");
    const client = new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "probabilities",
      provider: "openai",
      model: provider,
    });
    const response = await client.systemOne({
      state: "A lovely book.",
      questions: { positive: noul("The review is positive.") },
    });
    expect(response.nouls.positive?.noul).toBe(0.75);
    expect(endpoint.requests.length).toBe(1);
    await client.close();
  });

  it("builds and closes an owned gemini provider from constructor defaults", async () => {
    const interactions = geminiEndpoint(() =>
      geminiPayload('{"answers":{"positive":0.75}}'),
    );
    server.use(interactions.handler);
    const client = new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "probabilities",
      provider: "gemini",
      model: "gemini-3.8-flash",
    });
    const response = await client.systemOne({
      state: "A lovely book.",
      questions: { positive: noul("The review is positive.") },
    });
    expect(response.nouls.positive?.noul).toBe(0.75);
    expect(response.model).toBe("gemini-3.8-flash");
    expect(interactions.requests.length).toBe(1);
    await client.close();
  });
});
