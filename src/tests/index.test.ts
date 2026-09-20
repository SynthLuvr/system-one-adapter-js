import { describe, expect, it } from "vitest";
import {
  AnthropicProvider,
  APIError,
  buildProvider,
  choice,
  Message,
  noul,
  OpenAIProvider,
  ProviderResult,
  SystemOneAdapterClient,
  score,
  TypeSafeError,
} from "../index.js";
import {
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
    expect(buildProvider("openai", "test-model")).toBeInstanceOf(
      OpenAIProvider,
    );
    expect(buildProvider("anthropic", "test-model")).toBeInstanceOf(
      AnthropicProvider,
    );
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
});
