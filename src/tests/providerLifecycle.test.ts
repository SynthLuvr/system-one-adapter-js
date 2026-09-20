import { TypeSafeError } from "@typesafe-ai/sdk";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import {
  type Message,
  type Provider,
  type ProviderRequestOptions,
  type ProviderResult,
} from "../providers/base.js";
import { OpenAIProvider } from "../providers/openai.js";

const QUESTIONS = {
  positive: { type: "noul", instructions: "The review is positive." },
} as const;
const RESULT_TEXT = '{"answers":{"positive":true}}';

const noop = (): void => undefined;

/** A closable fake provider whose request can be scripted per call. */
const makeProvider = (
  respond: (
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ) => Promise<ProviderResult> = async () => ({
    text: RESULT_TEXT,
    inputTokens: 11,
    outputTokens: 7,
  }),
): Provider & { close: ReturnType<typeof vi.fn> } => {
  const provider: Provider & { close: ReturnType<typeof vi.fn> } = {
    modelName: "test-model",
    async request(messages, options) {
      return respond(messages, options);
    },
    translateError(error) {
      return new TypeSafeError(String(error));
    },
    close: vi.fn(async () => undefined),
  };
  return provider;
};

/** A client whose owned providers come from `providers` in order. */
const clientWithProviders = (
  providers: Provider[],
  options: Record<string, unknown> = {},
): SystemOneAdapterClient => {
  const client = new SystemOneAdapterClient({
    structuredOutputs: true,
    llmAnswerMode: "discrete",
    provider: "openai",
    ...options,
  } as never);
  let index = 0;
  vi.spyOn(client, "buildProvider").mockImplementation(() => {
    const provider = providers[index];
    index += 1;
    if (provider === undefined) throw new Error("no provider scripted");
    return provider as never;
  });
  return client;
};

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("provider lifecycle", () => {
  it("reuses the owned provider and closes it on close", async () => {
    const provider = makeProvider();
    const client = clientWithProviders([provider]);
    expect(provider.close).not.toHaveBeenCalled();
    for (let index = 0; index < 3; index += 1) {
      const response = await client.systemOne({
        state: "Great book",
        questions: QUESTIONS,
        model: "test-model",
      });
      expect(response.nouls.positive?.noul).toBe(1);
    }
    expect(provider.close).not.toHaveBeenCalled();
    await client.close();
    expect(provider.close).toHaveBeenCalledTimes(1);
    await expect(
      client.systemOne({
        state: "Great book",
        questions: QUESTIONS,
        model: "test-model",
      }),
    ).rejects.toThrow(/closed/);
  });

  it("caches by resolved provider and model, per client", async () => {
    const providers = [makeProvider(), makeProvider(), makeProvider()];
    const client = clientWithProviders(providers);
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      provider: "openai",
      model: "test-model",
    });
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "another-model",
    });
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      provider: "anthropic",
      model: "test-model",
    });
    const second = clientWithProviders([makeProvider()]);
    await second.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    await second.close();
    expect(providers[0].close).not.toHaveBeenCalled();
    await client.close();
    for (const provider of providers)
      expect(provider.close).toHaveBeenCalledTimes(1);
  });

  it("borrows injected providers and closes only owned ones", async () => {
    const injected = makeProvider();
    const owned = makeProvider();
    const client = clientWithProviders([owned]);
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: injected,
    });
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "owned-model",
    });
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: injected,
    });
    await client.close();
    expect(injected.close).not.toHaveBeenCalled();
    expect(owned.close).toHaveBeenCalledTimes(1);
    await expect(
      client.systemOne({ state: "doc", questions: QUESTIONS, model: injected }),
    ).rejects.toThrow(/closed/);
    await (injected.close as () => Promise<void>)();
  });

  it("supports custom providers without close", async () => {
    const provider: Provider = {
      modelName: "test-model",
      async request() {
        return { text: RESULT_TEXT, inputTokens: 11, outputTokens: 7 };
      },
      translateError(error) {
        return new TypeSafeError(String(error));
      },
    };
    const client = new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
    });
    const response = await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: provider,
    });
    expect(response.nouls.positive?.noul).toBe(1);
    await client.close();
  });

  it("closes owned providers when an evaluation fails", async () => {
    const provider = makeProvider(async () => {
      throw new TypeSafeError("failed");
    });
    const client = clientWithProviders([provider]);
    await expect(
      client.systemOne({
        state: "doc",
        questions: QUESTIONS,
        model: "test-model",
      }),
    ).rejects.toThrow("failed");
    await client.close();
    expect(provider.close).toHaveBeenCalledTimes(1);
  });

  it("closes owned providers when disposal runs on an exception", async () => {
    const provider = makeProvider();
    const client = clientWithProviders([provider]);
    await expect(
      (async () => {
        await using clientScope = client;
        const response = await clientScope.systemOne({
          state: "doc",
          questions: QUESTIONS,
          model: "test-model",
        });
        expect(response.nouls.positive?.noul).toBe(1);
        throw new Error("interrupted");
      })(),
    ).rejects.toThrow("interrupted");
    expect(provider.close).toHaveBeenCalledTimes(1);
  });

  it("continues cleanup after a failure and allows retrying it", async () => {
    const providers = [makeProvider(), makeProvider(), makeProvider()];
    const client = clientWithProviders(providers);
    await client.systemOne({ state: "doc", questions: QUESTIONS, model: "a" });
    await client.systemOne({ state: "doc", questions: QUESTIONS, model: "b" });
    await client.systemOne({ state: "doc", questions: QUESTIONS, model: "c" });
    const firstError = new Error("close failed");
    providers[0].close.mockRejectedValueOnce(firstError);
    providers[1].close.mockRejectedValueOnce(new Error("another close failed"));
    await expect(client.close()).rejects.toThrow("close failed");
    expect(providers[2].close).toHaveBeenCalledTimes(1);
    for (const provider of providers)
      expect(provider.close).toHaveBeenCalledTimes(1);
    await expect(
      client.systemOne({ state: "doc", questions: QUESTIONS, model: "a" }),
    ).rejects.toThrow(/closed/);
    await client.close();
    await client.close();
    for (const provider of providers)
      expect(provider.close).toHaveBeenCalledTimes(
        provider === providers[2] ? 1 : 2,
      );
  });

  it("does not construct providers when closed before first use", async () => {
    const client = clientWithProviders([makeProvider()]);
    await client.close();
    await client.close();
    await expect(
      client.systemOne({
        state: "doc",
        questions: QUESTIONS,
        model: "test-model",
      }),
    ).rejects.toThrow(/closed/);
  });

  it("does not cache failed construction", async () => {
    const provider = makeProvider();
    const client = clientWithProviders([provider]);
    const factory = vi
      .spyOn(client, "buildProvider")
      .mockImplementationOnce(() => {
        throw new Error("constructor failed");
      });
    await expect(
      client.systemOne({
        state: "doc",
        questions: QUESTIONS,
        model: "test-model",
      }),
    ).rejects.toThrow("constructor failed");
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    expect(factory).toHaveBeenCalledTimes(2);
    await client.close();
    expect(provider.close).toHaveBeenCalledTimes(1);
  });

  it("captures the environment on first use", async () => {
    const stubFetch = async (): Promise<Response> =>
      new Response(
        JSON.stringify({
          id: "chatcmpl-test",
          object: "chat.completion",
          created: 0,
          model: "test-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: RESULT_TEXT },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    vi.stubEnv("OPENAI_API_KEY", "first-test-key");
    vi.stubEnv("OPENAI_BASE_URL", "https://first.invalid/v1");
    const client = new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
      provider: "openai",
    });
    let provider: OpenAIProvider | undefined;
    vi.spyOn(client, "buildProvider").mockImplementation(() => {
      provider ??= new OpenAIProvider("test-model", { fetch: stubFetch });
      return provider as never;
    });
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    vi.stubEnv("OPENAI_API_KEY", "second-test-key");
    vi.stubEnv("OPENAI_BASE_URL", "https://second.invalid/v1");
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    expect(provider?.client.apiKey).toBe("first-test-key");
    if (provider === undefined) throw new Error("provider not built");
    expect(new URL(provider.client.baseURL).host).toBe("first.invalid");
    const fresh = new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
      provider: "openai",
    });
    const secondProvider = new OpenAIProvider("test-model", {
      fetch: stubFetch,
    });
    vi.spyOn(fresh, "buildProvider").mockReturnValue(secondProvider as never);
    await fresh.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    expect(secondProvider.client.apiKey).toBe("second-test-key");
    expect(new URL(secondProvider.client.baseURL).host).toBe("second.invalid");
  });

  it("makes concurrent close callers wait for the same cleanup", async () => {
    const provider = makeProvider();
    const client = clientWithProviders([provider]);
    await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    let release: () => void = noop;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    provider.close.mockImplementationOnce(async () => {
      await gate;
    });
    const first = client.close();
    const second = client.close();
    let secondDone = false;
    void second.then(() => {
      secondDone = true;
    });
    await new Promise((resolve) => {
      setTimeout(resolve, 10);
    });
    expect(secondDone).toBe(false);
    release();
    await first;
    await second;
    expect(provider.close).toHaveBeenCalledTimes(1);
  });

  it("isolates traces across concurrent first uses", async () => {
    let started = 0;
    let release: () => void = noop;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const provider = makeProvider(async () => {
      started += 1;
      if (started === 8) release();
      await gate;
      return { text: RESULT_TEXT, inputTokens: 11, outputTokens: 7 };
    });
    const client = clientWithProviders([provider]);
    const responses = await Promise.all(
      Array.from({ length: 8 }, (_, index) =>
        client.systemOne({
          state: `document-${index}`,
          questions: QUESTIONS,
          model: "test-model",
        }),
      ),
    );
    for (const [index, response] of responses.entries()) {
      expect(response.debug.llm_attempts.length).toBe(1);
      expect(response.debug.llm_attempts[0].messages[1].content).toContain(
        `document-${index}`,
      );
      expect(response.usage.input_tokens_total).toBe(11);
      expect(response.usage.n_retries).toBe(0);
    }
    await client.close();
  });
});
