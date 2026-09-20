import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import type {
  ClosableProvider,
  Message,
  Provider,
  ProviderName,
  ProviderRequestOptions,
  ProviderResult,
} from "../providers/index.js";
import { OpenAIProvider } from "../providers/openai.js";
import {
  anthropicEndpoint,
  anthropicPayload,
  jsonResponseError,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  server,
} from "./msw.js";

const QUESTIONS = {
  positive: { type: "noul", instructions: "The review is positive." },
} as const;
const ANSWER = '{"answers":{"positive":true}}';

/** Serve every OpenAI and Anthropic request with a valid discrete answer. */
const serveAnswers = (): void => {
  server.use(
    openAIResponsesEndpoint(() => openAIResponsesPayload(ANSWER)).handler,
    anthropicEndpoint(() => anthropicPayload(ANSWER)).handler,
  );
};

/** A real provider that records close() calls and delegates everything else. */
class RecordingCloseProvider implements ClosableProvider {
  closeCalls = 0;

  constructor(readonly inner: Provider & Partial<ClosableProvider>) {}

  get modelName(): string {
    return this.inner.modelName;
  }

  async request(
    messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    return this.inner.request(messages, options);
  }

  translateError(error: unknown): ReturnType<Provider["translateError"]> {
    return this.inner.translateError(error);
  }

  close(): void {
    this.closeCalls += 1;
    void this.inner.close?.();
  }
}

/** A recorded provider whose first `failures` close() calls throw. */
class FailingCloseProvider extends RecordingCloseProvider {
  failures: number;

  constructor(inner: Provider & Partial<ClosableProvider>, failures: number) {
    super(inner);
    this.failures = failures;
  }

  close(): void {
    super.close();
    if (this.failures > 0) {
      this.failures -= 1;
      throw new Error("close failed");
    }
  }
}

/** A provider whose construction really fails, like a bad credential flow. */
class UnconstructibleProvider extends OpenAIProvider {
  constructor() {
    super("test-model", { apiKey: "test-key" });
    throw new Error("constructor failed");
  }
}

/** A client whose owned providers are built from a scripted factory list. */
class ScriptedProviderClient extends SystemOneAdapterClient {
  buildCalls = 0;
  readonly built: RecordingCloseProvider[] = [];
  readonly #factories: readonly ((
    provider: ProviderName,
    model: string,
  ) => Provider & Partial<ClosableProvider>)[];

  constructor(
    factories: readonly ((
      provider: ProviderName,
      model: string,
    ) => Provider & Partial<ClosableProvider>)[],
    options: Record<string, unknown> = {},
  ) {
    super({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
      provider: "openai",
      ...options,
    } as never);
    this.#factories = factories;
  }

  buildProvider(provider: ProviderName, model: string): ClosableProvider {
    this.buildCalls += 1;
    const factory = this.#factories[this.buildCalls - 1];
    if (factory === undefined) throw new Error("no provider scripted");
    const inner = factory(provider, model);
    const recording = new RecordingCloseProvider(inner);
    this.built.push(recording);
    return recording;
  }
}

/** A real provider of `name` that talks to the MSW-intercepted endpoint. */
const liveProvider = (
  name: ProviderName,
  model: string,
): Provider & Partial<ClosableProvider> =>
  name === "openai"
    ? new OpenAIProvider(model, { apiKey: "test-key" })
    : new AnthropicProvider(model, { apiKey: "test-key" });

/** A scripted client with `n` fresh live providers, in build order. */
const clientWithProviders = (
  n: number,
  options: Record<string, unknown> = {},
): ScriptedProviderClient =>
  new ScriptedProviderClient(
    Array.from({ length: n }, () => liveProvider),
    options,
  );

describe("provider lifecycle", () => {
  it("reuses the owned provider and closes it on close", async () => {
    serveAnswers();
    const client = clientWithProviders(1);
    for (let index = 0; index < 3; index += 1) {
      const response = await client.systemOne({
        state: "Great book",
        questions: QUESTIONS,
        model: "test-model",
      });
      expect(response.nouls.positive?.noul).toBe(1);
    }
    const provider = client.built[0];
    expect(client.buildCalls).toBe(1);
    expect(provider.closeCalls).toBe(0);
    await client.close();
    expect(provider.closeCalls).toBe(1);
    await expect(
      client.systemOne({
        state: "Great book",
        questions: QUESTIONS,
        model: "test-model",
      }),
    ).rejects.toThrow(/closed/);
  });

  it("caches by resolved provider and model, per client", async () => {
    serveAnswers();
    const client = clientWithProviders(3);
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
    expect(client.buildCalls).toBe(3);

    const second = clientWithProviders(1);
    await second.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: "test-model",
    });
    await second.close();
    expect(client.built[0].closeCalls).toBe(0);
    await client.close();
    for (const provider of client.built) expect(provider.closeCalls).toBe(1);
  });

  it("borrows injected providers and closes only owned ones", async () => {
    serveAnswers();
    const injected = new RecordingCloseProvider(
      liveProvider("openai", "injected-model"),
    );
    const client = clientWithProviders(1);
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
    expect(client.buildCalls).toBe(1);
    await client.close();
    expect(injected.closeCalls).toBe(0);
    expect(client.built[0].closeCalls).toBe(1);
    await expect(
      client.systemOne({ state: "doc", questions: QUESTIONS, model: injected }),
    ).rejects.toThrow(/closed/);
    injected.close();
    expect(injected.closeCalls).toBe(1);
  });

  it("supports custom providers without close", async () => {
    serveAnswers();
    /** A custom provider that delegates over HTTP but owns nothing to free. */
    const closeless: Provider = {
      modelName: "closeless-model",
      request: (messages, options) =>
        liveProvider("openai", "test-model").request(messages, options),
      translateError: (error) =>
        liveProvider("openai", "test-model").translateError(error),
    };
    const client = new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "discrete",
    });
    const response = await client.systemOne({
      state: "doc",
      questions: QUESTIONS,
      model: closeless,
    });
    expect(response.nouls.positive?.noul).toBe(1);
    await client.close();
  });

  it("closes owned providers when an evaluation fails", async () => {
    server.use(
      openAIResponsesEndpoint(() => jsonResponseError(400, "bad request"))
        .handler,
    );
    const client = clientWithProviders(1);
    await expect(
      client.systemOne({
        state: "doc",
        questions: QUESTIONS,
        model: "test-model",
      }),
    ).rejects.toThrow(/bad request/);
    await client.close();
    expect(client.built[0].closeCalls).toBe(1);
  });

  it("closes owned providers when disposal runs on an exception", async () => {
    serveAnswers();
    const client = clientWithProviders(1);
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
    expect(client.built[0].closeCalls).toBe(1);
  });

  it("continues cleanup after a failure and allows retrying it", async () => {
    serveAnswers();
    const client = new ScriptedProviderClient([
      (provider, model) =>
        new FailingCloseProvider(liveProvider(provider, model), 1),
      (provider, model) =>
        new FailingCloseProvider(liveProvider(provider, model), 1),
      (provider, model) => liveProvider(provider, model),
    ]);
    await client.systemOne({ state: "doc", questions: QUESTIONS, model: "a" });
    await client.systemOne({ state: "doc", questions: QUESTIONS, model: "b" });
    await client.systemOne({ state: "doc", questions: QUESTIONS, model: "c" });
    const [first, second, third] = client.built;

    await expect(client.close()).rejects.toThrow("close failed");
    expect(third.closeCalls).toBe(1);
    for (const provider of client.built) expect(provider.closeCalls).toBe(1);
    await expect(
      client.systemOne({ state: "doc", questions: QUESTIONS, model: "a" }),
    ).rejects.toThrow(/closed/);

    await client.close();
    await client.close();
    expect(first.closeCalls).toBe(2);
    expect(second.closeCalls).toBe(2);
    expect(third.closeCalls).toBe(1);
  });

  it("does not construct providers when closed before first use", async () => {
    serveAnswers();
    const client = clientWithProviders(1);
    await client.close();
    await client.close();
    await expect(
      client.systemOne({
        state: "doc",
        questions: QUESTIONS,
        model: "test-model",
      }),
    ).rejects.toThrow(/closed/);
    expect(client.buildCalls).toBe(0);
  });

  it("does not cache failed construction", async () => {
    serveAnswers();
    const client = new ScriptedProviderClient([
      () => new UnconstructibleProvider(),
      (provider, model) => liveProvider(provider, model),
    ]);
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
    expect(client.buildCalls).toBe(2);
    await client.close();
    expect(client.built[0].closeCalls).toBe(1);
  });
});
