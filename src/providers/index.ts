import { AnthropicProvider } from "./anthropic.js";
import type { ClosableProvider, Provider, ProviderName } from "./base.js";
import { OpenAIProvider } from "./openai.js";

/** Build the provider named by a model selector. */
const buildProvider = (
  provider: ProviderName,
  modelName: string,
): ClosableProvider =>
  provider === "openai"
    ? new OpenAIProvider(modelName)
    : new AnthropicProvider(modelName);

export {
  captureAttempt,
  type LlmAttempt,
  type Message,
  type ProviderRequestOptions,
  type ProviderResult,
} from "./base.js";
export {
  AnthropicProvider,
  buildProvider,
  type ClosableProvider,
  OpenAIProvider,
  type Provider,
  type ProviderName,
};
