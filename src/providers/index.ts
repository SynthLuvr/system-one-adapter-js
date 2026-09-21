import { AnthropicProvider } from "./anthropic.js";
import type { ClosableProvider, Provider, ProviderName } from "./base.js";
import {
  ClaudeCodeProvider,
  type ClaudeCodeProviderOptions,
} from "./claude-code.js";
import { OpenAIProvider } from "./openai.js";

/** Build the provider named by a model selector. */
const buildProvider = (
  provider: ProviderName,
  modelName: string,
): ClosableProvider =>
  provider === "openai"
    ? new OpenAIProvider(modelName)
    : provider === "anthropic"
      ? new AnthropicProvider(modelName)
      : new ClaudeCodeProvider(modelName);

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
  ClaudeCodeProvider,
  type ClaudeCodeProviderOptions,
  type ClosableProvider,
  OpenAIProvider,
  type Provider,
  type ProviderName,
};
