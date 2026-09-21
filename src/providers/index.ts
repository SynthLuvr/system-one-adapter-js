import { AnthropicProvider } from "./anthropic.js";
import type { ClosableProvider, Provider, ProviderName } from "./base.js";
import {
  ClaudeCodeProvider,
  type ClaudeCodeProviderOptions,
} from "./claude-code.js";
import { OpenAIProvider } from "./openai.js";

/** The provider classes a model selector can name. */
const providerClasses = {
  openai: OpenAIProvider,
  anthropic: AnthropicProvider,
  claude_code: ClaudeCodeProvider,
} as const;

/** Build the provider named by a model selector. */
const buildProvider = (
  provider: ProviderName,
  modelName: string,
): ClosableProvider => new providerClasses[provider](modelName);

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
