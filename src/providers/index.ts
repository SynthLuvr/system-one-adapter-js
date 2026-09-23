import { AnthropicProvider } from "./anthropic.js";
import type {
  ClosableProvider,
  Provider,
  ProviderName,
  TypedQuestions,
} from "./base.js";
import {
  ClaudeCodeProvider,
  type ClaudeCodeProviderOptions,
} from "./claude-code.js";
import { type LayaModel, LayaProvider } from "./laya.js";
import { OpenAIProvider } from "./openai.js";

/** The provider classes a model selector can name. */
const providerClasses = {
  openai: OpenAIProvider,
  anthropic: AnthropicProvider,
  claude_code: ClaudeCodeProvider,
  laya: LayaProvider,
} as const;

/** Build the provider named by a model selector. */
const buildProvider = (
  provider: ProviderName,
  modelName: string,
): ClosableProvider =>
  // laya model names are a closed set, validated inside the provider.
  new providerClasses[provider](modelName as LayaModel);

export {
  captureAttempt,
  type LlmAttempt,
  type Message,
  type ProviderRequestOptions,
  type ProviderResult,
} from "./base.js";
export {
  type EngineModel,
  LAYA_MODELS,
  type LayaModel,
  type LayaModelLocation,
  type LayaOptions,
  LayaProvider,
} from "./laya.js";
export {
  AnthropicProvider,
  buildProvider,
  ClaudeCodeProvider,
  type ClaudeCodeProviderOptions,
  type ClosableProvider,
  OpenAIProvider,
  type Provider,
  type ProviderName,
  type TypedQuestions,
};
