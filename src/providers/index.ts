import { AnthropicProvider } from "./anthropic.js";
import type {
  ClosableProvider,
  Provider,
  ProviderName,
  TypedQuestions,
} from "./base.js";
import { type LayaModel, LayaProvider } from "./laya.js";
import { OpenAIProvider } from "./openai.js";

/** Build the provider named by a model selector. */
const buildProvider = (
  provider: ProviderName,
  modelName: string,
): ClosableProvider => {
  if (provider === "openai") return new OpenAIProvider(modelName);
  if (provider === "anthropic") return new AnthropicProvider(modelName);
  // laya model names are a closed set, validated inside the provider.
  return new LayaProvider(modelName as LayaModel);
};

export {
  captureAttempt,
  type LlmAttempt,
  type Message,
  type ProviderRequestOptions,
  type ProviderResult,
} from "./base.js";
export {
  LAYA_MODELS,
  type LayaModel,
  type LayaOptions,
  LayaProvider,
  type PythonResult,
  type PythonRunner,
} from "./laya.js";
export {
  AnthropicProvider,
  buildProvider,
  type ClosableProvider,
  OpenAIProvider,
  type Provider,
  type ProviderName,
  type TypedQuestions,
};
