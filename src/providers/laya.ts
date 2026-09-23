import { TypeSafeError } from "@typesafe-ai/sdk";
import {
  Agent,
  type QuestionDef,
  type SystemAnswer,
} from "../laya-ts/agent.js";
import type { SessionProvider } from "../laya-ts/providers.js";
import { type ModelName, Router } from "../laya-ts/router.js";
import {
  noulCriteriaNote,
  type Question,
  serializeInstructionValue,
} from "../schema.js";
import { describeError } from "../utils/errorHandling.js";
import type {
  ClosableProvider,
  Message,
  ProviderRequestOptions,
  ProviderResult,
} from "./base.js";

/** Local laya checkpoints (github.com/NandhaKishorM/laya) a judge can run. */
const LAYA_MODELS = [
  "router",
  "english",
  "multilingual",
  "typed-decisions",
] as const;

/** One laya provider model: the router or a named checkpoint. */
type LayaModel = (typeof LAYA_MODELS)[number];

/** The laya-ts checkpoints an engine can be built from (all but `router`). */
type EngineModel = Exclude<LayaModel, "router">;

/** The default checkpoint bundle all three engines come from. */
const DEFAULT_REPO = "convaiinnovations/laya";

/**
 * Default checkpoint locations, mirroring the official Hugging Face
 * bundle's layout: `english` at the root and the other two in subfolders.
 */
const DEFAULT_LOCATIONS: Record<
  EngineModel,
  { repo: string; subfolder: string | null }
> = {
  english: { repo: DEFAULT_REPO, subfolder: null },
  multilingual: { repo: DEFAULT_REPO, subfolder: "multilingual" },
  "typed-decisions": { repo: DEFAULT_REPO, subfolder: "typed-decisions" },
};

/**
 * Where one checkpoint's exported ONNX bundle lives: either a single
 * location (a local directory exported with `scripts/export_onnx.py`, or
 * a Hugging Face repo id hosting the ONNX files), or an explicit repo id
 * with subfolder.
 */
type LayaModelLocation = string | { repo: string; subfolder?: string | null };

/** Options for constructing a laya provider. */
interface LayaOptions {
  /**
   * Checkpoint location overrides, keyed by engine model name; defaults
   * to the official `convaiinnovations/laya` bundle layout, which does
   * not ship ONNX exports yet.
   */
  models?: Partial<Record<EngineModel, LayaModelLocation>>;
  /** ONNX Runtime device; `"cpu"` (default) or `"cuda"` with CPU fallback. */
  device?: "cpu" | "cuda";
  /** ONNX Runtime intra-op thread count. */
  numThreads?: number;
  /**
   * Prebuilt ONNX session shim replacing engine loading entirely, for
   * tests and custom runtimes; agents built this way use laya's default
   * tokenizer and configuration.
   */
  session?: SessionProvider;
}

/** One adapter answer after engine shaping. */
type AdapterAnswer = boolean | number | string | Record<string, number>;

/** One laya question in the engine's native question definition shape. */
type LayaQuestion =
  | { type: "choice"; instructions: string; criteria: Record<string, string> }
  | { type: "score"; instructions: string; criteria: string[] }
  | { type: "noul"; instructions: string };

/** The resolved engine inputs for one provider model. */
interface ResolvedModel {
  /** Repo id or local directory passed to `Agent.load`. */
  source: string;
  /** Subfolder inside the source bundle, when the source is a repo. */
  subfolder: string | null;
}

/** The state an engine call carries: text or a JSON-serializable object. */
type EngineCall = (
  state: unknown,
  questions: Record<string, QuestionDef>,
) => Promise<{ answers: Record<string, SystemAnswer> }>;

/**
 * Round half to even, matching python's `round`: the old laya provider
 * discretized score answers with `int(round(score))`, and expectations
 * landing exactly on `.5` must keep rounding to the neighboring even
 * level instead of javascript's half-up.
 */
const roundHalfEven = (value: number): number => {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
};

/** One laya question, built from a validated adapter question. */
const toLayaQuestion = (question: Question): LayaQuestion => {
  const instructions = serializeInstructionValue(question.instructions);
  if (question.type === "choice") {
    const criteria: Record<string, string> = {};
    for (const [label, criterion] of Object.entries(question.criteria))
      criteria[label] = serializeInstructionValue(criterion);
    return { type: "choice", instructions, criteria };
  }
  if (question.type === "score")
    return {
      type: "score",
      instructions,
      criteria: question.criteria.map(serializeInstructionValue),
    };
  return {
    type: "noul",
    instructions: `${instructions}${noulCriteriaNote(question)}`,
  };
};

/** Shape one engine answer for discrete mode: labels, booleans, levels. */
const discreteAnswer = (answer: SystemAnswer): AdapterAnswer => {
  if (answer.type === "choice") return answer.choice;
  if (answer.type === "score") return roundHalfEven(answer.score);
  return answer.noul >= 0.5;
};

/** Shape one engine answer for probabilistic mode. */
const probabilisticAnswer = (answer: SystemAnswer): AdapterAnswer => {
  if (answer.type === "noul") return answer.noul;
  return answer.probabilities;
};

/**
 * Provider running the local laya decision engine
 * (github.com/NandhaKishorM/laya) in-process through the vendored
 * `laya-ts` port (`src/laya-ts`): a BPE tokenizer, a pure string
 * sequence builder, and ONNX Runtime sessions over the exported
 * `encoder.onnx` + `head.onnx` weights. laya answers the adapter's typed
 * questions natively — choice, score, and noul — so no text generation
 * is involved and token counts stay zero.
 *
 * Unlike the python package it replaces, the engine cannot read
 * safetensors checkpoints, so the ONNX export must exist: either in the
 * directories named by the `models` option or `LAYA_MODEL_DIR`, or on
 * the Hugging Face hub at the location a checkpoint resolves to.
 */
class LayaProvider implements ClosableProvider {
  readonly modelName: string;
  readonly #model: LayaModel;
  readonly #device: "cpu" | "cuda";
  readonly #numThreads: number | undefined;
  readonly #session: SessionProvider | undefined;
  readonly #locations: Partial<Record<EngineModel, LayaModelLocation>>;
  readonly #envDir: string | undefined;
  readonly #agents = new Map<EngineModel, Agent>();
  #router: Router | undefined;

  constructor(model: LayaModel, options: LayaOptions = {}) {
    if (!(LAYA_MODELS as readonly string[]).includes(model))
      throw new Error(`laya model must be one of ${LAYA_MODELS.join(", ")}`);
    this.modelName = `laya/${model}`;
    this.#model = model;
    this.#device = options.device ?? "cpu";
    this.#numThreads = options.numThreads;
    this.#session = options.session;
    this.#locations = options.models ?? {};
    this.#envDir = process.env.LAYA_MODEL_DIR;
  }

  /** No-op: agents are cached lazily and dropped on close. */
  close(): void {
    this.#agents.clear();
    this.#router?.unload();
    this.#router = undefined;
  }

  /** Resolve where one checkpoint's ONNX bundle lives. */
  #resolve(model: EngineModel): ResolvedModel {
    const location = this.#locations[model];
    if (typeof location === "string")
      return { source: location, subfolder: null };
    if (location !== undefined)
      return {
        source: location.repo,
        subfolder: location.subfolder ?? null,
      };
    if (this.#envDir !== undefined)
      return { source: `${this.#envDir}/${model}`, subfolder: null };
    const fallback = DEFAULT_LOCATIONS[model];
    return { source: fallback.repo, subfolder: fallback.subfolder };
  }

  /** Load (or reuse) the agent for one engine model. */
  async #loadAgent(model: EngineModel): Promise<Agent> {
    const cached = this.#agents.get(model);
    if (cached !== undefined) return cached;
    let agent: Agent;
    try {
      agent =
        this.#session !== undefined
          ? new Agent({ provider: this.#session })
          : await this.#loadFromDisk(model);
    } catch (error) {
      throw new TypeSafeError(
        `the laya ${JSON.stringify(model)} checkpoint failed to load: ` +
          `${describeError(error)}; export its ONNX weights once with ` +
          "scripts/export_onnx.py, or point the `models` option or " +
          "LAYA_MODEL_DIR at an exported bundle",
      );
    }
    this.#agents.set(model, agent);
    return agent;
  }

  /** Build the ONNX-backed agent from the checkpoint's resolved location. */
  async #loadFromDisk(model: EngineModel): Promise<Agent> {
    const { source, subfolder } = this.#resolve(model);
    return Agent.load(source, {
      subfolder,
      device: this.#device,
      numThreads: this.#numThreads,
    });
  }

  /** The engine call for this provider's model. */
  async #engine(): Promise<EngineCall> {
    if (this.#model !== "router") {
      const agent = await this.#loadAgent(this.#model);
      return (state, questions) => agent.predict(state, questions);
    }
    // The router picks english or multilingual per state: the same
    // script/stopword detection the python Router runs, with agents
    // loaded through this provider's locations.
    this.#router ??= new Router({
      maxLoaded: 3,
      loader: (name: ModelName) => this.#loadAgent(name),
    });
    const router = this.#router;
    return (state, questions) => router.predict(state, questions);
  }

  /** Run one evaluation through the in-process laya engine. */
  async request(
    _messages: readonly Message[],
    options: ProviderRequestOptions,
  ): Promise<ProviderResult> {
    const typed = options.typed;
    if (typed === undefined)
      throw new TypeSafeError(
        "laya requests carry typed questions on the adapter client's " +
          "provider options; run evaluations through the client instead of " +
          "replaying attempts directly",
      );
    const questions: Record<string, LayaQuestion> = {};
    for (const [questionId, question] of Object.entries(typed.questions))
      questions[questionId] = toLayaQuestion(question);
    const predict = await this.#engine();
    const result = await predict(typed.state, questions);
    const answers: Record<string, AdapterAnswer> = {};
    for (const questionId of Object.keys(questions)) {
      const answer = result.answers[questionId];
      if (answer === undefined)
        throw new TypeSafeError(
          `laya returned no answer for question ${JSON.stringify(questionId)}`,
        );
      answers[questionId] =
        typed.answerMode === "discrete"
          ? discreteAnswer(answer)
          : probabilisticAnswer(answer);
    }
    // A local encoder has no token metering; the adapter still measures
    // latency, but cost columns stay excluded.
    return {
      text: JSON.stringify({ answers }),
      inputTokens: 0,
      outputTokens: 0,
    };
  }

  /** Map an engine failure to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    if (error instanceof TypeSafeError) return error;
    return new TypeSafeError(describeError(error));
  }
}

export type { EngineModel, LayaModel, LayaModelLocation, LayaOptions };
export { LAYA_MODELS, LayaProvider };
