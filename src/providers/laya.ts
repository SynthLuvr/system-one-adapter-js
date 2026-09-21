import { spawn } from "node:child_process";
import { TypeSafeError } from "@typesafe-ai/sdk";
import { type Question, serializeInstructionValue } from "../schema.js";
import type { AnswerMode } from "../utils/probabilityNormalization.js";
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

/** One laya checkpoint name. */
type LayaModel = (typeof LAYA_MODELS)[number];

/** Options for constructing a laya provider. */
interface LayaOptions {
  /** Python interpreter hosting the laya package; default `python3`. */
  readonly python?: string;
  /** Runs the one-shot python process; injectable for offline tests. */
  readonly runPython?: PythonRunner;
}

/** Result of one python one-shot invocation. */
interface PythonResult {
  readonly stdout: string;
  readonly stderr: string;
  readonly code: number;
}

/** Runs a python script with stdin, resolving when the process exits. */
type PythonRunner = (
  python: string,
  script: string,
  stdin: string,
) => Promise<PythonResult>;

/** One laya question in the python package's native format. */
type LayaQuestion =
  | { type: "choice"; instructions: string; criteria: Record<string, string> }
  | { type: "score"; instructions: string; criteria: string[] }
  | { type: "noul"; instructions: string };

/** Payload handed to the python bridge on stdin. */
interface LayaPayload {
  readonly model: LayaModel;
  readonly mode: AnswerMode;
  readonly state: unknown;
  readonly questions: Record<string, LayaQuestion>;
}

/**
 * One-shot python script: read {model, mode, state, questions} from stdin,
 * run the laya engine, print adapter-shaped answers to stdout. laya answers
 * typed questions natively (choice and score probabilities, noul
 * probabilities), so no text generation is involved.
 */
const LAYA_SCRIPT = `
import json, sys

payload = json.load(sys.stdin)
model = payload["model"]
mode = payload["mode"]
state = payload["state"]
questions = payload["questions"]

if model == "router":
    from laya import Router
    engine = Router().predict
else:
    import laya
    subfolder = {"english": None, "multilingual": "multilingual",
                 "typed-decisions": "typed-decisions"}[model]
    if subfolder is None:
        engine = laya.load("convaiinnovations/laya").predict
    else:
        engine = laya.load("convaiinnovations/laya", subfolder=subfolder).predict
result = engine(state, questions)

answers = {}
for question_id, question in questions.items():
    answer = result["answers"][question_id]
    if question["type"] == "noul":
        answers[question_id] = (
            bool(answer["noul"] >= 0.5) if mode == "discrete"
            else float(answer["noul"])
        )
    elif mode == "discrete":
        if question["type"] == "score":
            answers[question_id] = int(round(float(answer["score"])))
        else:
            answers[question_id] = answer["choice"]
    else:
        answers[question_id] = answer.get("probabilities", {})
print(json.dumps({"answers": answers}))
`;

/** Spawn the python one-shot process and collect its output. */
const defaultRunner: PythonRunner = (python, script, stdin) =>
  new Promise((resolve) => {
    const child = spawn(python, ["-c", script], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) =>
      resolve({
        stdout: "",
        stderr: `${error.message}\n${stderr}`,
        code: -1,
      }),
    );
    child.on("close", (code) => resolve({ stdout, stderr, code: code ?? -1 }));
    child.stdin.on("error", () => undefined);
    child.stdin.end(stdin, "utf8");
  });

/** One laya question, built from a validated adapter question. */
const layaQuestion = (question: Question): LayaQuestion => {
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
  const criteria = question.criteria ?? null;
  if (criteria === null) return { type: "noul", instructions };
  // Mirror the prompt text an LLM judge would see, so a laya judge
  // evaluates the same question an LLM provider is prompted with.
  const trueCriteria = serializeInstructionValue(criteria.true ?? null);
  const falseCriteria = serializeInstructionValue(criteria.false ?? null);
  return {
    type: "noul",
    instructions: `${instructions}\nTrue criteria: ${trueCriteria}\nFalse criteria: ${falseCriteria}`,
  };
};

/**
 * Provider running the local laya decision engine
 * (github.com/NandhaKishorM/laya, `pip install laya`) through a one-shot
 * python process per request. laya answers the adapter's typed questions
 * natively — choice, score, and noul — so no text generation is involved
 * and token counts stay zero.
 */
class LayaProvider implements ClosableProvider {
  readonly modelName: string;
  readonly #model: LayaModel;
  readonly #python: string;
  readonly #runner: PythonRunner;

  constructor(model: LayaModel, options: LayaOptions = {}) {
    if (!(LAYA_MODELS as readonly string[]).includes(model))
      throw new Error(`laya model must be one of ${LAYA_MODELS.join(", ")}`);
    this.modelName = `laya/${model}`;
    this.#model = model;
    this.#python = options.python ?? process.env.LAYA_PYTHON ?? "python3";
    this.#runner = options.runPython ?? defaultRunner;
  }

  /** No-op: each request spawns a fresh short-lived process. */
  close(): void {
    // Nothing to release; python processes exit with each request.
  }

  /** Run one evaluation through the local laya package. */
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
      questions[questionId] = layaQuestion(question);
    const payload: LayaPayload = {
      model: this.#model,
      mode: typed.answerMode,
      state: typed.state,
      questions,
    };
    const result = await this.#runner(
      this.#python,
      LAYA_SCRIPT,
      JSON.stringify(payload),
    );
    if (result.code !== 0) {
      const detail = result.stderr.trim().split("\n").slice(-4).join(" | ");
      throw new TypeSafeError(
        `laya python process exited ${result.code}: ${detail}`,
      );
    }
    try {
      JSON.parse(result.stdout);
    } catch (error) {
      throw new TypeSafeError(
        `laya python process printed invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    // A local encoder has no token metering; the adapter still measures
    // latency, but cost columns stay excluded.
    return { text: result.stdout, inputTokens: 0, outputTokens: 0 };
  }

  /** Map a laya process failure to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    if (error instanceof TypeSafeError) return error;
    return new TypeSafeError(
      error instanceof Error ? error.message : String(error),
    );
  }
}

export type {
  LayaModel,
  LayaOptions,
  LayaPayload,
  LayaQuestion,
  PythonResult,
  PythonRunner,
};
export { defaultRunner, LAYA_MODELS, LAYA_SCRIPT, LayaProvider };
