import { spawn } from "node:child_process";
import { TypeSafeError } from "@typesafe-ai/sdk";
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

/** One laya checkpoint name. */
type LayaModel = (typeof LAYA_MODELS)[number];

/** Options for constructing a laya provider. */
interface LayaOptions {
  /** Python interpreter hosting the laya package; default `python3`. */
  python?: string;
}

/** Result of one python one-shot invocation. */
interface PythonResult {
  stdout: string;
  stderr: string;
  code: number;
}

/**
 * Runs a python script with stdin, resolving even when the process cannot
 * spawn (exit code -1) — failures surface through the validated result.
 */
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

/**
 * One-shot python script: read {model, mode, state, questions} from stdin,
 * run the laya engine, print adapter-shaped answers to stdout.
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
    subfolder = None if model == "english" else model
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
    const child = spawn(python, ["-c", script]);
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error) =>
      resolve({ stdout: "", stderr: `${error.message}\n${stderr}`, code: -1 }),
    );
    child.on("close", (code) => resolve({ stdout, stderr, code: code ?? -1 }));
    // Python may exit before draining stdin; the EPIPE it raises is noise.
    child.stdin.on("error", () => undefined);
    child.stdin.end(stdin, "utf8");
  });

/** Reject a failed python run or stdout that is not JSON. */
const validatePythonResult = (result: PythonResult): void => {
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
      `laya python process printed invalid JSON: ${describeError(error)}`,
    );
  }
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

  constructor(model: LayaModel, options: LayaOptions = {}) {
    if (!(LAYA_MODELS as readonly string[]).includes(model))
      throw new Error(`laya model must be one of ${LAYA_MODELS.join(", ")}`);
    this.modelName = `laya/${model}`;
    this.#model = model;
    this.#python = options.python ?? process.env.LAYA_PYTHON ?? "python3";
  }

  /** No-op: each request spawns a fresh short-lived process. */
  close(): void {
    // Nothing to release.
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
      questions[questionId] = toLayaQuestion(question);
    const result = await defaultRunner(
      this.#python,
      LAYA_SCRIPT,
      JSON.stringify({
        model: this.#model,
        mode: typed.answerMode,
        state: typed.state,
        questions,
      }),
    );
    validatePythonResult(result);
    // A local encoder has no token metering; the adapter still measures
    // latency, but cost columns stay excluded.
    return { text: result.stdout, inputTokens: 0, outputTokens: 0 };
  }

  /** Map a laya process failure to an SDK error. */
  translateError(error: unknown): TypeSafeError {
    if (error instanceof TypeSafeError) return error;
    return new TypeSafeError(describeError(error));
  }
}

export type { LayaModel, LayaOptions, PythonResult, PythonRunner };
export { defaultRunner, LAYA_MODELS, LAYA_SCRIPT, LayaProvider };
