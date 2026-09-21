import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { choice, noul, score } from "@typesafe-ai/sdk";
import { beforeAll, describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import {
  LAYA_MODELS,
  type LayaModel,
  LayaProvider,
} from "../providers/index.js";
import {
  defaultRunner,
  LAYA_SCRIPT,
  type PythonResult,
} from "../providers/laya.js";

/** The stdin payload handed to the python bridge. */
type Payload = {
  model: string;
  mode: string;
  state: unknown;
  questions: Record<
    string,
    {
      type: string;
      instructions: string;
      criteria?: Record<string, string> | string[];
    }
  >;
};

/**
 * Fake runner standing in for python + laya: it answers the received
 * questions the way the embedded script maps laya results.
 */
const fakeRunner = async (
  _python: string,
  script: string,
  stdin: string,
): Promise<PythonResult> => {
  expect(script).toBe(LAYA_SCRIPT);
  const payload = JSON.parse(stdin) as Payload;
  const layaAnswers: Record<string, Record<string, unknown>> = {};
  for (const [questionId, question] of Object.entries(payload.questions))
    if (question.type === "choice") {
      const labels = Object.keys(question.criteria as Record<string, string>);
      layaAnswers[questionId] = {
        type: "choice",
        choice: labels[0],
        confidence: 0.6,
        probabilities: Object.fromEntries(
          labels.map((label, index) => [
            label,
            index === 0 ? 0.6 : 0.4 / Math.max(labels.length - 1, 1),
          ]),
        ),
      };
    } else if (question.type === "score") {
      layaAnswers[questionId] = {
        type: "score",
        score: 1.6,
        confidence: 0.5,
        probabilities: Object.fromEntries(
          (question.criteria as string[]).map((_, index) => [
            String(index),
            1 / (question.criteria as string[]).length,
          ]),
        ),
      };
    } else {
      layaAnswers[questionId] = { type: "noul", noul: 0.83 };
    }

  const answers: Record<string, unknown> = {};
  for (const [questionId, laya] of Object.entries(layaAnswers))
    if (payload.mode === "discrete") {
      if (payload.questions[questionId]?.type === "noul")
        answers[questionId] = (laya.noul as number) >= 0.5;
      else if (payload.questions[questionId]?.type === "score")
        answers[questionId] = Math.round(laya.score as number);
      else answers[questionId] = laya.choice;
    } else if (payload.questions[questionId]?.type === "noul") {
      answers[questionId] = laya.noul;
    } else {
      answers[questionId] = laya.probabilities;
    }

  return { stdout: JSON.stringify({ answers }), stderr: "", code: 0 };
};

/** A runner whose python process fails with the given stderr. */
const failingRunner = async (): Promise<PythonResult> => ({
  stdout: "",
  stderr:
    "Traceback (most recent call last):\nModuleNotFoundError: No module named 'laya'",
  code: 1,
});

/** A client evaluating through the given runner-backed provider. */
const client = (
  runPython = fakeRunner,
  llmAnswerMode: "probabilities" | "discrete" = "probabilities",
): SystemOneAdapterClient =>
  new SystemOneAdapterClient({
    structuredOutputs: true,
    llmAnswerMode,
    normalizeProbabilities: true,
    model: new LayaProvider("router", { runPython }),
  });

const QUESTIONS = {
  verdict: choice("Which response is better?", {
    A: "Assistant 1 is better",
    B: "Assistant 2 is better",
    tie: "Equally good",
  }),
  is_safe: noul("Is the exchange safe?", {
    true: "No harmful content",
    false: "Contains harmful content",
  }),
  rating: score("How helpful is the response?", [
    "Unhelpful",
    "Somewhat helpful",
    "Very helpful",
  ]),
};

describe("LayaProvider", () => {
  it("evaluates typed questions through the python bridge", async () => {
    const received: Payload[] = [];
    const spyRunner = async (
      python: string,
      script: string,
      stdin: string,
    ): Promise<PythonResult> => {
      received.push(JSON.parse(stdin) as Payload);
      return fakeRunner(python, script, stdin);
    };
    const adapter = client(spyRunner);
    const response = await adapter.systemOne({
      state: { user_message: "hi", assistant_1: "a", assistant_2: "b" },
      questions: QUESTIONS,
    });
    await adapter.close();

    expect(received).toHaveLength(1);
    expect(received[0]?.model).toBe("router");
    expect(received[0]?.mode).toBe("probabilities");
    expect(received[0]?.state).toEqual({
      user_message: "hi",
      assistant_1: "a",
      assistant_2: "b",
    });
    const verdict = received[0]?.questions.verdict;
    expect(verdict?.type).toBe("choice");
    expect(verdict?.instructions).toBe("Which response is better?");
    expect(verdict?.criteria).toEqual({
      A: "Assistant 1 is better",
      B: "Assistant 2 is better",
      tie: "Equally good",
    });
    const isSafe = received[0]?.questions.is_safe;
    expect(isSafe?.type).toBe("noul");
    expect(isSafe?.instructions).toBe(
      "Is the exchange safe?\nTrue criteria: No harmful content\n" +
        "False criteria: Contains harmful content",
    );
    const rating = received[0]?.questions.rating;
    expect(rating?.type).toBe("score");
    expect(rating?.criteria).toEqual([
      "Unhelpful",
      "Somewhat helpful",
      "Very helpful",
    ]);

    expect(response.model).toBe("laya/router");
    expect(response.usage.input_tokens_total).toBe(0);
    expect(response.usage.output_tokens_total).toBe(0);
    expect(response.nouls.is_safe?.noul).toBeCloseTo(0.83, 5);
    expect(response.choices.verdict?.choice).toBe("A");
    expect(response.choices.verdict?.probabilities?.A).toBeCloseTo(0.6, 5);
  });

  it("answers score questions with laya's native score support", async () => {
    const adapter = client();
    const response = await adapter.systemOne({
      state: { body: "A thoughtful and complete answer." },
      questions: { rating: QUESTIONS.rating },
    });
    await adapter.close();
    const answer = response.scores.rating;
    expect(answer?.score).toBeCloseTo(1, 5);
    expect(answer?.probabilities).toEqual({ 0: 1 / 3, 1: 1 / 3, 2: 1 / 3 });
    expect(answer?.legend).toEqual({
      0: "Unhelpful",
      1: "Somewhat helpful",
      2: "Very helpful",
    });
  });

  it("maps discrete answers: labels, boolean nouls, rounded scores", async () => {
    const adapter = client(fakeRunner, "discrete");
    const response = await adapter.systemOne({
      state: { body: "text" },
      questions: QUESTIONS,
    });
    await adapter.close();
    expect(response.choices.verdict?.choice).toBe("A");
    expect(response.nouls.is_safe?.noul).toBe(1);
    // The stub's 1.6 expected score rounds to level 2.
    expect(response.scores.rating?.score).toBe(2);
  });

  it("rejects unknown laya models", () => {
    expect(() => new LayaProvider("flash" as LayaModel)).toThrow(
      `laya model must be one of ${LAYA_MODELS.join(", ")}`,
    );
  });

  it("honors the LAYA_PYTHON environment variable", async () => {
    const previous = process.env.LAYA_PYTHON;
    process.env.LAYA_PYTHON = "laya-test-python";
    try {
      let seenPython = "";
      const recorder = async (
        python: string,
        script: string,
        stdin: string,
      ): Promise<PythonResult> => {
        seenPython = python;
        return fakeRunner(python, script, stdin);
      };
      const adapter = new SystemOneAdapterClient({
        structuredOutputs: true,
        llmAnswerMode: "probabilities",
        model: new LayaProvider("english", { runPython: recorder }),
      });
      await adapter.systemOne({
        state: { body: "x" },
        questions: { positive: noul("Good?") },
      });
      await adapter.close();
      expect(seenPython).toBe("laya-test-python");
    } finally {
      if (previous === undefined) delete process.env.LAYA_PYTHON;
      else process.env.LAYA_PYTHON = previous;
    }
  });

  it("surfaces python failures as provider errors", async () => {
    const adapter = client(failingRunner);
    await expect(
      adapter.systemOne({ state: { body: "x" }, questions: QUESTIONS }),
    ).rejects.toThrow(/laya python process exited 1.*ModuleNotFoundError/u);
    await adapter.close();
  });

  it("surfaces a missing laya install through the built-in provider", async () => {
    const adapter = new SystemOneAdapterClient({
      structuredOutputs: true,
      llmAnswerMode: "probabilities",
      provider: "laya",
      model: "router",
    });
    await expect(
      adapter.systemOne({ state: { body: "x" }, questions: QUESTIONS }),
    ).rejects.toThrow(/laya python process exited/u);
    await adapter.close();
  });

  it("rejects python output that is not JSON", async () => {
    const provider = new LayaProvider("router", {
      runPython: async () => ({ stdout: "not json", stderr: "", code: 0 }),
    });
    await expect(
      provider.request([{ role: "user", content: "unused" }], {
        schema: {},
        structured: true,
        typed: {
          state: { body: "x" },
          questions: {
            positive: { type: "noul", instructions: null, criteria: null },
          },
          answerMode: "probabilities",
        },
      }),
    ).rejects.toThrow(/invalid JSON/u);
  });

  it("rejects requests without typed questions", async () => {
    const provider = new LayaProvider("router", { runPython: fakeRunner });
    await expect(
      provider.request([{ role: "user", content: "unused" }], {
        schema: {},
        structured: true,
      }),
    ).rejects.toThrow(/typed questions/u);
  });

  it("maps foreign errors to SDK errors", () => {
    const provider = new LayaProvider("router");
    expect(provider.translateError(new Error("boom")).message).toBe("boom");
    expect(provider.translateError("raw").message).toBe("raw");
  });

  it("reports spawn failures through the default runner", async () => {
    const result = await defaultRunner(
      "adapter-no-such-python",
      LAYA_SCRIPT,
      "{}",
    );
    expect(result.code).toBe(-1);
    expect(result.stderr).toContain("adapter-no-such-python");
  });
});

/** The stub `laya` module the embedded script is exercised against. */
const LAYA_STUB = `
class _Agent:
    def __init__(self, noul_value):
        self.noul_value = noul_value

    def predict(self, state, questions):
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                labels = list(q["criteria"].keys())
                probs = {}
                for i, label in enumerate(labels):
                    probs[label] = 0.6 if i == 0 else 0.4 / max(len(labels) - 1, 1)
                answers[qid] = {"type": "choice", "choice": labels[0],
                                "confidence": 0.6, "probabilities": probs}
            elif q["type"] == "score":
                k = len(q["criteria"])
                probs = {str(i): 1.0 / k for i in range(k)}
                answers[qid] = {"type": "score", "score": 1.6,
                                "probabilities": probs, "confidence": 0.5}
            else:
                answers[qid] = {"type": "noul", "noul": self.noul_value}
        return {"answers": answers}

class Router:
    def __init__(self, **kwargs):
        pass

    def predict(self, state, questions):
        return _Agent(0.83).predict(state, questions)

def load(repo, subfolder=None):
    return _Agent(0.42 if subfolder == "multilingual" else 0.83)
`;

describe("embedded python script", () => {
  let pythonAvailable = false;

  beforeAll(async () => {
    pythonAvailable = await new Promise((resolve) => {
      const child = spawn("python3", ["-c", "pass"]);
      child.on("error", () => resolve(false));
      child.on("close", () => resolve(true));
    });
  });

  /** Run the real script against the stubbed laya package. */
  const runScript = async (
    stdin: string,
  ): Promise<{ code: number; stdout: string; stderr: string }> => {
    const dir = await mkdtemp(join(tmpdir(), "adapter-laya-"));
    const previousPath = process.env.PYTHONPATH;
    try {
      await writeFile(join(dir, "laya.py"), LAYA_STUB, "utf8");
      process.env.PYTHONPATH = dir;
      return await defaultRunner("python3", LAYA_SCRIPT, stdin);
    } finally {
      if (previousPath === undefined) delete process.env.PYTHONPATH;
      else process.env.PYTHONPATH = previousPath;
      await rm(dir, { recursive: true, force: true });
    }
  };

  /** The question specs the embedded-script tests run through the stub. */
  const scriptQuestions = (labels: Record<string, string>) => ({
    verdict: {
      type: "choice",
      instructions: "Which is better?",
      criteria: labels,
    },
    rating: {
      type: "score",
      instructions: "How helpful?",
      criteria: ["bad", "ok", "good"],
    },
    urgent: { type: "noul", instructions: "Urgent?" },
  });

  /** Run the script and parse its answers, asserting a clean exit. */
  const scriptAnswers = async (
    stdin: string,
  ): Promise<Record<string, unknown>> => {
    const result = await runScript(stdin);
    expect(result.code).toBe(0);
    return (JSON.parse(result.stdout) as { answers: Record<string, unknown> })
      .answers;
  };

  it("maps router answers to probabilities", async () => {
    if (!pythonAvailable) return;
    const answers = await scriptAnswers(
      JSON.stringify({
        model: "router",
        mode: "probabilities",
        state: { body: "duplicate charge" },
        questions: scriptQuestions({ A: "first", B: "second", tie: "equal" }),
      }),
    );
    expect(answers.verdict).toEqual({ A: 0.6, B: 0.2, tie: 0.2 });
    expect(answers.rating).toEqual({ 0: 1 / 3, 1: 1 / 3, 2: 1 / 3 });
    expect(answers.urgent).toBe(0.83);
  });

  it("maps checkpoint answers to discrete values", async () => {
    if (!pythonAvailable) return;
    const answers = await scriptAnswers(
      JSON.stringify({
        model: "english",
        mode: "discrete",
        state: { body: "duplicate charge" },
        questions: scriptQuestions({ A: "first", B: "second" }),
      }),
    );
    expect(answers.verdict).toBe("A");
    expect(answers.rating).toBe(2);
    expect(answers.urgent).toBe(true);
  });

  it("loads the multilingual checkpoint with its subfolder", async () => {
    if (!pythonAvailable) return;
    const answers = await scriptAnswers(
      JSON.stringify({
        model: "multilingual",
        mode: "probabilities",
        state: { body: "doppelte Abbuchung" },
        questions: { urgent: { type: "noul", instructions: "Urgent?" } },
      }),
    );
    // Only the multilingual stub agent answers 0.42.
    expect(answers.urgent).toBe(0.42);
  });
});
