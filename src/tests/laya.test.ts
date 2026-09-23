import { spawn } from "node:child_process";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { choice, noul, score } from "@typesafe-ai/sdk";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import {
  defaultRunner,
  LAYA_MODELS,
  LAYA_SCRIPT,
  type LayaModel,
  LayaProvider,
} from "../providers/laya.js";

/**
 * Every evaluation in this file runs the real laya engine: the provider's
 * embedded python script is spawned for real and answers with the actual
 * `convaiinnovations/laya` checkpoints (downloaded from the Hugging Face
 * hub on first use). Nothing is stubbed, faked, or intercepted.
 */

/** Per-test ceiling sized for a cold Hugging Face checkpoint download. */
const MODEL_TIMEOUT = 600_000;

/** The repo-local venv directory the laya package is installed into. */
const VENV_DIR = join(process.cwd(), "node_modules", ".cache", "laya-venv");

/** The venv interpreter; POSIX and Windows lay it out differently. */
const VENV_PYTHON =
  process.platform === "win32"
    ? join(VENV_DIR, "Scripts", "python.exe")
    : join(VENV_DIR, "bin", "python");

/** Resolve `true` when the given path exists. */
const fileExists = (path: string): Promise<boolean> =>
  stat(path).then(
    () => true,
    () => false,
  );

/** Whether the interpreter exits cleanly on the given arguments. */
const exitsCleanly = (python: string, args: string[]): Promise<boolean> =>
  new Promise((resolve) => {
    const child = spawn(python, args);
    child.on("error", () => resolve(false));
    child.on("close", (code) => resolve(code === 0));
  });

/** Resolve whether the given interpreter has the laya package importable. */
const canImportLaya = (python: string): Promise<boolean> =>
  exitsCleanly(python, [
    "-c",
    "import importlib.util, sys; " +
      "sys.exit(0 if importlib.util.find_spec('laya') is not None else 1)",
  ]);

/** Run the body with LAYA_PYTHON set, restoring the previous value after. */
const withLayaPython = (
  interpreter: string,
  body: () => Promise<void>,
): Promise<void> => {
  const previous = process.env.LAYA_PYTHON;
  process.env.LAYA_PYTHON = interpreter;
  return body().finally(() => {
    if (previous === undefined) delete process.env.LAYA_PYTHON;
    else process.env.LAYA_PYTHON = previous;
  });
};

/**
 * Interpreter hosting the real laya package: an explicit `LAYA_PYTHON`
 * wins, then the repo-local venv under `node_modules/.cache` (invisible
 * to git and the repo tooling), then any `python3` or `python` on PATH.
 */
const resolveLayaPython = async (): Promise<string | undefined> => {
  if (process.env.LAYA_PYTHON !== undefined) return process.env.LAYA_PYTHON;
  if ((await fileExists(VENV_PYTHON)) && (await canImportLaya(VENV_PYTHON)))
    return VENV_PYTHON;
  if (await canImportLaya("python3")) return "python3";
  if (await canImportLaya("python")) return "python";
  return undefined;
};

const layaPython = await resolveLayaPython();

/** How to obtain an interpreter hosting the laya package. */
const INSTALL_HINT =
  "install one with `uv venv node_modules/.cache/laya-venv && " +
  "uv pip install --python node_modules/.cache/laya-venv laya " +
  "--torch-backend=cpu` (or `pip install laya`), or point LAYA_PYTHON " +
  "at an interpreter that has it";

/** The resolved interpreter; engine tests fail fast when it is missing. */
const python = (): string => {
  if (layaPython === undefined)
    throw new Error(`no interpreter with the laya package: ${INSTALL_HINT}`);
  return layaPython;
};

/**
 * Run the body with python's UTF-8 mode disabled and the C locale forced
 * — the codec situation a Windows `cp1252` host presents to the spawned
 * one-shot (surrogateescape turns undecodable bytes into lone
 * surrogates there, `strict` raises here; either way the payload breaks).
 */
const withNonUtf8Host = (body: () => Promise<void>): Promise<void> => {
  const overrides: Record<string, string> = {
    LC_ALL: "C",
    PYTHONCOERCECLOCALE: "0",
    PYTHONUTF8: "0",
  };
  const previous = Object.fromEntries(
    Object.keys(overrides).map((key) => [key, process.env[key]]),
  );
  Object.assign(process.env, overrides);
  return body().finally(() => {
    for (const [key, value] of Object.entries(previous))
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
  });
};

/** Any interpreter a probe script can run on, laya host or bare system one. */
const anyPython = async (): Promise<string> => {
  if (layaPython !== undefined) return layaPython;
  for (const interpreter of ["python3", "python"])
    if (await exitsCleanly(interpreter, ["-c", "pass"])) return interpreter;
  throw new Error("no python interpreter on PATH");
};

/** A client evaluating through the real laya engine at the given model. */
const client = (
  model: LayaModel,
  llmAnswerMode: "probabilities" | "discrete" = "probabilities",
): SystemOneAdapterClient =>
  new SystemOneAdapterClient({
    structuredOutputs: true,
    llmAnswerMode,
    normalizeProbabilities: true,
    model: new LayaProvider(model, { python: python() }),
  });

/** A client resolving `provider: "laya"` through the provider registry. */
const builtInClient = (): SystemOneAdapterClient =>
  new SystemOneAdapterClient({
    structuredOutputs: true,
    llmAnswerMode: "probabilities",
    provider: "laya",
    model: "router",
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

/** Assert a probability-like value lies in [0, 1]. */
const expectUnitInterval = (value: number | undefined): void => {
  expect(value).toBeGreaterThanOrEqual(0);
  expect(value).toBeLessThanOrEqual(1);
};

/** Assert every probability lies in [0, 1] and they sum to ~1. */
const expectDistribution = (
  probabilities: Record<string, number> | undefined,
): void => {
  const values = Object.values(probabilities ?? {});
  expect(values).not.toHaveLength(0);
  for (const value of values) expectUnitInterval(value);
  const sum = values.reduce((total, value) => total + value, 0);
  expect(Math.abs(sum - 1)).toBeLessThan(0.01);
};

describe("LayaProvider", () => {
  it("rejects unknown laya models", () => {
    expect(() => new LayaProvider("flash" as LayaModel)).toThrow(
      `laya model must be one of ${LAYA_MODELS.join(", ")}`,
    );
  });

  it("rejects requests without typed questions", async () => {
    const provider = new LayaProvider("router");
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

  it("reads its UTF-8 payload even on non-UTF-8 hosts", async () => {
    // The adapter pipes UTF-8 JSON to the child; python must decode it
    // as UTF-8 even when its locale says otherwise — on Windows that is
    // cp1252, where ” (U+201D) becomes a lone surrogate the engine
    // rejects with a TextEncodeInput TypeError.
    const text = "curly ” quotes — ünïcode";
    const probe = [
      "import json, sys",
      "payload = json.load(sys.stdin)",
      "report = {'encoding': sys.stdin.encoding, 'text': payload['text']}",
      "print(json.dumps(report))",
    ].join("\n");
    await withNonUtf8Host(async () => {
      const result = await defaultRunner(
        await anyPython(),
        probe,
        JSON.stringify({ text }),
      );
      expect(result.code).toBe(0);
      const report = JSON.parse(result.stdout);
      expect(report.encoding.toLowerCase()).toBe("utf-8");
      expect(report.text).toBe(text);
    });
  });

  it("routes requests through the LAYA_PYTHON interpreter", async () => {
    await withLayaPython("adapter-no-such-laya-python", async () => {
      const adapter = builtInClient();
      await expect(
        adapter.systemOne({
          state: { body: "x" },
          questions: { positive: noul("Good?") },
        }),
      ).rejects.toThrow(/adapter-no-such-laya-python/u);
      await adapter.close();
    });
  });
});

describe("LayaProvider against the real laya engine", () => {
  it("evaluates typed questions with calibrated probabilities", {
    timeout: MODEL_TIMEOUT,
  }, async () => {
    const adapter = client("router");
    const response = await adapter.systemOne({
      state: { user_message: "hi", assistant_1: "a", assistant_2: "b" },
      questions: QUESTIONS,
    });
    await adapter.close();

    // A local encoder has no token metering, but latency is measured.
    expect(response.model).toBe("laya/router");
    expect(response.usage.input_tokens_total).toBe(0);
    expect(response.usage.output_tokens_total).toBe(0);
    expect(response.usage.latency).toBeGreaterThan(0);

    const verdict = response.choices.verdict;
    expect(["A", "B", "tie"]).toContain(verdict?.choice);
    expect(Object.keys(verdict?.probabilities ?? {}).sort()).toEqual([
      "A",
      "B",
      "tie",
    ]);
    expectDistribution(verdict?.probabilities);
    // The discrete choice is the argmax of the returned distribution.
    const argmax = (["A", "B", "tie"] as const).reduce((best, label) =>
      (verdict?.probabilities?.[label] ?? 0) >
      (verdict?.probabilities?.[best] ?? 0)
        ? label
        : best,
    );
    expect(verdict?.choice).toBe(argmax);
    expectUnitInterval(verdict?.confidence);

    const isSafe = response.nouls.is_safe;
    expectUnitInterval(isSafe?.noul);

    const rating = response.scores.rating;
    expect(rating?.legend).toEqual({
      0: "Unhelpful",
      1: "Somewhat helpful",
      2: "Very helpful",
    });
    expect(Object.keys(rating?.probabilities ?? {}).sort()).toEqual([
      "0",
      "1",
      "2",
    ]);
    expectDistribution(rating?.probabilities);
    // The reported score is the expected value over that distribution.
    const expected = (["0", "1", "2"] as const).reduce(
      (total, level) =>
        total + Number(level) * (rating?.probabilities?.[level] ?? 0),
      0,
    );
    expect(Math.abs((rating?.score ?? -1) - expected)).toBeLessThan(0.01);
    expectUnitInterval(rating?.confidence);
  });

  it("maps discrete answers: labels, booleans, rounded scores", {
    timeout: MODEL_TIMEOUT,
  }, async () => {
    const adapter = client("english", "discrete");
    const response = await adapter.systemOne({
      state: { body: "A thoughtful and complete answer." },
      questions: QUESTIONS,
    });
    await adapter.close();

    const verdict = response.choices.verdict;
    expect(["A", "B", "tie"]).toContain(verdict?.choice);
    // A discrete answer carries all its probability on the chosen label.
    for (const label of ["A", "B", "tie"] as const)
      expect(verdict?.probabilities?.[label]).toBe(
        label === verdict?.choice ? 1 : 0,
      );

    expect([0, 1]).toContain(response.nouls.is_safe?.noul);

    const rating = response.scores.rating;
    expect(Number.isInteger(rating?.score)).toBe(true);
    expect(rating?.score).toBeGreaterThanOrEqual(0);
    expect(rating?.score).toBeLessThanOrEqual(2);
    expect(Object.keys(rating?.probabilities ?? {}).sort()).toEqual([
      "0",
      "1",
      "2",
    ]);
    const level = String(rating?.score);
    expect(rating?.probabilities?.[0]).toBe(level === "0" ? 1 : 0);
    expect(rating?.probabilities?.[1]).toBe(level === "1" ? 1 : 0);
    expect(rating?.probabilities?.[2]).toBe(level === "2" ? 1 : 0);
    expect(rating?.legend).toEqual({
      0: "Unhelpful",
      1: "Somewhat helpful",
      2: "Very helpful",
    });
  });

  it("loads the multilingual checkpoint with its subfolder", {
    timeout: MODEL_TIMEOUT,
  }, async () => {
    const adapter = client("multilingual");
    const response = await adapter.systemOne({
      state: { body: "Mein Konto wurde zweimal belastet" },
      questions: { urgent: noul("Urgent?") },
    });
    await adapter.close();

    expect(response.model).toBe("laya/multilingual");
    expectUnitInterval(response.nouls.urgent?.noul);
  });

  it("runs the built-in provider through the LAYA_PYTHON interpreter", {
    timeout: MODEL_TIMEOUT,
  }, async () => {
    await withLayaPython(python(), async () => {
      const adapter = builtInClient();
      const response = await adapter.systemOne({
        state: { body: "A thoughtful and complete answer." },
        questions: { positive: noul("The answer is helpful.") },
      });
      await adapter.close();

      expect(response.model).toBe("laya/router");
      expectUnitInterval(response.nouls.positive?.noul);
    });
  });
});

describe("a missing laya install", () => {
  // A freshly created venv has no packages, so its interpreter always
  // reproduces the real ModuleNotFoundError a machine without laya sees.
  let bareVenvDir: string | undefined;
  let barePython: string | undefined;

  const discard = async (): Promise<void> => {
    if (bareVenvDir === undefined) return;
    await rm(bareVenvDir, { recursive: true, force: true });
    bareVenvDir = undefined;
    barePython = undefined;
  };

  beforeAll(async () => {
    const dir = await mkdtemp(join(tmpdir(), "adapter-laya-bare-"));
    bareVenvDir = dir;
    // `python3` is the POSIX name; `python` is the usual one on Windows.
    for (const interpreter of ["python3", "python"]) {
      if (
        !(await exitsCleanly(interpreter, ["-m", "venv", "--without-pip", dir]))
      )
        continue;
      barePython =
        process.platform === "win32"
          ? join(bareVenvDir, "Scripts", "python.exe")
          : join(bareVenvDir, "bin", "python");
      return;
    }
    await discard();
  }, 60_000);

  afterAll(async () => {
    await discard();
  });

  it("surfaces the python failure through the built-in provider", async () => {
    // Never skipped: a machine that cannot even build the bare venv is
    // broken in a way the suite must report, not paper over.
    if (barePython === undefined)
      throw new Error("could not create a bare venv without the laya package");
    await withLayaPython(barePython, async () => {
      const adapter = builtInClient();
      await expect(
        adapter.systemOne({ state: { body: "x" }, questions: QUESTIONS }),
      ).rejects.toThrow(/laya python process exited.*No module named 'laya'/u);
      await adapter.close();
    });
  });
});
