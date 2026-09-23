import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { choice, noul, score } from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { Agent, Router, VERSION } from "../laya-ts/index.js";
import type { Batch, SessionProvider } from "../laya-ts/providers.js";
import type { ModelName } from "../laya-ts/router.js";
import {
  LAYA_MODELS,
  type LayaModel,
  LayaProvider,
} from "../providers/laya.js";

/**
 * The engine under test is the vendored laya-ts port (`src/laya-ts`) driven
 * by a deterministic fake ONNX session, so these tests exercise the full
 * provider path — question building, batching, answer shaping, routing —
 * without weights. The suite never reaches the network: msw fails any
 * unintended request, which also proves the fake path never tries to fetch
 * checkpoints.
 */

/** Log-probability logits; softmax recovers the given probabilities. */
const logitsFor = (probabilities: number[]): number[] =>
  probabilities.map((p) => Math.log(p));

/** Deterministic ONNX session: label 0 wins, mid score, 70% noul. */
const fakeSession = (): SessionProvider => ({
  runEncoder: async (batch: Batch) => ({
    lastHidden: batch.inputIds.map((row) =>
      row.map(() => [0.5, 0.5, 0.5, 0.5]),
    ),
  }),
  runHead: async (
    _hidden: unknown,
    batch: Batch,
  ): Promise<{
    logits: number[][];
    act: number[][];
  }> => ({
    logits: batch.qtype.map((qtype, r) => {
      // The item's true option count: markerPos is padded to the batch
      // maximum, so the marker mask carries the real length.
      const k = batch.markerMask[r].filter(Boolean).length;
      if (qtype === 0)
        return Array.from({ length: k }, (_, i) => (i === 0 ? 2 : 0));
      if (qtype === 1) {
        // Two even levels give an exact 0.5 expectation; three levels a
        // [0.25, 0.5, 0.25] distribution with an exact expectation of 1.
        const probabilities =
          k === 2 ? [0.5, 0.5] : [0.25, 0.5, 0.25].slice(0, k);
        return logitsFor(probabilities);
      }
      return logitsFor([0.3, 0.7]);
    }),
    act: batch.qtype.map(() => [2, 1]),
  }),
});

/** A client evaluating through the fake session at the given model. */
const client = (
  model: LayaModel,
  llmAnswerMode: "probabilities" | "discrete" = "probabilities",
): SystemOneAdapterClient =>
  new SystemOneAdapterClient({
    structuredOutputs: true,
    llmAnswerMode,
    normalizeProbabilities: true,
    model: new LayaProvider(model, { session: fakeSession() }),
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

/** Run the body with the given environment overrides, restoring after. */
const withEnv = (
  overrides: Record<string, string>,
  body: () => Promise<void>,
): Promise<void> => {
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

/** One provider request carrying a single noul question. */
const noulRequest = (provider: LayaProvider) =>
  provider.request([{ role: "user", content: "unused" }], {
    schema: {},
    structured: true,
    typed: {
      state: { body: "x" },
      questions: { positive: noul("Good?") },
      answerMode: "probabilities",
    },
  });

describe("LayaProvider", () => {
  it("rejects unknown laya models", () => {
    expect(() => new LayaProvider("flash" as LayaModel)).toThrow(
      `laya model must be one of ${LAYA_MODELS.join(", ")}`,
    );
  });

  it("rejects requests without typed questions", async () => {
    const provider = new LayaProvider("router", { session: fakeSession() });
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

  it("surfaces a missing ONNX export with recovery guidance", async () => {
    const dir = await mkdtemp(join(tmpdir(), "adapter-laya-empty-"));
    try {
      const provider = new LayaProvider("english", {
        models: { english: dir },
      });
      await expect(noulRequest(provider)).rejects.toThrow(
        /failed to load.*export_onnx/su,
      );
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("resolves LAYA_MODEL_DIR to one exported bundle per checkpoint", async () => {
    const base = await mkdtemp(join(tmpdir(), "adapter-laya-dir-"));
    try {
      await mkdir(join(base, "english"), { recursive: true });
      await withEnv({ LAYA_MODEL_DIR: base }, async () => {
        // The constructor reads LAYA_MODEL_DIR, so it lives inside the
        // override: without the variable, the checkpoint would resolve to
        // the default Hugging Face locations instead of the local tree.
        const provider = new LayaProvider("english");
        await expect(noulRequest(provider)).rejects.toThrow(
          join(base, "english"),
        );
      });
    } finally {
      await rm(base, { recursive: true, force: true });
    }
  });
});

describe("LayaProvider against the in-process engine", () => {
  it("evaluates typed questions through the routing engine", async () => {
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
    // The fake head's logits always peak at the first label.
    expect(verdict?.choice).toBe("A");
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
    expect(isSafe?.noul).toBe(0.7);

    const rating = response.scores.rating;
    expect(rating?.legend).toEqual({
      0: "Unhelpful",
      1: "Somewhat helpful",
      2: "Very helpful",
    });
    expect(rating?.probabilities).toEqual({ 0: 0.25, 1: 0.5, 2: 0.25 });
    // The reported score is the expected value over that distribution.
    expect(rating?.score).toBe(1);
    expectUnitInterval(rating?.confidence);
  });

  it("maps discrete answers: labels, booleans, and python-rounded levels", async () => {
    // The two-level score's exact 0.5 expectation discriminates python's
    // round-half-to-even (level 0) from javascript's half-up (level 1).
    const adapter = client("english", "discrete");
    const response = await adapter.systemOne({
      state: { body: "A thoughtful and complete answer." },
      questions: {
        verdict: QUESTIONS.verdict,
        is_safe: QUESTIONS.is_safe,
        rating: score("How helpful is the response?", ["Unhelpful", "Helpful"]),
      },
    });
    await adapter.close();

    const verdict = response.choices.verdict;
    expect(verdict?.choice).toBe("A");
    // A discrete answer carries all its probability on the chosen label.
    for (const label of ["A", "B", "tie"] as const)
      expect(verdict?.probabilities?.[label]).toBe(
        label === verdict?.choice ? 1 : 0,
      );

    expect([0, 1]).toContain(response.nouls.is_safe?.noul);

    const rating = response.scores.rating;
    expect(Number.isInteger(rating?.score)).toBe(true);
    expect(rating?.score).toBe(0);
    expect(rating?.legend).toEqual({ 0: "Unhelpful", 1: "Helpful" });
    expect(rating?.probabilities?.[0]).toBe(1);
    expect(rating?.probabilities?.[1]).toBe(0);
  });

  it("drops cached agents on close and still serves later requests", async () => {
    const provider = new LayaProvider("english", { session: fakeSession() });
    const first = await noulRequest(provider);
    provider.close();
    const second = await noulRequest(provider);
    expect(JSON.parse(second.text)).toEqual(JSON.parse(first.text));
  });
});

describe("routing", () => {
  it("sends english text to the english checkpoint and latin non-english to multilingual", async () => {
    // The wrapper delegates language detection to the vendored Router; this
    // pins the routing contract its loader relies on.
    const loaded: ModelName[] = [];
    const router = new Router({
      loader: async (name: ModelName) => {
        loaded.push(name);
        return new Agent({ provider: fakeSession() });
      },
    });
    await router.predict(
      {
        body: "Der Kunde möchte seine Bestellung stornieren und das Geld zurück",
      },
      { urgent: { type: "noul", instructions: "Urgent?" } },
    );
    await router.predict(
      { body: "please refund my order" },
      { urgent: { type: "noul", instructions: "Urgent?" } },
    );
    expect(loaded).toEqual(["multilingual", "english"]);
  });
});

describe("vendored laya-ts", () => {
  it("keeps the full public surface importable", async () => {
    // Importing the package entry executes every vendored module, guarding
    // the arrow-const conversion against initialization-order breakage.
    const layaTs = await import("../laya-ts/index.js");
    expect(layaTs.VERSION).toBe(VERSION);
    expect(Agent).toBeTypeOf("function");
    expect(Router).toBeTypeOf("function");
    expect(layaTs.buildSequence).toBeTypeOf("function");
    expect(layaTs.bpeEncode).toBeTypeOf("function");
  });
});
