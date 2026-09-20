import { describe, expect, it } from "vitest";
import { SystemOneAdapterClient } from "../client.js";
import { OpenAIProvider } from "../providers/openai.js";
import { OutputValidationError } from "../schema.js";
import {
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  server,
} from "./msw.js";

const SCHEMA_KEYWORDS = [
  "title",
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
] as const;

const FIELD_NAMES = [
  ...SCHEMA_KEYWORDS,
  "model_dump",
  "model_config",
  "_private",
  "",
  "with spaces",
  "answer_0",
  "probability_0",
] as const;

const ANSWER = (answers: Record<string, unknown>): string =>
  JSON.stringify({ answers });

/** The generated JSON schema of a structured request. */
const requestSchema = (
  request: Record<string, unknown>,
): Record<string, unknown> =>
  (request.text as { format: { schema: Record<string, unknown> } }).format
    .schema;

const asRecord = (value: unknown): Record<string, unknown> =>
  value as Record<string, unknown>;

/** A one-question client call that surfaces the answer payload or its error. */
const evaluate = async (
  options: Record<string, unknown>,
  questions: Record<string, unknown>,
  payload: string,
): Promise<
  | {
      answers: Record<string, unknown>;
      debug: Record<string, unknown>;
      requests: Record<string, unknown>[];
    }
  | { error: Error & { cause?: Error } }
> => {
  const endpoint = openAIResponsesEndpoint(() =>
    openAIResponsesPayload(payload),
  );
  server.use(endpoint.handler);
  const client = new SystemOneAdapterClient({
    structuredOutputs: true,
    model: new OpenAIProvider("test-model", { apiKey: "test-key" }),
    ...options,
  } as never);
  try {
    const response = await client.systemOne({
      state: "A delightful novel.",
      questions: questions as never,
    });
    return {
      answers: response.answers as unknown as Record<string, unknown>,
      debug: response.debug as unknown as Record<string, unknown>,
      requests: endpoint.requests,
    };
  } catch (caught) {
    return { error: caught as Error & { cause?: Error } };
  }
};

describe("questions, schemas, and answers over the wire", () => {
  it.each(["probabilities", "discrete"] as const)(
    "preserves question ids with arbitrary names in %s mode",
    async (mode) => {
      const questions = Object.fromEntries(
        FIELD_NAMES.map(
          (key) =>
            [key, { type: "noul", instructions: `Evaluate ${key}.` }] as const,
        ),
      );
      const outcome = await evaluate(
        { llmAnswerMode: mode },
        questions,
        ANSWER(
          Object.fromEntries(
            FIELD_NAMES.map((key) => [key, mode === "discrete" ? true : 0.8]),
          ),
        ),
      );
      if (!("answers" in outcome)) throw outcome.error;

      const schema = requestSchema(outcome.requests[0]);
      const answers = asRecord(asRecord(schema.$defs).TypeSafeAnswers);

      expect(asRecord(asRecord(schema.properties).answers)).toEqual({
        $ref: "#/$defs/TypeSafeAnswers",
      });
      expect(answers.description).toContain(
        "Use these property names verbatim",
      );
      const properties = asRecord(answers.properties);
      expect(Object.keys(properties)).toEqual(FIELD_NAMES);
      expect(answers.required).toEqual(FIELD_NAMES);
      for (const key of FIELD_NAMES) {
        const answer = asRecord(properties[key]);
        expect(answer.description).toContain(`Evaluate ${key}.`);
        for (const keyword of SCHEMA_KEYWORDS)
          expect(keyword in answer).toBe(false);
        expect(answer.type).toBe(mode === "discrete" ? "boolean" : "number");
      }
      expect(Object.keys(outcome.answers)).toEqual(FIELD_NAMES);
    },
  );

  it("preserves probability labels with arbitrary names", async () => {
    const criteria = Object.fromEntries(
      FIELD_NAMES.map((key) => [key, `The ${key} option.`] as const),
    );
    const questions = { level: { type: "choice", criteria } };
    const uniform = Object.fromEntries(
      FIELD_NAMES.map((key) => [key, 1 / FIELD_NAMES.length] as const),
    );

    const outcome = await evaluate(
      { llmAnswerMode: "probabilities" },
      questions,
      ANSWER({ level: uniform }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    expect(outcome.answers.level).toMatchObject({ probabilities: uniform });

    const schema = requestSchema(outcome.requests[0]);
    const probabilities = asRecord(asRecord(schema.$defs).ProbabilityMap0);
    const properties = asRecord(probabilities.properties);
    expect(Object.keys(properties)).toEqual(FIELD_NAMES);
    expect(probabilities.required).toEqual(FIELD_NAMES);
    expect("title" in probabilities).toBe(false);
    for (const key of FIELD_NAMES)
      expect(properties[key]).toEqual({
        description: `The ${key} option.`,
        type: "number",
      });

    const oversize = Object.fromEntries(
      FIELD_NAMES.map((key) => [key, 2] as const),
    );
    const oversizeOutcome = await evaluate(
      { llmAnswerMode: "probabilities" },
      questions,
      ANSWER({ level: oversize }),
    );
    if (!("error" in oversizeOutcome)) throw new Error("expected a rejection");
    expect(oversizeOutcome.error.cause).toBeInstanceOf(OutputValidationError);
  });

  it("describes noul true/false criteria in the answer schema", async () => {
    const outcome = await evaluate(
      { llmAnswerMode: "probabilities" },
      {
        positive: {
          type: "noul",
          criteria: { true: "Positive.", false: "Negative." },
        },
      },
      ANSWER({ positive: 0.5 }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    const answer = asRecord(
      asRecord(
        asRecord(requestSchema(outcome.requests[0]).$defs).TypeSafeAnswers,
      ).properties,
    ).positive;
    expect(String(asRecord(answer).description)).toContain(
      "True criteria: Positive.",
    );
    expect(String(asRecord(answer).description)).toContain(
      "False criteria: Negative.",
    );
  });

  it.each([
    ["a string for a boolean answer", "discrete", { type: "noul" }, '"true"'],
    ["a number for a boolean answer", "discrete", { type: "noul" }, "1"],
    ["a string for a probability", "probabilities", { type: "noul" }, '"0.5"'],
    ["a boolean for a probability", "probabilities", { type: "noul" }, "true"],
    ["a negative probability", "probabilities", { type: "noul" }, "-0.1"],
    ["a probability above one", "probabilities", { type: "noul" }, "1.1"],
    ["an infinite probability", "probabilities", { type: "noul" }, "1e999"],
    [
      "a boolean for an integer answer",
      "discrete",
      { type: "score", criteria: ["Bad.", "Good."] },
      "true",
    ],
    [
      "a fractional integer answer",
      "discrete",
      { type: "score", criteria: ["Bad.", "Good."] },
      "1.5",
    ],
    [
      "an out-of-range integer answer",
      "discrete",
      { type: "score", criteria: ["Bad.", "Good."] },
      "2",
    ],
    [
      "an unknown choice label",
      "discrete",
      { type: "choice", criteria: { yes: null, no: null } },
      '"maybe"',
    ],
    [
      "a non-object probability map",
      "probabilities",
      { type: "choice", criteria: { yes: null, no: null } },
      "5",
    ],
    [
      "a probability map missing a required key",
      "probabilities",
      { type: "choice", criteria: { yes: null, no: null } },
      '{"yes":0.5}',
    ],
    [
      "a probability map with an extra key",
      "probabilities",
      { type: "choice", criteria: { yes: null, no: null } },
      '{"yes":0.5,"no":0.5,"maybe":0}',
    ],
  ] as [string, string, Record<string, unknown>, string][])(
    "model answers are schema-validated: rejects %s",
    async (_label, mode, question, rawAnswer) => {
      const outcome = await evaluate(
        { llmAnswerMode: mode },
        { answer: question },
        `{"answers":{"answer":${rawAnswer}}}`,
      );
      if (!("error" in outcome)) throw new Error("expected a rejection");
      expect(outcome.error.cause).toBeInstanceOf(OutputValidationError);
      expect(
        outcome.error.cause?.message.startsWith("Output validation failed:"),
      ).toBe(true);
    },
  );

  it.each([
    ["a bare string payload", '"a model aside"'],
    ["an extra top-level property", '{"answers":{},"aside":"chatter"}'],
    ["a missing answers property", '{"aside":"chatter"}'],
    ["a non-object answers property", '{"answers":5}'],
  ] as [string, string][])(
    "model payloads are schema-validated: rejects %s",
    async (_label, payload) => {
      const outcome = await evaluate(
        { llmAnswerMode: "probabilities" },
        { answer: { type: "noul" } },
        payload,
      );
      if (!("error" in outcome)) throw new Error("expected a rejection");
      expect(outcome.error.cause).toBeInstanceOf(OutputValidationError);
      expect(outcome.error.cause?.message.length).toBeGreaterThan(
        "Output validation failed:".length,
      );
    },
  );

  it("accepts an integral float for an integer answer", async () => {
    const outcome = await evaluate(
      { llmAnswerMode: "discrete" },
      { answer: { type: "score", criteria: ["Bad.", "Good."] } },
      ANSWER({ answer: 1.0 }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    expect((outcome.answers.answer as { score: number }).score).toBe(1);
  });

  it.each([
    ["true", true, 1],
    ["false", false, 0],
  ])(
    "discrete noul answers map %s to probability %d",
    async (_label, raw, expected) => {
      const outcome = await evaluate(
        { llmAnswerMode: "discrete" },
        { answer: { type: "noul" } },
        ANSWER({ answer: raw }),
      );
      if (!("answers" in outcome)) throw outcome.error;
      expect(outcome.answers.answer).toMatchObject({ noul: expected });
    },
  );

  it("discrete score and choice answers carry peak probability and confidence", async () => {
    const outcome = await evaluate(
      { llmAnswerMode: "discrete" },
      {
        stars: { type: "score", criteria: ["Bad.", "Fair.", "Good."] },
        genre: {
          type: "choice",
          criteria: { fiction: null, nonfiction: null },
        },
      },
      ANSWER({ stars: 2, genre: "fiction" }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    expect(outcome.answers.stars).toMatchObject({
      score: 2,
      confidence: 1,
      probabilities: { 0: 0, 1: 0, 2: 1 },
    });
    expect(outcome.answers.genre).toMatchObject({
      choice: "fiction",
      confidence: 1,
      probabilities: { fiction: 1, nonfiction: 0 },
    });
  });

  it("rescales off-distribution answers when normalization is enabled", async () => {
    const outcome = await evaluate(
      { llmAnswerMode: "probabilities", normalizeProbabilities: true },
      {
        genre: {
          type: "choice",
          criteria: { fiction: null, nonfiction: null },
        },
      },
      ANSWER({ genre: { fiction: 0.6, nonfiction: 0.6 } }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    expect(outcome.answers.genre).toMatchObject({
      probabilities: { fiction: 0.5, nonfiction: 0.5 },
    });
    const debug = asRecord(outcome.debug);
    expect(debug.invalid_probs).toBe(1);
    expect(debug.max_error).toBeCloseTo(0.2, 5);
    expect(asRecord(debug.probability_errors).genre).toBeCloseTo(0.2, 5);
    expect(asRecord(asRecord(debug.original_probabilities).genre)).toEqual({
      fiction: 0.6,
      nonfiction: 0.6,
    });
  });

  it("leaves off-distribution answers untouched when normalization is disabled", async () => {
    const outcome = await evaluate(
      { llmAnswerMode: "probabilities", normalizeProbabilities: false },
      {
        genre: {
          type: "choice",
          criteria: { fiction: null, nonfiction: null },
        },
      },
      ANSWER({ genre: { fiction: 0.6, nonfiction: 0.6 } }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    expect(outcome.answers.genre).toMatchObject({
      probabilities: { fiction: 0.6, nonfiction: 0.6 },
    });
    const debug = asRecord(outcome.debug);
    expect(debug.invalid_probs).toBe(1);
    expect("original_probabilities" in debug).toBe(false);
  });

  it("scores a zero-sum distribution as uniform", async () => {
    const outcome = await evaluate(
      { llmAnswerMode: "probabilities" },
      { stars: { type: "score", criteria: ["Bad.", "Good."] } },
      ANSWER({ stars: { 0: 0, 1: 0 } }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    const stars = asRecord(outcome.answers.stars);
    expect(stars.score).toBeCloseTo(0.5, 5);
    expect(stars.confidence).toBeCloseTo(0, 5);
    expect(stars.probabilities).toEqual({ 0: 0, 1: 0 });
  });

  it.each([
    [
      "a peaked choice distribution",
      { fiction: 0.82, nonfiction: 0.18 },
      "choices",
      "choice",
      "fiction",
      0.64,
    ],
    [
      "a uniform choice distribution",
      { fiction: 0.5, nonfiction: 0.5 },
      "choices",
      "choice",
      "fiction",
      0,
    ],
  ] as [string, Record<string, number>, string, string, string, number][])(
    "reports confidence for %s",
    async (_label, probabilities, _view, _field, expectedChoice, expected) => {
      const outcome = await evaluate(
        { llmAnswerMode: "probabilities" },
        {
          genre: {
            type: "choice",
            criteria: { fiction: null, nonfiction: null },
          },
        },
        ANSWER({ genre: probabilities }),
      );
      if (!("answers" in outcome)) throw outcome.error;
      expect(outcome.answers.genre).toMatchObject({ choice: expectedChoice });
      expect(
        (outcome.answers.genre as { confidence: number }).confidence,
      ).toBeCloseTo(expected, 5);
    },
  );

  it("reports score confidence from the distribution around the mode", async () => {
    const probabilities = { 0: 0.01, 1: 0.02, 2: 0.07, 3: 0.3, 4: 0.6 };
    const outcome = await evaluate(
      { llmAnswerMode: "probabilities" },
      {
        stars: {
          type: "score",
          criteria: ["Awful.", "Bad.", "Fair.", "Good.", "Great."],
        },
      },
      ANSWER({ stars: probabilities }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    const stars = asRecord(outcome.answers.stars);
    // 0.01*0 + 0.02*1 + 0.07*2 + 0.3*3 + 0.6*4 = 3.46
    expect(stars.score).toBeCloseTo(3.46, 5);
    expect(stars.confidence).toBeCloseTo(0.55, 5);
    expect(stars.probabilities).toEqual(probabilities);
    expect(stars.legend).toEqual({
      0: "Awful.",
      1: "Bad.",
      2: "Fair.",
      3: "Good.",
      4: "Great.",
    });
  });

  it("flags multiple invalid distributions in debug", async () => {
    const outcome = await evaluate(
      { llmAnswerMode: "probabilities", normalizeProbabilities: true },
      {
        first: { type: "noul" },
        second: { type: "choice", criteria: { a: null, b: null } },
        third: { type: "choice", criteria: { c: null, d: null } },
      },
      ANSWER({
        first: 0.5,
        second: { a: 0.7, b: 0.7 },
        third: { c: 0.5, d: 0.5 },
      }),
    );
    if (!("answers" in outcome)) throw outcome.error;
    const debug = asRecord(outcome.debug);
    expect(debug.invalid_probs).toBe(1);
    expect(debug.max_error).toBeCloseTo(0.4, 5);
    expect(Object.keys(asRecord(debug.probability_errors))).toEqual(["second"]);
    expect(Object.keys(asRecord(debug.original_probabilities))).toEqual([
      "second",
    ]);
  });
});
