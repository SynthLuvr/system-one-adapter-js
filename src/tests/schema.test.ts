import { expect, it } from "vitest";
import {
  type AnswerMode,
  buildOutputSpec,
  buildSchema,
  InvalidQuestionsError,
  OutputValidationError,
  validateOutput,
  validateQuestions,
} from "../schema.js";

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

type QuestionsInput = Parameters<typeof validateQuestions>[0];
type SdkQuestion = QuestionsInput["answer"];

const asRecord = (value: unknown): Record<string, unknown> =>
  value as Record<string, unknown>;

const invalidDictionaryQuestions: readonly unknown[] = [
  { type: "unknown" },
  { type: "noul", instructions: 42 },
  { type: "choice", criteria: ["yes", "no"] },
  { type: "score", criteria: { "0": "Bad.", "1": "Good." } },
];

it.each(invalidDictionaryQuestions)(
  "rejects invalid dictionary questions %#",
  (question) => {
    expect(() => validateQuestions({ answer: question } as never)).toThrow(
      InvalidQuestionsError,
    );
  },
);

it("revalidates SDK question fields", () => {
  // The Python suite mutates an SDK model, but TypeScript has no mutable SDK
  // models, so the invalid field is passed through a plain object instead.
  const question = { type: "noul", criteria: { true: 42 } } as const;
  expect(() => validateQuestions({ answer: question } as never)).toThrow(
    InvalidQuestionsError,
  );
});

it.each(["probabilities", "discrete"] as const)(
  "preserves question ids with arbitrary names in %s mode",
  (mode) => {
    const questions = Object.fromEntries(
      FIELD_NAMES.map(
        (key) =>
          [key, { type: "noul", instructions: `Evaluate ${key}.` }] as const,
      ),
    );
    const spec = buildOutputSpec(validateQuestions(questions), mode);
    const schema = buildSchema(spec);
    const answers = asRecord(asRecord(schema.$defs).TypeSafeAnswers);

    expect(asRecord(asRecord(schema.properties).answers)).toEqual({
      $ref: "#/$defs/TypeSafeAnswers",
    });
    expect(answers.description).toContain("Use these property names verbatim");
    const properties = asRecord(answers.properties);
    expect(Object.keys(properties)).toEqual(FIELD_NAMES);
    expect(answers.required).toEqual(FIELD_NAMES);
    for (const key of FIELD_NAMES) {
      const answer = asRecord(properties[key]);
      expect(answer.description).toContain(`Evaluate ${key}.`);
      for (const keyword of SCHEMA_KEYWORDS)
        expect(keyword in answer).toBe(false);
    }

    const payload = {
      answers: Object.fromEntries(
        FIELD_NAMES.map((key) => [key, mode === "discrete" ? true : 0.8]),
      ),
    };
    expect(validateOutput(spec, payload)).toEqual(payload.answers);
  },
);

it("preserves probability labels with arbitrary names", () => {
  const criteria = Object.fromEntries(
    FIELD_NAMES.map((key) => [key, `The ${key} option.`] as const),
  );
  const spec = buildOutputSpec(
    validateQuestions({ level: { type: "choice", criteria } }),
    "probabilities",
  );
  const schema = buildSchema(spec);
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

  const uniform = Object.fromEntries(
    FIELD_NAMES.map((key) => [key, 1 / FIELD_NAMES.length] as const),
  );
  expect(validateOutput(spec, { answers: { level: uniform } })).toEqual({
    level: uniform,
  });

  const oversize = Object.fromEntries(
    FIELD_NAMES.map((key) => [key, 2] as const),
  );
  expect(() => validateOutput(spec, { answers: { level: oversize } })).toThrow(
    OutputValidationError,
  );
});

type InvalidAnswerCase = readonly [
  label: string,
  question: SdkQuestion,
  mode: AnswerMode,
  answer: unknown,
];

const invalidAnswers: readonly InvalidAnswerCase[] = [
  ["a string for a boolean answer", { type: "noul" }, "discrete", "true"],
  ["a number for a boolean answer", { type: "noul" }, "discrete", 1],
  ["a string for a probability", { type: "noul" }, "probabilities", "0.5"],
  ["a boolean for a probability", { type: "noul" }, "probabilities", true],
  ["a negative probability", { type: "noul" }, "probabilities", -0.1],
  ["a probability above one", { type: "noul" }, "probabilities", 1.1],
  ["NaN for a probability", { type: "noul" }, "probabilities", Number.NaN],
  [
    "a boolean for an integer answer",
    { type: "score", criteria: ["Bad.", "Good."] },
    "discrete",
    true,
  ],
  [
    "an out-of-range integer answer",
    { type: "score", criteria: ["Bad.", "Good."] },
    "discrete",
    2,
  ],
  [
    "an unknown choice label",
    { type: "choice", criteria: { yes: null, no: null } },
    "discrete",
    "maybe",
  ],
  [
    "a probability map missing a required key",
    { type: "choice", criteria: { yes: null, no: null } },
    "probabilities",
    { yes: 0.5 },
  ],
  [
    "a probability map with an extra key",
    { type: "choice", criteria: { yes: null, no: null } },
    "probabilities",
    { yes: 0.5, no: 0.5, maybe: 0 },
  ],
];

it.each(invalidAnswers)(
  "preserves types, bounds, and allowed values: rejects %s",
  (_label, question, mode, answer) => {
    const spec = buildOutputSpec(validateQuestions({ answer: question }), mode);
    expect(() => validateOutput(spec, { answers: { answer } })).toThrow(
      OutputValidationError,
    );
  },
);

// Python also rejects the JSON float 1.0 for an integer answer, but JavaScript
// numbers cannot distinguish 1.0 from 1, so that case cannot throw here.
it("accepts an integral float for an integer answer", () => {
  const spec = buildOutputSpec(
    validateQuestions({
      answer: { type: "score", criteria: ["Bad.", "Good."] },
    }),
    "discrete",
  );
  expect(validateOutput(spec, { answers: { answer: 1.0 } })).toEqual({
    answer: 1,
  });
});

const buildNoulProbabilitiesSpec = () =>
  buildOutputSpec(
    validateQuestions({ answer: { type: "noul" } }),
    "probabilities",
  );

it.each([
  { answers: { answer: 0.5 }, extra: 1 },
  { answers: { answer: 0.5, extra: 1 } },
  { answers: { answer_0: 0.5 } },
])("rejects extra fields and internal field names: %#", (payload) => {
  expect(() => validateOutput(buildNoulProbabilitiesSpec(), payload)).toThrow(
    OutputValidationError,
  );
});

it("rejects payloads without a valid answers object", () => {
  const spec = buildNoulProbabilitiesSpec();
  expect(() => validateOutput(spec, "hello")).toThrow(OutputValidationError);
  expect(() => validateOutput(spec, { answers: "nope" })).toThrow(
    OutputValidationError,
  );
  expect(() => validateOutput(spec, {})).toThrow(OutputValidationError);
});
