import { type ArkError, scope, type Traversal, type Type, type } from "arktype";
import type { AnswerMode } from "./utils/probabilityNormalization.js";

/** Score and choice questions need at least two outcomes to choose between. */
const MIN_CRITERIA = 2;

/**
 * Runtime validation of the JSON values the SDK accepts for instructions,
 * criteria, and state.
 *
 * `JsonObject` recursion mirrors the SDK's `JsonValue` so the validated
 * inference stays structurally identical to the SDK types.
 */
const jsonTypes = scope({
  JsonValue: "string|number|boolean|null|JsonValue[]|JsonObject",
  JsonObject: {
    "[string]": "JsonValue",
  },
}).export();

/** Whether a value is a plain object: neither an array nor a function. */
const isPlainObject = (value: object): boolean =>
  !Array.isArray(value) && typeof value !== "function";

/**
 * A JSON object.
 *
 * The predicate rejects arrays and functions, which also satisfy arktype's
 * `object` keyword, matching the SDK's plain-object entry values.
 */
const PlainObject = jsonTypes.JsonObject.narrow(isPlainObject);

/** Runtime validation of one instruction or criterion value. */
const entryTypes = scope({
  PlainObject,
  JsonValue: jsonTypes.JsonValue,
  EntryType: "string|PlainObject|JsonValue[]|null",
}).export();

/**
 * Runtime validation of noul criteria: a plain object of optional entry
 * values for the true and false outcomes.
 *
 * The predicate rejects arrays, which satisfy arktype's optional-key checks.
 */
const NoulCriteria = type({
  "true?": entryTypes.EntryType.or("undefined"),
  "false?": entryTypes.EntryType.or("undefined"),
}).narrow(isPlainObject);

/**
 * Runtime validation of choice criteria: a plain object of entry values.
 *
 * The predicate rejects arrays, which satisfy arktype index signatures.
 */
const ChoiceCriteria = type({
  "[string]": entryTypes.EntryType,
}).narrow(isPlainObject);

/** Runtime validation of the SDK question schema. */
const questionTypes = scope({
  EntryType: entryTypes.EntryType,
  NoulCriteria,
  ChoiceCriteria,
  NoulQuestion: {
    type: "'noul'",
    "instructions?": "EntryType|undefined",
    "criteria?": "NoulCriteria|null|undefined",
  },
  ScoreQuestion: {
    type: "'score'",
    "instructions?": "EntryType|undefined",
    criteria: "EntryType[]",
  },
  ChoiceQuestion: {
    type: "'choice'",
    "instructions?": "EntryType|undefined",
    criteria: "ChoiceCriteria",
  },
  Question: "NoulQuestion|ScoreQuestion|ChoiceQuestion",
  QuestionCollection: {
    "[string]": "Question",
  },
}).export();

/** A JSON value used for instructions, criteria, and state. */
type EntryType = typeof questionTypes.EntryType.infer;

/** A validated question in provider-neutral form. */
type Question = typeof questionTypes.Question.infer;

/** The runtime type of one arktype-validated model answer. */
type ValidatedAnswer = boolean | number | string | Record<string, number>;

/** A JSON schema object as sent to providers. */
type JsonSchema = Record<string, unknown>;

/** Error raised when the question collection does not match the SDK schema. */
class InvalidQuestionsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidQuestionsError";
  }
}

/** Error raised when model output does not match the generated schema. */
class OutputValidationError extends Error {
  constructor(issues: readonly string[]) {
    super(`Output validation failed:\n${issues.join("\n")}`);
    this.name = "OutputValidationError";
  }
}

/** Normalize one validated question to plain data with defaulted fields. */
const normalizeQuestion = (question: Question): Question => {
  const instructions = question.instructions ?? null;
  switch (question.type) {
    case "noul":
      return {
        type: "noul",
        instructions,
        criteria: question.criteria ?? null,
      };
    case "score":
      return { type: "score", instructions, criteria: question.criteria };
    default:
      return { type: "choice", instructions, criteria: question.criteria };
  }
};

/** Validate a question collection and normalize it to plain data. */
const validateQuestions = (questions: unknown): Record<string, Question> => {
  const validated = questionTypes.QuestionCollection(questions);
  if (validated instanceof type.errors)
    throw new InvalidQuestionsError(
      `Questions must match the SDK question schema:\n${validated.summary}`,
    );
  const normalized: Record<string, Question> = {};
  for (const [questionId, question] of Object.entries(validated))
    normalized[questionId] = structuredClone(normalizeQuestion(question));
  if (Object.keys(normalized).length === 0)
    throw new InvalidQuestionsError("At least one question is required.");
  for (const question of Object.values(normalized))
    if (
      question.type !== "noul" &&
      Object.keys(question.criteria).length < MIN_CRITERIA
    )
      throw new InvalidQuestionsError(
        "Score and choice questions require at least two criteria.",
      );
  return normalized;
};

/** Serialize an instruction or criterion value for inclusion in a prompt. */
const serializeInstructionValue = (value: unknown): string => {
  if (value === null || value === undefined)
    return "No additional instructions.";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
};

/** The ordered answer labels of a score or choice question. */
const answerLabels = (question: Question): string[] =>
  question.type === "score"
    ? question.criteria.map((_, score) => String(score))
    : question.type === "choice"
      ? Object.keys(question.criteria)
      : [];

/** The criteria of a score or choice question, keyed by answer label. */
const criteriaByLabel = (question: Question): [string, EntryType][] =>
  question.type === "score"
    ? question.criteria.map((criterion, score): [string, EntryType] => [
        String(score),
        criterion,
      ])
    : question.type === "choice"
      ? Object.entries(question.criteria)
      : [];

/** The per-question description used on probability-map definitions. */
const questionDescription = (question: Question, mode: AnswerMode): string => {
  const description = serializeInstructionValue(question.instructions);
  if (question.type === "noul" && mode === "probabilities")
    return `Probability that the answer is yes or the assertion is true. 0 means no or false, 0.5 means uncertain, and 1 means yes or true.\nQuestion: ${description}`;
  if (question.type === "score" && mode === "probabilities")
    return `Each property maps a rubric level to the probability that the document matches it.\nQuestion: ${description}`;
  if (question.type === "choice" && mode === "probabilities")
    return `Each property maps an option to the probability that it is the best answer.\nQuestion: ${description}`;
  return description;
};

/** The description placed on an answer field, with its allowed values. */
const fieldDescription = (question: Question, mode: AnswerMode): string => {
  const description = questionDescription(question, mode);

  if (question.type === "score") {
    const levels = question.criteria
      .map(
        (criterion, score) =>
          `${score} = ${serializeInstructionValue(criterion)}`,
      )
      .join("\n");
    if (mode === "discrete")
      return `${description}\nScore levels, answer with the integer:\n${levels}`;
    return `${description}\nRequired probability keys:\n${levels}`;
  }

  if (question.type === "choice") {
    const choices = Object.entries(question.criteria)
      .map(
        ([answer, criterion]) =>
          `${answer} = ${serializeInstructionValue(criterion)}`,
      )
      .join("\n");
    if (mode === "discrete")
      return `${description}\nChoice labels, answer with one label:\n${choices}`;
    return `${description}\nRequired probability keys:\n${choices}`;
  }

  if (question.criteria === null || question.criteria === undefined)
    return description;

  const trueCriteria = serializeInstructionValue(
    question.criteria.true ?? null,
  );
  const falseCriteria = serializeInstructionValue(
    question.criteria.false ?? null,
  );
  return `${description}\nTrue criteria: ${trueCriteria}\nFalse criteria: ${falseCriteria}`;
};

/** The value shape one question's answer must take. */
type AnswerSpec =
  | { kind: "boolean" }
  | { kind: "probability" }
  | { kind: "integer"; max: number }
  | { kind: "enum"; values: string[] }
  | { kind: "probabilityMap"; labels: string[] };

/** Everything needed to generate a schema and validate output for one request. */
interface OutputSpec {
  /** Question identifiers in collection order. */
  questionIds: readonly string[];
  /** Answer shape per question identifier. */
  answers: Record<string, AnswerSpec>;
  /** Field description per question identifier, when the field carries one. */
  descriptions: Record<string, string>;
  /** Probability-map definitions, keyed by their `$defs` name. */
  probabilityMaps: Record<
    string,
    { description: string; properties: Record<string, string> }
  >;
}

/** The answer spec pinning one question to its allowed values. */
const answerSpecForQuestion = (
  question: Question,
  mode: AnswerMode,
): AnswerSpec => {
  if (question.type === "noul")
    return mode === "discrete" ? { kind: "boolean" } : { kind: "probability" };
  const labels = answerLabels(question);
  if (question.type === "score")
    return mode === "discrete"
      ? { kind: "integer", max: question.criteria.length }
      : { kind: "probabilityMap", labels };
  return mode === "discrete"
    ? { kind: "enum", values: labels }
    : { kind: "probabilityMap", labels };
};

/** Build the per-request output spec for the given questions and answer mode. */
const buildOutputSpec = (
  questions: Readonly<Record<string, Question>>,
  llmAnswerMode: AnswerMode,
): OutputSpec => {
  const spec: OutputSpec = {
    questionIds: Object.keys(questions),
    answers: {},
    descriptions: {},
    probabilityMaps: {},
  };
  for (const [index, [questionId, question]] of Object.entries(
    questions,
  ).entries()) {
    const answerSpec = answerSpecForQuestion(question, llmAnswerMode);
    spec.answers[questionId] = answerSpec;
    if (answerSpec.kind === "probabilityMap") {
      const properties: Record<string, string> = {};
      for (const [label, criterion] of criteriaByLabel(question))
        properties[label] = serializeInstructionValue(criterion);
      spec.probabilityMaps[`ProbabilityMap${index}`] = {
        description: questionDescription(question, llmAnswerMode),
        properties,
      };
    } else {
      spec.descriptions[questionId] = fieldDescription(question, llmAnswerMode);
    }
  }
  return spec;
};

/** The JSON schema of one answer field. */
const answerSchema = (
  answerSpec: AnswerSpec,
  description: string | undefined,
  index: number,
): JsonSchema => {
  switch (answerSpec.kind) {
    case "probabilityMap":
      return { $ref: `#/$defs/ProbabilityMap${index}` };
    case "boolean":
      return { description, type: "boolean" };
    case "probability":
      return { description, type: "number" };
    case "integer":
      return { description, type: "integer" };
    case "enum":
      return { description, type: "string", enum: answerSpec.values };
  }
};

/** Create the self-contained JSON schema for one request's answers. */
const buildSchema = (spec: OutputSpec): JsonSchema => {
  const defs: Record<string, JsonSchema> = {};
  for (const [name, map] of Object.entries(spec.probabilityMaps)) {
    const properties: Record<string, JsonSchema> = {};
    for (const [label, description] of Object.entries(map.properties))
      properties[label] = { description, type: "number" };
    defs[name] = {
      description: map.description,
      type: "object",
      properties,
      required: Object.keys(properties),
      additionalProperties: false,
    };
  }

  const answerProperties: Record<string, JsonSchema> = {};
  for (const [index, questionId] of spec.questionIds.entries())
    answerProperties[questionId] = answerSchema(
      spec.answers[questionId],
      spec.descriptions[questionId],
      index,
    );

  defs.TypeSafeAnswers = {
    description:
      "Exactly one answer per property below. Use these property names verbatim and do not add, rename, or nest them under any other key.",
    type: "object",
    properties: answerProperties,
    required: [...spec.questionIds],
    additionalProperties: false,
  };

  return {
    type: "object",
    properties: { answers: { $ref: "#/$defs/TypeSafeAnswers" } },
    required: ["answers"],
    additionalProperties: false,
    $defs: defs,
  };
};

/** arktype type of one probability: a finite number in [0, 1]. */
const Probability = type("number").atLeast(0).atMost(1);

/** arktype type of a discrete boolean answer. */
const BooleanAnswer = type("boolean");

/** Require exactly the given keys on a probability map. */
const exactKeySet =
  (labels: readonly string[]) =>
  (value: Record<string, number>, ctx: Traversal): boolean => {
    const keys = Object.keys(value);
    if (
      labels.every((label) => keys.includes(label)) &&
      keys.every((key) => labels.includes(key))
    )
      return true;
    return ctx.mustBe(
      `an object with exactly the properties ${JSON.stringify(labels)}`,
    );
  };

/** Build the arktype type validating one answer per its answer spec. */
const answerType = (answerSpec: AnswerSpec): Type<ValidatedAnswer> => {
  switch (answerSpec.kind) {
    case "boolean":
      return BooleanAnswer;
    case "probability":
      return Probability;
    case "integer":
      return type("number").divisibleBy(1).atLeast(0).lessThan(answerSpec.max);
    case "enum":
      return type.enumerated(...answerSpec.values);
    case "probabilityMap":
      return type({ "[string]": Probability }).narrow(
        exactKeySet(answerSpec.labels),
      );
  }
};

/** arktype type of the answers record inside a model payload. */
const AnswersRecord = type({ "[string]": "unknown" });

/** arktype type of a parsed model payload: exactly an `answers` object. */
const ModelPayload = type({ answers: AnswersRecord }).onUndeclaredKey("reject");

/** Render one arktype error as a validation issue string. */
const issueFor = (prefix: string, error: ArkError): string => {
  const path = error.path.map(String).join(".");
  return path === ""
    ? `${prefix}${error.problem}`
    : `${prefix}${path}: ${error.problem}`;
};

/** Validate a parsed model response against the output spec. */
const validateOutput = (
  spec: OutputSpec,
  payload: unknown,
): Record<string, ValidatedAnswer> => {
  const issues: string[] = [];
  const container = ModelPayload(payload);
  if (container instanceof type.errors) {
    for (const error of container) issues.push(issueFor("", error));
    throw new OutputValidationError(issues);
  }
  const answers = container.answers;
  for (const questionId of spec.questionIds)
    if (!(questionId in answers))
      issues.push(
        `answers: missing required property ${JSON.stringify(questionId)}`,
      );
  for (const key of Object.keys(answers))
    if (!spec.questionIds.includes(key))
      issues.push(
        `answers: extra property ${JSON.stringify(key)} is not allowed`,
      );
  const validated: Record<string, ValidatedAnswer> = {};
  for (const questionId of spec.questionIds) {
    if (!(questionId in answers)) continue;
    const answer = answerType(spec.answers[questionId])(answers[questionId]);
    if (answer instanceof type.errors)
      for (const error of answer)
        issues.push(issueFor(`answers.${questionId}.`, error));
    else validated[questionId] = answer;
  }
  if (issues.length > 0) throw new OutputValidationError(issues);
  return validated;
};

export {
  type AnswerMode,
  type AnswerSpec,
  answerLabels,
  buildOutputSpec,
  buildSchema,
  type EntryType,
  InvalidQuestionsError,
  type JsonSchema,
  type OutputSpec,
  OutputValidationError,
  type Question,
  type ValidatedAnswer,
  validateOutput,
  validateQuestions,
};
