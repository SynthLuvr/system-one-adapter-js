import type { EntryType, Questions } from "@typesafe-ai/sdk";
import type { AnswerMode } from "./utils/probabilityNormalization.js";

/** A validated question in provider-neutral form. */
type Question =
  | {
      type: "noul";
      instructions?: EntryType;
      criteria?: { true?: EntryType; false?: EntryType } | null;
    }
  | { type: "score"; instructions?: EntryType; criteria: EntryType[] }
  | {
      type: "choice";
      instructions?: EntryType;
      criteria: Record<string, EntryType>;
    };

/** A JSON schema object as sent to providers. */
type JsonSchema = Record<string, unknown>;

/** Score and choice questions need at least two outcomes to choose between. */
const MIN_CRITERIA = 2;

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

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isEntryType = (value: unknown): value is EntryType =>
  value === null ||
  typeof value === "string" ||
  isPlainObject(value) ||
  Array.isArray(value);

/** Validate one question value against the SDK question schema. */
const validateQuestion = (questionId: string, value: unknown): Question => {
  const where = `questions.${questionId}`;
  if (!isPlainObject(value))
    throw new InvalidQuestionsError(`${where}: expected a question object`);
  if (
    value.type !== "noul" &&
    value.type !== "score" &&
    value.type !== "choice"
  )
    throw new InvalidQuestionsError(
      `${where}.type: expected "noul", "score", or "choice"`,
    );
  if (value.instructions !== undefined && !isEntryType(value.instructions))
    throw new InvalidQuestionsError(
      `${where}.instructions: expected text, a JSON value, or null`,
    );
  if (value.type === "noul") {
    if (value.criteria !== undefined && value.criteria !== null) {
      if (!isPlainObject(value.criteria))
        throw new InvalidQuestionsError(
          `${where}.criteria: expected an object`,
        );
      for (const key of ["true", "false"] as const)
        if (
          value.criteria[key] !== undefined &&
          !isEntryType(value.criteria[key])
        )
          throw new InvalidQuestionsError(
            `${where}.criteria.${key}: expected text, a JSON value, or null`,
          );
    }
    return {
      type: "noul",
      instructions: value.instructions ?? null,
      criteria:
        (value.criteria as
          | { true?: EntryType; false?: EntryType }
          | null
          | undefined) ?? null,
    };
  }
  if (value.type === "score") {
    if (!Array.isArray(value.criteria))
      throw new InvalidQuestionsError(`${where}.criteria: expected an array`);
    for (const [index, criterion] of value.criteria.entries())
      if (!isEntryType(criterion))
        throw new InvalidQuestionsError(
          `${where}.criteria.${index}: expected text, a JSON value, or null`,
        );
    return {
      type: "score",
      instructions: value.instructions ?? null,
      criteria: value.criteria,
    };
  }
  if (!isPlainObject(value.criteria))
    throw new InvalidQuestionsError(`${where}.criteria: expected an object`);
  for (const [label, criterion] of Object.entries(value.criteria))
    if (!isEntryType(criterion))
      throw new InvalidQuestionsError(
        `${where}.criteria.${JSON.stringify(label)}: expected text, a JSON value, or null`,
      );
  return {
    type: "choice",
    instructions: value.instructions ?? null,
    criteria: value.criteria as Record<string, EntryType>,
  };
};

/** Validate a question collection and normalize it to plain data. */
const validateQuestions = (questions: Questions): Record<string, Question> => {
  const entries = Object.entries(questions ?? {});
  if (entries.length === 0)
    throw new InvalidQuestionsError("At least one question is required.");
  const validated: Record<string, Question> = {};
  for (const [questionId, value] of entries)
    validated[questionId] = structuredClone(
      validateQuestion(questionId, value),
    );
  for (const question of Object.values(validated))
    if (
      question.type !== "noul" &&
      Object.keys(question.criteria).length < MIN_CRITERIA
    )
      throw new InvalidQuestionsError(
        "Score and choice questions require at least two criteria.",
      );
  return validated;
};

/** Serialize an instruction or criterion value for inclusion in a prompt. */
const serializeInstructionValue = (value: unknown): string => {
  if (value === null || value === undefined)
    return "No additional instructions.";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
};

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
  if (question.type === "score") {
    if (mode === "discrete")
      return { kind: "integer", max: question.criteria.length };
    return {
      kind: "probabilityMap",
      labels: question.criteria.map((_, score) => String(score)),
    };
  }
  const labels = Object.keys(question.criteria);
  if (mode === "discrete") return { kind: "enum", values: labels };
  return { kind: "probabilityMap", labels };
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
      const criteria: [string, EntryType][] =
        question.type === "score"
          ? question.criteria.map((criterion, score) => [
              String(score),
              criterion,
            ])
          : question.type === "choice"
            ? Object.entries(question.criteria)
            : [];
      for (const [label, criterion] of criteria)
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
  for (const [index, questionId] of spec.questionIds.entries()) {
    const answerSpec = spec.answers[questionId];
    if (answerSpec.kind === "probabilityMap")
      answerProperties[questionId] = { $ref: `#/$defs/ProbabilityMap${index}` };
    else if (answerSpec.kind === "boolean")
      answerProperties[questionId] = {
        description: spec.descriptions[questionId],
        type: "boolean",
      };
    else if (answerSpec.kind === "probability")
      answerProperties[questionId] = {
        description: spec.descriptions[questionId],
        type: "number",
      };
    else if (answerSpec.kind === "integer")
      answerProperties[questionId] = {
        description: spec.descriptions[questionId],
        type: "integer",
      };
    else
      answerProperties[questionId] = {
        description: spec.descriptions[questionId],
        type: "string",
        enum: answerSpec.values,
      };
  }

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

const isProbability = (value: unknown): boolean =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= 1;

/** Describe a value for a validation message. */
const describeValue = (value: unknown): string => {
  const json = JSON.stringify(value);
  const type =
    value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
  return `${json} (${type})`;
};

/** Validate one answer value against its spec, appending any issues. */
const validateAnswerValue = (
  questionId: string,
  spec: AnswerSpec,
  value: unknown,
  issues: string[],
): void => {
  const where = `answers.${questionId}`;
  if (spec.kind === "boolean") {
    if (typeof value !== "boolean")
      issues.push(
        `${where}: expected true or false, got ${describeValue(value)}`,
      );
    return;
  }
  if (spec.kind === "probability") {
    if (!isProbability(value))
      issues.push(
        `${where}: expected a number between 0 and 1, got ${describeValue(value)}`,
      );
    return;
  }
  if (spec.kind === "integer") {
    if (
      typeof value !== "number" ||
      !Number.isInteger(value) ||
      value < 0 ||
      value >= spec.max
    )
      issues.push(
        `${where}: expected an integer in [0, ${spec.max}), got ${describeValue(value)}`,
      );
    return;
  }
  if (spec.kind === "enum") {
    if (typeof value !== "string" || !spec.values.includes(value))
      issues.push(
        `${where}: expected one of ${JSON.stringify(spec.values)}, got ${describeValue(value)}`,
      );
    return;
  }
  if (!isPlainObject(value)) {
    issues.push(`${where}: expected an object, got ${describeValue(value)}`);
    return;
  }
  for (const label of spec.labels)
    if (!(label in value))
      issues.push(
        `${where}: missing required property ${JSON.stringify(label)}`,
      );
  for (const key of Object.keys(value))
    if (!spec.labels.includes(key))
      issues.push(
        `${where}: extra property ${JSON.stringify(key)} is not allowed`,
      );
  for (const label of spec.labels)
    if (label in value && !isProbability(value[label]))
      issues.push(
        `${where}.${label}: expected a number between 0 and 1, got ${describeValue(value[label])}`,
      );
};

/** Validate a parsed model response against the output spec. */
const validateOutput = (
  spec: OutputSpec,
  payload: unknown,
): Record<string, unknown> => {
  const issues: string[] = [];
  if (!isPlainObject(payload)) {
    issues.push(`expected a JSON object, got ${describeValue(payload)}`);
    throw new OutputValidationError(issues);
  }
  for (const key of Object.keys(payload))
    if (key !== "answers")
      issues.push(`extra property ${JSON.stringify(key)} is not allowed`);
  const answers = payload.answers;
  if (!("answers" in payload)) {
    issues.push('missing required property "answers"');
    throw new OutputValidationError(issues);
  }
  if (!isPlainObject(answers)) {
    issues.push(`answers: expected an object, got ${describeValue(answers)}`);
    throw new OutputValidationError(issues);
  }
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
  for (const questionId of spec.questionIds)
    if (questionId in answers)
      validateAnswerValue(
        questionId,
        spec.answers[questionId],
        answers[questionId],
        issues,
      );
  if (issues.length > 0) throw new OutputValidationError(issues);
  return answers;
};

export {
  type AnswerMode,
  type AnswerSpec,
  buildOutputSpec,
  buildSchema,
  InvalidQuestionsError,
  type JsonSchema,
  type OutputSpec,
  OutputValidationError,
  type Question,
  validateOutput,
  validateQuestions,
};
