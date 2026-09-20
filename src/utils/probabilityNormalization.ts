import type { ValidatedAnswer } from "../schema.js";

const PROBABILITY_TOLERANCE = 1e-6;

/** How the LLM is asked to answer: per-label distributions or one value. */
type AnswerMode = "probabilities" | "discrete";

/** Result of processing one probability distribution. */
interface ProbabilityNormalization {
  /** Probabilities after optional normalization. */
  probabilities: Record<string, number>;
  /** Absolute distance of the original sum from one. */
  error: number;
  /** The original distribution when normalization changed it. */
  originalProbabilities?: Record<string, number>;
}

/** Probability diagnostics attached to adapter responses and errors. */
type ProbabilityDebug = {
  /** Largest absolute distance of an original probability sum from one. */
  max_error: number;
  /** Number of questions whose probability sum deviated from one. */
  invalid_probs: number;
  /** Absolute distance of each deviating question's sum from one. */
  probability_errors: Record<string, number>;
  /** Original distributions that normalization changed. */
  original_probabilities?: Record<string, Record<string, number>>;
};

/** Build probability diagnostics from question normalization results. */
const probabilityDebugData = (
  probabilityNormalizations: Readonly<
    Record<string, ProbabilityNormalization | undefined>
  >,
): ProbabilityDebug => {
  const errors: Record<string, number> = {};
  for (const [questionId, normalization] of Object.entries(
    probabilityNormalizations,
  ))
    if (normalization !== undefined) errors[questionId] = normalization.error;

  const probabilityErrors: Record<string, number> = {};
  for (const [questionId, error] of Object.entries(errors))
    if (error > PROBABILITY_TOLERANCE) probabilityErrors[questionId] = error;

  const originalProbabilities: Record<string, Record<string, number>> = {};
  for (const [questionId, normalization] of Object.entries(
    probabilityNormalizations,
  ))
    if (normalization?.originalProbabilities !== undefined)
      originalProbabilities[questionId] = normalization.originalProbabilities;

  const debugData: ProbabilityDebug = {
    max_error: Math.max(0, ...Object.values(errors)),
    invalid_probs: Object.keys(probabilityErrors).length,
    probability_errors: probabilityErrors,
  };
  if (Object.keys(originalProbabilities).length > 0)
    debugData.original_probabilities = originalProbabilities;
  return debugData;
};

/** Rescale probabilities to sum to 1, falling back to uniform for a zero total. */
const rescaleProbabilities = (
  probabilities: Readonly<Record<string, number>>,
): Record<string, number> => {
  const entries = Object.entries(probabilities);
  const total = entries.reduce((sum, [, probability]) => sum + probability, 0);
  if (total === 0) {
    const uniformProbability = 1 / entries.length;
    return Object.fromEntries(
      entries.map(([answer]) => [answer, uniformProbability]),
    );
  }
  return Object.fromEntries(
    entries.map(([answer, probability]) => [answer, probability / total]),
  );
};

/** Build and optionally normalize one answer's probability distribution. */
const normalizeAnswerProbabilities = (
  labels: readonly string[],
  value: ValidatedAnswer,
  { enabled }: { enabled: boolean },
): ProbabilityNormalization => {
  if (typeof value !== "object") {
    // Discrete answers carry the selected boolean, integer, or label.
    const selected = String(value);
    const probabilities = Object.fromEntries(
      labels.map((label) => [label, label === selected ? 1 : 0]),
    );
    return { probabilities, error: 0 };
  }

  const originalProbabilities = Object.fromEntries(
    labels.map((label) => [label, value[label]]),
  );
  const total = Object.values(originalProbabilities).reduce(
    (sum, probability) => sum + probability,
    0,
  );
  const error = Math.abs(total - 1);
  if (!enabled || error <= PROBABILITY_TOLERANCE)
    return { probabilities: originalProbabilities, error };

  return {
    probabilities: rescaleProbabilities(originalProbabilities),
    error,
    originalProbabilities,
  };
};

export {
  type AnswerMode,
  normalizeAnswerProbabilities,
  type ProbabilityDebug,
  type ProbabilityNormalization,
  probabilityDebugData,
  rescaleProbabilities,
};
