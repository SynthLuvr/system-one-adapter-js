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

/** Build probability diagnostics from question normalization results. */
const probabilityDebugData = (
  probabilityNormalizations: Readonly<
    Record<string, ProbabilityNormalization | undefined>
  >,
): Record<string, unknown> => {
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

  const debugData: Record<string, unknown> = {
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

/** Build and optionally normalize a probability distribution. */
const normalizeProbabilitiesOfAllAnswers = (
  answers: readonly string[],
  value: unknown,
  answerMode: AnswerMode,
  { enabled }: { enabled: boolean },
): ProbabilityNormalization => {
  if (answerMode === "discrete") {
    const selected = String(value);
    const probabilities = Object.fromEntries(
      answers.map((answer) => [answer, answer === selected ? 1 : 0]),
    );
    return { probabilities, error: 0 };
  }

  const record = value as Record<string, unknown>;
  const originalProbabilities = Object.fromEntries(
    answers.map((answer) => [answer, Number(record[answer])]),
  );
  const total = Object.values(originalProbabilities).reduce(
    (sum, probability) => sum + probability,
    0,
  );
  const error = Math.abs(total - 1);
  if (!enabled || error <= PROBABILITY_TOLERANCE)
    return { probabilities: originalProbabilities, error };

  const probabilities = rescaleProbabilities(originalProbabilities);
  return { probabilities, error, originalProbabilities };
};

export {
  type AnswerMode,
  normalizeProbabilitiesOfAllAnswers,
  PROBABILITY_TOLERANCE,
  type ProbabilityNormalization,
  probabilityDebugData,
  rescaleProbabilities,
};
