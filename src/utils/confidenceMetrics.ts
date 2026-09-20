const SINGLE_ANSWER_CONFIDENCE = 1;

/** Measure score concentration around its modal score. */
const scoreConfidence = (probs: readonly number[]): number => {
  if (probs.length === 1) return SINGLE_ANSWER_CONFIDENCE;

  const normalizedProbs = normalize(probs);
  const modeIndex = normalizedProbs.reduce(
    (best, probability, index) =>
      probability > normalizedProbs[best] ? index : best,
    0,
  );
  let distanceFromMode = 0;
  for (const [index, probability] of normalizedProbs.entries())
    distanceFromMode += probability * Math.abs(index - modeIndex);
  const uniformCenter = (normalizedProbs.length - 1) / 2;
  let uniformMad = 0;
  for (let index = 0; index < normalizedProbs.length; index++)
    uniformMad += Math.abs(index - uniformCenter);
  uniformMad /= normalizedProbs.length;
  return Math.max(0, 1 - distanceFromMode / uniformMad);
};

/** Scale peak choice probability from uniform to certainty. */
const choiceConfidence = (probs: readonly number[]): number => {
  if (probs.length === 1) return SINGLE_ANSWER_CONFIDENCE;

  const normalizedProbs = normalize(probs);
  const uniformProbability = 1 / normalizedProbs.length;
  return (
    (Math.max(...normalizedProbs) - uniformProbability) /
    (1 - uniformProbability)
  );
};

/** Normalize confidence inputs, using uniform probabilities for zero totals. */
const normalize = (probs: readonly number[]): number[] => {
  const total = probs.reduce((sum, probability) => sum + probability, 0);
  if (total === 0)
    return Array.from<number>({ length: probs.length }).fill(1 / probs.length);
  return probs.map((probability) => probability / total);
};

export { choiceConfidence, scoreConfidence };
