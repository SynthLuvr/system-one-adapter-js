import { describe, expect, it } from "vitest";
import {
  normalizeAnswerProbabilities,
  probabilityDebugData,
} from "../utils/probabilityNormalization.js";

interface NormalizationCase {
  enabled: boolean;
  rawProbability: number;
  expectedProbability: number;
  expectedOriginals?: Record<string, Record<string, number>>;
  expectedMaxError: number;
  expectedProbabilityErrors: Record<string, number>;
}

const cases: readonly NormalizationCase[] = [
  {
    enabled: false,
    rawProbability: 0.2,
    expectedProbability: 0.2,
    expectedOriginals: undefined,
    expectedMaxError: 0.6,
    expectedProbabilityErrors: { stars: 0.6, genre: 0.6 },
  },
  {
    enabled: true,
    rawProbability: 0.2,
    expectedProbability: 0.5,
    expectedOriginals: {
      stars: { "0": 0.2, "1": 0.2 },
      genre: { fiction: 0.2, nonfiction: 0.2 },
    },
    expectedMaxError: 0.6,
    expectedProbabilityErrors: { stars: 0.6, genre: 0.6 },
  },
  {
    enabled: false,
    rawProbability: 0.50000025,
    expectedProbability: 0.50000025,
    expectedOriginals: undefined,
    expectedMaxError: 5e-7,
    expectedProbabilityErrors: {},
  },
];

describe("probability normalization and debug data", () => {
  for (const normalizationCase of cases) {
    const { enabled, rawProbability } = normalizationCase;
    const label = `enabled=${String(enabled)}, raw=${String(rawProbability)}`;
    it(`normalizes and reports debug data (${label})`, () => {
      const {
        expectedProbability,
        expectedOriginals,
        expectedMaxError,
        expectedProbabilityErrors,
      } = normalizationCase;
      const scoreProbabilities = {
        "0": rawProbability,
        "1": rawProbability,
      };
      const choiceProbabilities = {
        fiction: rawProbability,
        nonfiction: rawProbability,
      };
      const probabilityNormalizations = {
        positive: undefined,
        stars: normalizeAnswerProbabilities(["0", "1"], scoreProbabilities, {
          enabled,
        }),
        genre: normalizeAnswerProbabilities(
          ["fiction", "nonfiction"],
          choiceProbabilities,
          { enabled },
        ),
      };
      const stars = probabilityNormalizations.stars;
      const genre = probabilityNormalizations.genre;
      expect(stars.probabilities["0"]).toBeCloseTo(expectedProbability, 6);
      expect(stars.probabilities["1"]).toBeCloseTo(expectedProbability, 6);
      expect(genre.probabilities.fiction).toBeCloseTo(expectedProbability, 6);
      expect(genre.probabilities.nonfiction).toBeCloseTo(
        expectedProbability,
        6,
      );

      const debugData = probabilityDebugData(probabilityNormalizations);
      expect(debugData.max_error).toBeCloseTo(expectedMaxError, 6);
      expect(debugData.invalid_probs).toEqual(
        Object.keys(expectedProbabilityErrors).length,
      );
      const probabilityErrors = debugData.probability_errors;
      expect(Object.keys(probabilityErrors).sort()).toEqual(
        Object.keys(expectedProbabilityErrors).sort(),
      );
      const expectedErrorEntries = Object.entries(expectedProbabilityErrors);
      for (const [questionId, expectedError] of expectedErrorEntries)
        expect(probabilityErrors[questionId]).toBeCloseTo(expectedError, 6);
      expect(debugData.original_probabilities).toEqual(expectedOriginals);
    });
  }

  it("builds discrete distributions from a selected label", () => {
    const normalization = normalizeAnswerProbabilities(["yes", "no"], "yes", {
      enabled: true,
    });
    expect(normalization.probabilities).toEqual({ yes: 1, no: 0 });
    expect(normalization.error).toBe(0);
  });
});
