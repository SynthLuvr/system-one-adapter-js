import { describe, expect, it } from "vitest";
import {
  choiceConfidence,
  scoreConfidence,
} from "../utils/confidenceMetrics.js";

interface ConfidenceCase {
  metric: (probs: readonly number[]) => number;
  probabilities: readonly number[];
  expected: number;
}

const cases: readonly ConfidenceCase[] = [
  {
    metric: scoreConfidence,
    probabilities: [0.2, 0.2, 0.2, 0.2, 0.2],
    expected: 0.0,
  },
  {
    metric: scoreConfidence,
    probabilities: [0.04, 0.04, 0.04, 0.04, 0.04],
    expected: 0.0,
  },
  {
    metric: scoreConfidence,
    probabilities: [0.01, 0.02, 0.07, 0.3, 0.6],
    expected: 0.55,
  },
  {
    metric: choiceConfidence,
    probabilities: [0.5, 0.5],
    expected: 0.0,
  },
  {
    metric: choiceConfidence,
    probabilities: [0.2, 0.2],
    expected: 0.0,
  },
  {
    metric: choiceConfidence,
    probabilities: [0.82, 0.18],
    expected: 0.64,
  },
  {
    metric: scoreConfidence,
    probabilities: [1.0],
    expected: 1.0,
  },
  {
    metric: choiceConfidence,
    probabilities: [1.0],
    expected: 1.0,
  },
];

describe("confidence metrics", () => {
  for (const { metric, probabilities, expected } of cases)
    it(`${metric.name}(${probabilities.join(", ")})`, () => {
      expect(metric(probabilities)).toBeCloseTo(expected, 5);
    });
});
