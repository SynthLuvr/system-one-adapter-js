import type {
  ChoiceQuestion,
  NoulQuestion,
  Questions,
  ResultFor,
  ScoreQuestion,
  Usage,
} from "@typesafe-ai/sdk";
import type { LlmAttempt } from "./providers/base.js";

/** Token usage of the final attempt alongside cumulative retry accounting. */
interface AdapterUsage extends Usage {
  input_tokens_total: number;
  output_tokens_total: number;
  n_retries: number;
  n_retries_malformed_structure: number;
  latency: number;
}

/** Diagnostics attached to responses and terminal errors. */
interface AdapterDebug {
  max_error: number;
  invalid_probs: number;
  probability_errors: Record<string, number>;
  original_probabilities?: Record<string, Record<string, number>>;
  llm_attempts: LlmAttempt[];
  retry_reasons: [string, string][];
}

/** A TypeSafeError carrying attempt traces and retry reasons. */
type DebuggedError = Error & { debug?: Record<string, unknown> };

/** Answers of noul questions, keyed by question name. */
type NoulView<Q extends Questions> = {
  [K in keyof Q as Q[K] extends NoulQuestion ? K : never]: ResultFor<Q[K]>;
};

/** Answers of score questions, keyed by question name. */
type ScoreView<Q extends Questions> = {
  [K in keyof Q as Q[K] extends ScoreQuestion ? K : never]: ResultFor<Q[K]>;
};

/** Answers of choice questions, keyed by question name. */
type ChoiceView<Q extends Questions> = {
  [K in keyof Q as Q[K] extends ChoiceQuestion ? K : never]: ResultFor<Q[K]>;
};

/** SDK answers and typed views with retry and probability diagnostics. */
interface SystemOneResponse<Q extends Questions = Questions> {
  /** The model used to answer the request. */
  readonly model: string;
  /** Answers with types inferred from the supplied questions. */
  readonly answers: { readonly [K in keyof Q]: ResultFor<Q[K]> };
  /** Final-attempt usage with cumulative retry accounting. */
  readonly usage: AdapterUsage;
  /** Probability diagnostics and traces of every provider attempt. */
  readonly debug: AdapterDebug;
  /** Noul answers, keyed by question name. */
  readonly nouls: NoulView<Q>;
  /** Score answers, keyed by question name. */
  readonly scores: ScoreView<Q>;
  /** Choice answers, keyed by question name. */
  readonly choices: ChoiceView<Q>;
  /** A JSON-compatible view of the response. */
  toJSON(): {
    model: string;
    usage: AdapterUsage;
    answers: { readonly [K in keyof Q]: ResultFor<Q[K]> };
    debug: Record<string, unknown>;
  };
}

/** Attach attempt traces and retry reasons to a terminal SDK error. */
const attachDebug = (error: unknown, debug: Record<string, unknown>): void => {
  if (error instanceof Error) (error as DebuggedError).debug = debug;
};

export {
  type AdapterDebug,
  type AdapterUsage,
  attachDebug,
  type SystemOneResponse,
};
