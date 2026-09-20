import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  type RetryPolicy,
  TypeSafeError,
} from "@typesafe-ai/sdk";

/** Reason for performing one retry. */
interface RetryReason {
  /** Retry mechanism that requested another attempt. */
  category: "provider_error" | "malformed_structure";
  /** Detailed retry cause. */
  msg: string;
}

/** The SDK's default retry policy, with retries disabled by default. */
const DEFAULT_RETRY_POLICY: RetryPolicy = {
  maxRetries: 0,
  backoffInitialMs: 500,
  backoffMaxMs: 5000,
  backoffJitter: 0.25,
  httpStatuses: new Set([
    408,
    429,
    ...Array.from({ length: 100 }, (_, i) => 500 + i),
  ]),
  respectRetryAfter: true,
  maxRetryAfterMs: 60000,
  apiConnectionError: true,
  apiTimeoutError: true,
};

/** Fill unset retry fields with the adapter defaults. */
const resolveRetryPolicy = (partial?: Partial<RetryPolicy>): RetryPolicy => ({
  ...DEFAULT_RETRY_POLICY,
  ...partial,
  httpStatuses: partial?.httpStatuses ?? DEFAULT_RETRY_POLICY.httpStatuses,
});

/** The message of an unknown thrown value. */
const describeError = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

/** Whether the policy retries this SDK error. */
const isRetryable = (error: TypeSafeError, policy: RetryPolicy): boolean => {
  if (error instanceof APIError) return policy.httpStatuses.has(error.status);
  if (error instanceof APITimeoutError) return policy.apiTimeoutError;
  if (error instanceof APIConnectionError) return policy.apiConnectionError;
  return false;
};

/** Parse `retry-after-ms` or `Retry-After` into milliseconds. */
const parseRetryAfter = (
  headers: Headers,
  now = Date.now(),
): number | undefined => {
  const ms = Number(headers.get("retry-after-ms"));
  if (headers.has("retry-after-ms") && Number.isFinite(ms) && ms >= 0)
    return ms;
  const raw = headers.get("retry-after");
  if (raw === null) return undefined;
  const seconds = Number(raw);
  if (Number.isFinite(seconds))
    return seconds >= 0 ? seconds * 1000 : undefined;
  const date = Date.parse(raw);
  if (!Number.isNaN(date)) return Math.max(0, date - now);
  return undefined;
};

/** Calculate the delay for a zero-based retry attempt. */
const retryDelayMs = (
  attempt: number,
  headers: Headers | undefined,
  policy: RetryPolicy,
  random: () => number = Math.random,
): number => {
  if (policy.respectRetryAfter && headers !== undefined) {
    const retryAfter = parseRetryAfter(headers);
    if (retryAfter !== undefined && retryAfter <= policy.maxRetryAfterMs)
      return retryAfter;
  }
  const exponential = Math.min(
    policy.backoffInitialMs * 2 ** attempt,
    policy.backoffMaxMs,
  );
  return Math.round(exponential * (1 - random() * policy.backoffJitter));
};

/** Wait `ms` milliseconds. */
const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

/** The headers of an API error, when it carried any. */
const errorHeaders = (error: TypeSafeError): Headers | undefined =>
  error instanceof APIError ? error.headers : undefined;

/** Result of one retried call. */
interface RetryOutcome<T> {
  /** The value the call finally returned. */
  result: T;
  /** Transient retries spent before the final attempt. */
  nRetries: number;
}

/** Apply the SDK retry policy to a provider call that raises SDK errors. */
const runWithRetries = async <T>(
  fn: () => Promise<T>,
  retry: RetryPolicy,
  retryReasons?: RetryReason[],
): Promise<RetryOutcome<T>> => {
  for (let attempt = 0; ; attempt++) {
    try {
      const result = await fn();
      return { result, nRetries: attempt };
    } catch (error) {
      if (!(error instanceof TypeSafeError) || !isRetryable(error, retry))
        throw error;
      if (attempt >= retry.maxRetries) throw error;
      retryReasons?.push({
        category: "provider_error",
        msg: describeError(error),
      });
      await sleep(retryDelayMs(attempt, errorHeaders(error), retry));
    }
  }
};

/** Re-raise any provider SDK error from the block as an SDK error. */
const translating = async <T>(
  fn: () => Promise<T>,
  translate: (error: unknown) => TypeSafeError,
): Promise<T> => {
  try {
    return await fn();
  } catch (error) {
    if (error instanceof TypeSafeError) throw error;
    const translated = translate(error);
    translated.cause = error;
    throw translated;
  }
};

/** Map a provider timeout to the SDK timeout error. */
const toTimeoutError = (error: unknown): APITimeoutError =>
  new APITimeoutError(0, { cause: error });

/** Map a provider transport failure to the SDK connection error. */
const toConnectionError = (error: unknown): APIConnectionError =>
  new APIConnectionError(describeError(error), { cause: error });

/** Map a provider HTTP failure to the matching SDK error class. */
const toStatusError = (error: {
  status?: number;
  error?: unknown;
  headers?: Headers;
}): APIError =>
  APIError.fromResponse(
    error.status ?? 0,
    error.error,
    error.headers ?? new Headers(),
  );

export {
  DEFAULT_RETRY_POLICY,
  describeError,
  type RetryOutcome,
  type RetryReason,
  resolveRetryPolicy,
  runWithRetries,
  toConnectionError,
  toStatusError,
  toTimeoutError,
  translating,
};
