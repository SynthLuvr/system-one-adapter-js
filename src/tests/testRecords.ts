import { type ArkErrors, type } from "arktype";

/** A recorded JSON object with unknown values. */
const JsonRecord = type({ "[string]": "unknown" });

/** A provider-neutral chat message. */
const ProviderMessage = type({
  role: "'system'|'user'|'assistant'",
  content: "string",
});

/** A value carrying attached debug data. */
const DebugCarrier = type({ debug: JsonRecord });

/** Validate a value with an arktype check, throwing a summary on mismatch. */
const validated = <t>(check: () => t | ArkErrors, expected: string): t => {
  const result = check();
  if (result instanceof type.errors)
    throw new Error(`Expected ${expected}: ${result.summary}`);
  return result;
};

/** Validate an unknown value as a JSON record, throwing on mismatch. */
const asRecord = (value: unknown): Record<string, unknown> =>
  validated(() => JsonRecord(value), "a JSON record");

/** Validate an unknown value as an array of JSON records. */
const asRecords = (value: unknown): Record<string, unknown>[] =>
  validated(() => JsonRecord.array()(value), "JSON records");

/** Validate an unknown value as a string, throwing on mismatch. */
const asString = (value: unknown): string =>
  validated(() => type("string")(value), "a string");

/** Validate an unknown value as a provider-neutral chat message. */
const asMessage = (
  value: unknown,
): { role: "system" | "user" | "assistant"; content: string } =>
  validated(() => ProviderMessage(value), "a message");

/** Read the debug data attached to an adapter error or response. */
const debugOf = (value: unknown): Record<string, unknown> =>
  validated(() => DebugCarrier(value), "attached debug data").debug;

/** The request body text of a captured fetch call. */
const bodyText = (init: RequestInit | undefined): string =>
  typeof init?.body === "string" ? init.body : "";

export { asMessage, asRecord, asRecords, asString, bodyText, debugOf };
