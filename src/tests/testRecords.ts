import { type } from "arktype";

/** arktype type of a JSON record with unknown values. */
const JsonRecord = type({ "[string]": "unknown" });

/** arktype type of an array of JSON records. */
const JsonRecordArray = JsonRecord.array();

/** arktype type of a string value. */
const StringValue = type("string");

/** arktype type of a provider-neutral chat message. */
const ProviderMessage = type({
  role: "'system'|'user'|'assistant'",
  content: "string",
});

/** arktype type of an error or response carrying attached debug data. */
const DebugCarrier = type({ debug: JsonRecord });

/** Validate an unknown value as a JSON record, throwing on mismatch. */
const asRecord = (value: unknown): Record<string, unknown> => {
  const record = JsonRecord(value);
  if (record instanceof type.errors)
    throw new Error(`Expected a JSON record: ${record.summary}`);
  return record;
};

/** Validate an unknown value as an array of JSON records. */
const asRecords = (value: unknown): Record<string, unknown>[] => {
  const records = JsonRecordArray(value);
  if (records instanceof type.errors)
    throw new Error(`Expected JSON records: ${records.summary}`);
  return records;
};

/** Read the debug data attached to an adapter error or response. */
const debugOf = (value: unknown): Record<string, unknown> => {
  const carrier = DebugCarrier(value);
  if (carrier instanceof type.errors)
    throw new Error(`Expected attached debug data: ${carrier.summary}`);
  return carrier.debug;
};

/** Validate an unknown value as a string, throwing on mismatch. */
const asString = (value: unknown): string => {
  const result = StringValue(value);
  if (result instanceof type.errors)
    throw new Error(`Expected a string: ${result.summary}`);
  return result;
};

/** Validate an unknown value as a provider-neutral chat message. */
const asMessage = (
  value: unknown,
): { role: "system" | "user" | "assistant"; content: string } => {
  const message = ProviderMessage(value);
  if (message instanceof type.errors)
    throw new Error(`Expected a message: ${message.summary}`);
  return message;
};

/** The request body text of a captured fetch call. */
const bodyText = (init: RequestInit | undefined): string =>
  typeof init?.body === "string" ? init.body : "";

export { asMessage, asRecord, asRecords, asString, bodyText, debugOf };
