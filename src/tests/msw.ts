import { HttpResponse, http } from "msw";
import { setupServer } from "msw/node";

/** One recorded request body, parsed from JSON. */
type RequestBody = Record<string, unknown>;

/** A response an endpoint reply hands back to the caller. */
type ReplyResponse = InstanceType<typeof HttpResponse>;

/** The reply for one request: a JSON payload or a ready-made response. */
type Reply = (
  body: RequestBody,
  index: number,
) =>
  | Record<string, unknown>
  | ReplyResponse
  | Promise<Record<string, unknown> | ReplyResponse>;

/** An endpoint whose served request bodies are recorded for assertions. */
interface RecordedEndpoint {
  /** The handler a test installs with `server.use(endpoint.handler)`. */
  handler: ReturnType<typeof http.post>;
  /** The JSON body of every served request, in arrival order. */
  requests: RequestBody[];
}

/**
 * The integration-test HTTP server. `setup.ts` starts, resets, and stops it
 * around every test; requests without an installed handler are errors, so a
 * request the suite did not intend fails the test instead of reaching the
 * network.
 */
const server = setupServer();

/** Record one POST endpoint and reply to each request through `reply`. */
const postEndpoint = (path: string, reply: Reply): RecordedEndpoint => {
  const requests: RequestBody[] = [];
  const handler = http.post(`*${path}`, async ({ request }) => {
    const body = (await request.json()) as RequestBody;
    const response = await reply(body, requests.length);
    requests.push(body);
    return response instanceof HttpResponse
      ? response
      : HttpResponse.json(response);
  });
  return { handler, requests };
};

/** A recorded `POST /v1/responses` endpoint, the OpenAI Responses API. */
const openAIResponsesEndpoint = (reply: Reply): RecordedEndpoint =>
  postEndpoint("/v1/responses", reply);

/** A recorded `POST /v1/chat/completions` endpoint. */
const openAIChatEndpoint = (reply: Reply): RecordedEndpoint =>
  postEndpoint("/v1/chat/completions", reply);

/** A recorded `POST /v1/messages` endpoint, the Anthropic Messages API. */
const anthropicEndpoint = (reply: Reply): RecordedEndpoint =>
  postEndpoint("/v1/messages", reply);

/** A completed OpenAI Responses API payload carrying `text`. */
const openAIResponsesPayload = (
  text: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({
  id: "resp-test",
  object: "response",
  created_at: 0,
  status: "completed",
  model: "test-model",
  output: [
    {
      id: "msg-test",
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text, annotations: [] }],
    },
  ],
  usage: { input_tokens: 12, output_tokens: 7, total_tokens: 19 },
  ...overrides,
});

/** A completed OpenAI Chat Completions payload carrying `text`. */
const openAIChatPayload = (
  text: string | null,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({
  id: "chatcmpl-test",
  object: "chat.completion",
  created: 0,
  model: "test-model",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: text },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 },
  ...overrides,
});

/** A completed Anthropic Messages payload carrying `text`. */
const anthropicPayload = (
  text: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({
  id: "msg-test",
  type: "message",
  role: "assistant",
  model: "claude-haiku-4-5",
  stop_reason: "end_turn",
  stop_sequence: null,
  content: [{ type: "text", text }],
  usage: { input_tokens: 20, output_tokens: 10 },
  ...overrides,
});

/** An HTTP error response with a provider-shaped JSON error body. */
const jsonResponseError = (
  status: number,
  message = "unavailable",
  headers: Record<string, string> = {},
): ReplyResponse =>
  HttpResponse.json({ error: { message } }, { status, headers });

export {
  anthropicEndpoint,
  anthropicPayload,
  jsonResponseError,
  openAIChatEndpoint,
  openAIChatPayload,
  openAIResponsesEndpoint,
  openAIResponsesPayload,
  type RequestBody,
  server,
};
