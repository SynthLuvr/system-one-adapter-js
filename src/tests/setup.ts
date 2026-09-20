import { afterAll, afterEach, beforeAll } from "vitest";
import { server } from "./msw.js";

// Provider SDKs demand a key when a client is built, before any HTTP happens.
process.env.OPENAI_API_KEY ??= "test-key";
process.env.ANTHROPIC_API_KEY ??= "test-key";

// Unhandled requests are errors, so unintended traffic fails tests instead of
// reaching the network; the transport-failure tests rely on exactly that.
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
