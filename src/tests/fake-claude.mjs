#!/usr/bin/env node
// A fake `claude` CLI for the integration suite: it accepts the same flags
// the provider passes, records every invocation, and prints scripted result
// payloads shaped like the real CLI's `--output-format json` output.
import { readFileSync, writeFileSync } from "node:fs";

const argv = process.argv.slice(2);
const env = process.env;

const readStdin = async () => {
  let text = "";
  for await (const chunk of process.stdin) text += chunk;
  return text;
};

const stdin = await readStdin();

const logFile = env.FAKE_CLAUDE_LOG;
let calls = 0;
if (logFile !== undefined) {
  let logged = "";
  try {
    logged = readFileSync(logFile, "utf8");
  } catch {
    // The first invocation of a fresh log.
  }
  calls = logged.split("\n").filter((line) => line !== "").length;
  const entry = JSON.stringify({
    argv,
    stdin,
    env: { MAX_THINKING_TOKENS: env.MAX_THINKING_TOKENS },
  });
  writeFileSync(logFile, `${logged}${entry}\n`);
}

// One scripted behavior per invocation, replaying the last when spent: a
// payload object is merged over a success payload, a "!directive" string
// fails the process instead.
const script = JSON.parse(env.FAKE_CLAUDE_SCRIPT ?? "[]");
const behavior =
  script.length === 0 ? {} : script[Math.min(calls, script.length - 1)];

const okPayload = {
  subtype: "success",
  is_error: false,
  stop_reason: "end_turn",
  result: '{"answers":{"positive":true}}',
  usage: {
    input_tokens: 3,
    output_tokens: 4,
    cache_creation_input_tokens: 100,
    cache_read_input_tokens: 200,
  },
};

const emit = (payload) => {
  process.stdout.write(
    JSON.stringify({ type: "result", ...okPayload, ...payload }),
  );
  process.exit(0);
};

if (typeof behavior === "string" && behavior.startsWith("!")) {
  const [directive, ...rest] = behavior.slice(1).split(":");
  if (directive === "exit") {
    const [code, ...message] = rest;
    process.stderr.write(message.join(":"));
    process.exit(Number(code));
  }
  if (directive === "garbage") {
    process.stdout.write("definitely not JSON");
    process.exit(0);
  }
  if (directive === "sleep") {
    await new Promise((resolve) => setTimeout(resolve, Number(rest[0])));
    emit({});
  }
  process.stderr.write(`unknown directive: ${behavior}`);
  process.exit(2);
}
emit(behavior);
