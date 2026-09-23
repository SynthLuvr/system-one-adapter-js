import { vitestPreset } from "ts-canon/presets/vitest";

const config = vitestPreset({
  setupFiles: ["src/tests/setup.ts"],
  coverage: {
    provider: "v8",
    include: ["src/**/*.ts"],
    exclude: ["src/tests/**"],
    thresholds: {
      lines: 80,
      functions: 80,
      statements: 80,
      branches: 80,
    },
  },
});

export { config as default };
