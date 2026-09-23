import { vitestPreset } from "ts-canon/presets/vitest";

const config = vitestPreset({
  setupFiles: ["src/tests/setup.ts"],
  coverage: {
    provider: "v8",
    include: ["src/**/*.ts"],
    // src/laya-ts is vendored upstream code (see src/laya-ts/VENDOR.md);
    // its own suite lives upstream, so it stays out of this repo's gate.
    exclude: ["src/tests/**", "src/laya-ts/**"],
    thresholds: {
      lines: 80,
      functions: 80,
      statements: 80,
      branches: 80,
    },
  },
});

export { config as default };
