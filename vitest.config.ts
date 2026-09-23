import { vitestPreset } from "ts-canon/presets/vitest";

const config = vitestPreset({ setupFiles: ["src/tests/setup.ts"] });

export { config as default };
