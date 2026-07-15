import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// Load .env from repo root (../.env relative to this script)
const __dirname = dirname(fileURLToPath(import.meta.url));
const envPath = resolve(__dirname, "..", ".env");
try {
  const text = readFileSync(envPath, "utf-8");
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) continue;
    const key = trimmed.slice(0, eqIndex).trim();
    const value = trimmed.slice(eqIndex + 1).trim();
    if (key && !(key in process.env)) {
      process.env[key] = value;
    }
  }
} catch {
  // .env file is optional — skip silently
}

type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

type ChatResponse = {
  choices?: Array<{ message?: { content?: string } }>;
};

const apiKey = process.env.OPENROUTER_API_KEY;
const model = process.env.OPENROUTER_MODEL ?? "openrouter/auto";
const apiUrl = "https://openrouter.ai/api/v1/chat/completions";

if (!apiKey) {
  console.error("Set OPENROUTER_API_KEY before running this program.");
  process.exit(1);
}

async function complete(messages: Message[]): Promise<string> {
  const response = await fetch(apiUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, messages }),
  });

  if (!response.ok) {
    throw new Error(`OpenRouter returned HTTP ${response.status}: ${await response.text()}`);
  }

  const data = (await response.json()) as ChatResponse;
  const content = data.choices?.[0]?.message?.content;
  if (typeof content !== "string") {
    throw new Error(`OpenRouter returned an unexpected response: ${JSON.stringify(data)}`);
  }

  return content;
}

const messages: Message[] = [
  { role: "system", content: "You are a helpful, concise assistant." },
];
const rl = createInterface({ input, output });

console.log(`OpenRouter chat agent using ${model}. Type 'exit' to quit.`);

rl.setPrompt("You: ");
rl.prompt();

for await (const line of rl) {
  const userInput = line.trim();
  if (!userInput) continue;
  if (["exit", "quit"].includes(userInput.toLowerCase())) break;

  messages.push({ role: "user", content: userInput });

  try {
    const answer = await complete(messages);
    console.log(`Assistant: ${answer}\n`);
    messages.push({ role: "assistant", content: answer });
  } catch (error) {
    console.error("Error:", error instanceof Error ? error.message : error);
  }

  rl.prompt();
}
rl.close();
