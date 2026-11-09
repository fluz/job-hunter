import OpenAI from 'openai';
import { GoogleGenAI, type GenerateContentConfig } from '@google/genai';
import type { TokenUsage } from './aiScorer';

export type AiProvider = 'openai' | 'gemini';

export const DEFAULT_AI_PROVIDER: AiProvider = 'openai';
export const DEFAULT_GEMINI_MODEL = 'gemini-3.8-flash';
export const DEFAULT_GEMINI_HARD_MODEL = 'gemini-3.8-flash';
export const GEMINI_MODEL_OPTIONS = [
  { value: 'gemini-3.8-flash', label: 'Gemini 3.8 Flash (recommended)' },
  { value: 'gemini-3.6-flash', label: 'Gemini 3.6 Flash' },
  { value: 'gemini-3.5-flash-lite', label: 'Gemini 3.5 Flash-Lite (lowest cost)' },
  { value: 'gemini-3.1-flash-lite', label: 'Gemini 3.1 Flash-Lite' },
  { value: 'gemini-3.1-pro-preview', label: 'Gemini 3.1 Pro (advanced reasoning)' },
] as const;
export const AI_PROVIDER_LABELS: Record<AiProvider, string> = {
  openai: 'OpenAI',
  gemini: 'Gemini',
};

export interface JsonGenerationRequest {
  systemPrompt?: string;
  userMessage: string;
  model: string;
  maxOutputTokens: number;
  schema: Record<string, unknown>;
  schemaName: string;
}

export interface TextGenerationRequest {
  systemPrompt?: string;
  userMessage: string;
  model: string;
  maxOutputTokens?: number;
  file?: {
    filename?: string;
    mimeType: string;
    dataBase64: string;
  };
}

export interface GeneratedJson<T> {
  result: T;
  usage: TokenUsage;
}

export interface GeneratedText {
  text: string;
  usage: TokenUsage;
}

export interface AiClient {
  readonly provider: AiProvider;
  generateJson<T>(request: JsonGenerationRequest): Promise<GeneratedJson<T>>;
  generateText(request: TextGenerationRequest): Promise<GeneratedText>;
}

function openAiUsage(response: {
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    input_tokens_details?: { cached_tokens?: number };
  };
}): TokenUsage {
  return {
    inputTokens: response.usage?.input_tokens ?? 0,
    cachedInputTokens: response.usage?.input_tokens_details?.cached_tokens ?? 0,
    outputTokens: response.usage?.output_tokens ?? 0,
  };
}

function geminiUsage(response: {
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
  };
}): TokenUsage {
  return {
    inputTokens: response.usageMetadata?.promptTokenCount ?? 0,
    cachedInputTokens: 0,
    outputTokens: response.usageMetadata?.candidatesTokenCount ?? 0,
  };
}

function requireResponseText(text: string | undefined, provider: AiProvider): string {
  if (!text?.trim()) throw new Error(`Empty response from ${AI_PROVIDER_LABELS[provider]}`);
  return text;
}

const GEMINI_MAX_ATTEMPTS = 3;
const GEMINI_REQUESTS_PER_MINUTE = 5;
const GEMINI_WINDOW_MS = 60_000;

class GeminiRequestLimiter {
  private readonly timestamps = new Map<string, number[]>();
  private readonly waiters = new Map<string, Promise<void>>();

  async wait(key: string): Promise<void> {
    const previous = this.waiters.get(key) ?? Promise.resolve();
    const next = previous.then(async () => {
      for (;;) {
        const now = Date.now();
        const recent = (this.timestamps.get(key) ?? []).filter((timestamp) => now - timestamp < GEMINI_WINDOW_MS);
        if (recent.length < GEMINI_REQUESTS_PER_MINUTE) {
          recent.push(now);
          this.timestamps.set(key, recent);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, Math.max(50, GEMINI_WINDOW_MS - (now - recent[0]) + 25)));
      }
    });
    this.waiters.set(key, next);
    try {
      await next;
    } finally {
      if (this.waiters.get(key) === next) this.waiters.delete(key);
    }
  }
}

const geminiRequestLimiter = new GeminiRequestLimiter();

function isTransientGeminiError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const candidate = error as { status?: unknown; code?: unknown; message?: unknown };
  const status = Number(candidate.status ?? candidate.code);
  return [429, 500, 502, 503, 504].includes(status)
    || (typeof candidate.message === 'string' && /\b(429|500|502|503|504)\b|high demand|temporarily unavailable|rate limit/i.test(candidate.message));
}

function geminiRetryDelayMs(error: unknown, attempt: number): number {
  const fallback = 500 * 2 ** (attempt - 1);
  if (!error || typeof error !== 'object') return fallback;
  const candidate = error as { message?: unknown; details?: unknown };
  const text = JSON.stringify(candidate);
  const seconds = text.match(/retry(?: in|delay["':\s]+)(\d+(?:\.\d+)?)\s*s/i)?.[1];
  return seconds ? Math.max(fallback, Math.ceil(Number(seconds) * 1000)) : fallback;
}

class OpenAiClient implements AiClient {
  readonly provider = 'openai' as const;
  private readonly client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  async generateJson<T>(request: JsonGenerationRequest): Promise<GeneratedJson<T>> {
    const response = await this.client.responses.create({
      model: request.model,
      input: [
        ...(request.systemPrompt ? [{ role: 'system' as const, content: request.systemPrompt }] : []),
        { role: 'user' as const, content: request.userMessage },
      ],
      max_output_tokens: request.maxOutputTokens,
      text: {
        format: {
          type: 'json_schema',
          name: request.schemaName,
          strict: true,
          schema: request.schema,
        },
      },
    });
    const text = requireResponseText(response.output_text, this.provider);
    return { result: JSON.parse(text) as T, usage: openAiUsage(response) };
  }

  async generateText(request: TextGenerationRequest): Promise<GeneratedText> {
    const content = request.file
      ? [
          {
            type: 'input_file' as const,
            filename: request.file.filename || 'cv.pdf',
            file_data: `data:${request.file.mimeType};base64,${request.file.dataBase64}`,
          },
          { type: 'input_text' as const, text: request.userMessage },
        ]
      : request.userMessage;
    const response = await this.client.responses.create({
      model: request.model,
      input: [{ role: 'user', content }],
      ...(request.maxOutputTokens ? { max_output_tokens: request.maxOutputTokens } : {}),
    });
    return { text: requireResponseText(response.output_text, this.provider), usage: openAiUsage(response) };
  }
}

class GeminiClient implements AiClient {
  readonly provider = 'gemini' as const;
  private readonly client: GoogleGenAI;
  private readonly apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.client = new GoogleGenAI({ apiKey });
  }

  private async generate(request: {
    model: string;
    systemPrompt?: string;
    userMessage: string;
    maxOutputTokens?: number;
    responseJsonSchema?: Record<string, unknown>;
    file?: { mimeType: string; dataBase64: string };
  }) {
    const config: GenerateContentConfig = {
      ...(request.systemPrompt ? { systemInstruction: request.systemPrompt } : {}),
      ...(request.maxOutputTokens ? { maxOutputTokens: request.maxOutputTokens } : {}),
      ...(request.responseJsonSchema
        ? { responseMimeType: 'application/json', responseJsonSchema: request.responseJsonSchema }
        : {}),
    };
    const contents = request.file
      ? [{
          role: 'user' as const,
          parts: [
            { inlineData: { mimeType: request.file.mimeType, data: request.file.dataBase64 } },
            { text: request.userMessage },
          ],
        }]
      : request.userMessage;
    for (let attempt = 1; ; attempt += 1) {
      try {
        await geminiRequestLimiter.wait(`${this.apiKey}:${request.model}`);
        return await this.client.models.generateContent({ model: request.model, contents, config });
      } catch (error) {
        if (attempt >= GEMINI_MAX_ATTEMPTS || !isTransientGeminiError(error)) throw error;
        await new Promise((resolve) => setTimeout(resolve, geminiRetryDelayMs(error, attempt)));
      }
    }
  }

  async generateJson<T>(request: JsonGenerationRequest): Promise<GeneratedJson<T>> {
    const response = await this.generate({
      model: request.model,
      systemPrompt: request.systemPrompt,
      userMessage: request.userMessage,
      maxOutputTokens: request.maxOutputTokens,
      responseJsonSchema: request.schema,
    });
    const text = requireResponseText(response.text, this.provider);
    return { result: JSON.parse(text) as T, usage: geminiUsage(response) };
  }

  async generateText(request: TextGenerationRequest): Promise<GeneratedText> {
    const response = await this.generate(request);
    return { text: requireResponseText(response.text, this.provider), usage: geminiUsage(response) };
  }
}

export function normalizeAiProvider(value: string | null | undefined): AiProvider {
  return value === 'gemini' ? 'gemini' : DEFAULT_AI_PROVIDER;
}

export function resolveAiModel(provider: AiProvider, configuredModel: string, task: 'soft' | 'hard'): string {
  if (provider === 'gemini') {
    const model = configuredModel.trim();
    return model.startsWith('gemini-')
      ? model
      : task === 'hard' ? DEFAULT_GEMINI_HARD_MODEL : DEFAULT_GEMINI_MODEL;
  }
  return configuredModel;
}

export function createAiClient(provider: AiProvider, apiKey: string): AiClient {
  if (!apiKey.trim()) throw new Error(`${AI_PROVIDER_LABELS[provider]} API key is not configured.`);
  return provider === 'gemini' ? new GeminiClient(apiKey) : new OpenAiClient(apiKey);
}
