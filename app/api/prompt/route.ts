// app/api/prompt/route.ts
import { NextResponse } from "next/server";

const EMBEDDING_MODEL =
  process.env.EMBEDDING_MODEL ?? "RPRTHPB-text-embedding-3-small";
const CHAT_MODEL = process.env.CHAT_MODEL ?? "RPRTHPB-gpt-5-mini";

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_HOST = process.env.PINECONE_HOST;

const LLMOD_API_KEY = process.env.LLMOD_API_KEY;
const LLMOD_BASE_URL = process.env.LLMOD_BASE_URL ?? "";

// Must match /api/stats (informational)
const CHUNK_SIZE = 1024;
const OVERLAP_RATIO = 0.2;
const TOP_K = 5;

// --- RAG guardrails / stabilization ---
const MIN_SCORE = Number(process.env.MIN_SCORE ?? "0.55");
const MODEL_CONTEXT_K = Math.max(
  1,
  Number(process.env.MODEL_CONTEXT_K ?? "3")
);
const MAX_CONTEXT_CHARS_PER_CHUNK = Number(
  process.env.MAX_CONTEXT_CHARS_PER_CHUNK ?? "1800"
);
const MAX_TOTAL_CONTEXT_CHARS = Number(
  process.env.MAX_TOTAL_CONTEXT_CHARS ?? "6500"
);

type ContextItem = {
  talk_id: string;
  title: string;
  chunk: string;
  score: number;
};

type PineconeMatch = {
  id?: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

type PineconeQueryResponse = {
  matches?: PineconeMatch[];
};

function mustEnv(name: string, v: string | undefined | null): string {
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

function isNonEmptyString(x: unknown): x is string {
  return typeof x === "string" && x.trim().length > 0;
}

function getString(md: Record<string, unknown>, key: string): string {
  const v = md[key];
  return typeof v === "string" ? v : String(v ?? "");
}

function extractChunk(md: Record<string, unknown>): string {
  const v = md["chunk_text"] ?? md["chunk"] ?? md["text"] ?? md["passage"] ?? "";
  return String(v ?? "");
}

function normalize(s: string): string {
  return (s ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function pickKeywords(question: string): string[] {
  const STOP = new Set([
    "what","does","who","why","when","where","how",
    "suggest","people","tell","say",
    "based","provided","data","ted"
  ]);

  // חשוב: עדיפות למילים מסוף השאלה (בד"כ הכי ספציפיות)
  const words = question
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length >= 4 && !STOP.has(w));

  // הפוך כדי להתחיל מהסוף (home, emissions, reduce...)
  return words.reverse().slice(0, 12);
}

function smartSnippet(text: string, question: string, maxChars: number): string {
  const t = normalize(text);
  if (t.length <= maxChars) return t;

  const kws = pickKeywords(question);
  const lower = t.toLowerCase();

  // בוחרים "hit" הכי מאוחר כדי להגדיל סיכוי לכלול את אזור התשובה
  let bestHit = -1;
  for (const k of kws) {
    const i = lower.lastIndexOf(k);
    if (i > bestHit) bestHit = i;
  }

  if (bestHit === -1) return t.slice(0, maxChars).trim() + " …";

  const before = Math.floor(maxChars * 0.35);
  const start = Math.max(0, bestHit - before);
  const end = Math.min(t.length, start + maxChars);

  return (
    (start > 0 ? "… " : "") +
    t.slice(start, end).trim() +
    (end < t.length ? " …" : "")
  );
}


async function embedQuery(text: string): Promise<number[]> {
  const baseUrl = mustEnv("LLMOD_BASE_URL", LLMOD_BASE_URL).replace(/\/$/, "");
  const apiKey = mustEnv("LLMOD_API_KEY", LLMOD_API_KEY);

  const url = `${baseUrl}/embeddings`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Embeddings error (${res.status}): ${t}`);
  }

  const j: any = await res.json();
  const vec = j?.data?.[0]?.embedding;

  if (!Array.isArray(vec)) {
    throw new Error(
      `Unexpected embeddings response: ${JSON.stringify(j).slice(0, 400)}`
    );
  }

  return vec as number[];
}

async function pineconeQuery(vec: number[], topK: number): Promise<PineconeQueryResponse> {
  const host = mustEnv("PINECONE_HOST", PINECONE_HOST).replace(/\/$/, "");
  const apiKey = mustEnv("PINECONE_API_KEY", PINECONE_API_KEY);

  const url = `${host}/query`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Api-Key": apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ vector: vec, topK, includeMetadata: true }),
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Pinecone query error (${res.status}): ${t}`);
  }

  return (await res.json()) as PineconeQueryResponse;
}

async function callChat(system: string, user: string): Promise<string> {
  const baseUrl = mustEnv("LLMOD_BASE_URL", LLMOD_BASE_URL).replace(/\/$/, "");
  const apiKey = mustEnv("LLMOD_API_KEY", LLMOD_API_KEY);

  const url = `${baseUrl}/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      temperature: 1, // gpt-5 via litellm requires temperature=1
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Chat error (${res.status}): ${t}`);
  }

  const j: any = await res.json();
  const content = j?.choices?.[0]?.message?.content;

  if (typeof content !== "string") {
    throw new Error(
      `Unexpected chat response: ${JSON.stringify(j).slice(0, 400)}`
    );
  }

  return content.trim();
}

function buildContextBlock(modelContext: ContextItem[], question: string): string {
  let total = 0;
  const parts: string[] = [];

  for (let i = 0; i < modelContext.length; i++) {
    const c = modelContext[i];
    const snippet = smartSnippet(c.chunk, question, MAX_CONTEXT_CHARS_PER_CHUNK);

    const part =
      `[#${i + 1}] talk_id="${c.talk_id}" title="${c.title}" score=${c.score}\n` +
      snippet;

    if (total + part.length > MAX_TOTAL_CONTEXT_CHARS) break;

    parts.push(part);
    total += part.length + 8;
  }

  return parts.length ? parts.join("\n\n---\n\n") : "(No context retrieved)";
}

export async function POST(req: Request) {
  try {
    const body: unknown = await req.json();
    const question = (body as any)?.question;

    if (!isNonEmptyString(question)) {
      return NextResponse.json({ error: "Missing question" }, { status: 400 });
    }

    // 1) Embed
    const qVec = await embedQuery(question);

    // 2) Retrieve high topK (stabilization)
    const pineconeTopK = Math.min(30, Math.max(15, TOP_K * 6));
    const pc = await pineconeQuery(qVec, pineconeTopK);

    // 3) Normalize matches → ContextItem with score ALWAYS number
    const retrieved: ContextItem[] = (pc.matches ?? [])
      .map((m: PineconeMatch): ContextItem => {
        const md = (m.metadata ?? {}) as Record<string, unknown>;
        const score = typeof m.score === "number" ? m.score : 0;

        return {
          talk_id: getString(md, "talk_id"),
          title: getString(md, "title"),
          chunk: extractChunk(md),
          score,
        };
      })
      .filter((c) => isNonEmptyString(c.talk_id) && isNonEmptyString(c.chunk))
      .sort((a, b) => b.score - a.score);

    // 4) Pass only top 2–3 (MODEL_CONTEXT_K) but after MIN_SCORE
    const modelContext = retrieved
      .filter((c) => c.score >= MIN_SCORE)
      .slice(0, MODEL_CONTEXT_K);

    // 5) Required prompts
    const systemPrompt = `You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data." Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.`;

    const contextBlock = buildContextBlock(modelContext, question);

    const userPrompt =
      `QUESTION:\n${question}\n\n` +
      `CONTEXT:\n${contextBlock}\n\n` +
      `INSTRUCTIONS:\n` +
      `- Use only the context above.\n` +
      `- If the context contains the answer, you MUST answer.\n` +
      `- Cite the exact phrase(s) that support your answer.\n` +
      `- Otherwise say: "I don't know based on the provided TED data."`;

    // 6) If no relevant context, don't call LLM
    if (modelContext.length === 0) {
      return NextResponse.json({
        response: "I don't know based on the provided TED data.",
        context: [],
        Augmented_prompt: { System: systemPrompt, User: userPrompt },
      });
    }

    // 7) Chat
    const response = await callChat(systemPrompt, userPrompt);

    return NextResponse.json({
    response,
    context: modelContext,
    Augmented_prompt: { System: systemPrompt, User: userPrompt },
    });
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Unknown error" },
      { status: 500 }
    );
  }
}
