// app/api/prompt/route.ts
import { NextResponse } from "next/server";

const EMBEDDING_MODEL =
  process.env.EMBEDDING_MODEL ?? "RPRTHPB-text-embedding-3-small";
const CHAT_MODEL = process.env.CHAT_MODEL ?? "RPRTHPB-gpt-5-mini";

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_HOST = process.env.PINECONE_HOST;

const LLMOD_API_KEY = process.env.LLMOD_API_KEY;
const LLMOD_BASE_URL = process.env.LLMOD_BASE_URL ?? "";

// Informational only (aligns with your ingest)
const CHUNK_SIZE = 1024;
const OVERLAP_RATIO = 0.2;

// Default baseline
const TOP_K = 5;

// --- Strict / fact QA guardrails ---
const MIN_SCORE = Number(process.env.MIN_SCORE ?? "0.2");
const MODEL_CONTEXT_K = Math.max(1, Number(process.env.MODEL_CONTEXT_K ?? "3"));

const MAX_CONTEXT_CHARS_PER_CHUNK = Number(
  process.env.MAX_CONTEXT_CHARS_PER_CHUNK ?? "1800"
);
const MAX_TOTAL_CONTEXT_CHARS = Number(
  process.env.MAX_TOTAL_CONTEXT_CHARS ?? "6500"
);

// --- Listing / recommendation tuning ---
const LISTING_TOPK = Number(process.env.LISTING_TOPK ?? "30");
const LISTING_MIN_SCORE = Number(process.env.LISTING_MIN_SCORE ?? "0.20");

// --- Fallback retrieval (general robustness) ---
const FALLBACK_TOPK = Number(process.env.FALLBACK_TOPK ?? "30");
const FALLBACK_MIN_SCORE = Number(process.env.FALLBACK_MIN_SCORE ?? "0.35");

// ✅ Assignment constraint: Pinecone topK must be <= 30
const MAX_TOPK = 30;

// Small helpers
type ContextItem = {
  talk_id: string;
  title: string;
  chunk: string;
  score: number;
  url?: string;
  speaker_1?: string;
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
  const v =
    md["chunk_text"] ?? md["chunk"] ?? md["text"] ?? md["passage"] ?? "";
  return String(v ?? "");
}

function normalize(s: string): string {
  return (s ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * Extract first quoted string from question (single/double quotes).
 * This is general-purpose: many users specify talk names like '...'.
 */
function extractQuotedTitle(question: string): string | null {
  const q = question ?? "";
  const m1 = q.match(/'([^']{3,120})'/);
  if (m1?.[1]) return m1[1].trim();
  const m2 = q.match(/"([^"]{3,120})"/);
  if (m2?.[1]) return m2[1].trim();
  return null;
}

function normTitle(s: string): string {
  return (s ?? "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .trim();
}

/**
 * Keywords used only for snippet windowing (not retrieval).
 */
function pickKeywords(question: string): string[] {
  const STOP = new Set([
    "what",
    "does",
    "who",
    "why",
    "when",
    "where",
    "how",
    "tell",
    "say",
    "suggest",
    "recommend",
    "based",
    "provided",
    "data",
    "ted",
    "talk",
    "talks",
    "list",
    "return",
    "exactly",
    "about",
    "give",
    "provide",
    "find",
  ]);

  const words = question
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length >= 4 && !STOP.has(w));

  return words.reverse().slice(0, 12);
}

function smartSnippet(text: string, question: string, maxChars: number): string {
  const t = normalize(text);
  if (t.length <= maxChars) return t;

  const kws = pickKeywords(question);
  const lower = t.toLowerCase();

  let bestHit = -1;
  for (const k of kws) {
    const i = lower.indexOf(k);
    if (i !== -1 && (bestHit === -1 || i < bestHit)) bestHit = i;
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

// -------------------------
// Query capability detection
// -------------------------
function isMultiResultListingQ(q: string): boolean {
  const s = q.toLowerCase();
  const wants3 =
    s.includes("exactly 3") || s.includes("three") || /\b3\b/.test(s);
  const listing =
    s.includes("talk titles") ||
    s.includes("titles") ||
    s.includes("list") ||
    s.includes("return a list");
  return wants3 && listing;
}

function isRecommendationQ(q: string): boolean {
  const s = q.toLowerCase();
  return (
    s.includes("recommend") ||
    s.includes("which talk should") ||
    s.includes("what talk should") ||
    s.includes("suggest a talk") ||
    s.includes("what should i watch")
  );
}

function isKeyIdeaSummaryQ(q: string): boolean {
  const s = q.toLowerCase();
  return (
    s.includes("main idea") ||
    s.includes("key idea") ||
    s.includes("summarize") ||
    s.includes("summary") ||
    s.includes("what is this talk about")
  );
}

// -------------------------
// Embeddings / Pinecone / Chat
// -------------------------
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

async function pineconeQuery(
  vec: number[],
  topK: number,
  filter?: Record<string, any>
): Promise<PineconeQueryResponse> {
  const host = mustEnv("PINECONE_HOST", PINECONE_HOST).replace(/\/$/, "");
  const apiKey = mustEnv("PINECONE_API_KEY", PINECONE_API_KEY);

  const url = `${host}/query`;
  const body: any = { vector: vec, topK: Math.min(MAX_TOPK, topK), includeMetadata: true };
  if (filter && Object.keys(filter).length > 0) body.filter = filter;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Api-Key": apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
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
      temperature: 1, // required by your stack
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

// -------------------------
// Context helpers
// -------------------------
function buildTranscriptContextBlock(
  modelContext: ContextItem[],
  question: string
): string {
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

function buildTitleOnlyContextBlock(modelContext: ContextItem[]): string {
  if (!modelContext.length) return "(No context retrieved)";
  return modelContext
    .map(
      (c, i) =>
        `[#${i + 1}] talk_id="${c.talk_id}" title="${c.title}" score=${c.score}`
    )
    .join("\n");
}

// Keep best match per talk_id (diversity)
function dedupeByTalk(items: ContextItem[]): ContextItem[] {
  const best = new Map<string, ContextItem>();
  for (const it of items) {
    const prev = best.get(it.talk_id);
    if (!prev || it.score > prev.score) best.set(it.talk_id, it);
  }
  return Array.from(best.values());
}

// If user quoted a title, pull best chunk(s) from that talk if present in retrieved set
function pickByQuotedTitle(
  retrieved: ContextItem[],
  quotedTitle: string,
  k: number
): ContextItem[] {
  const qt = normTitle(quotedTitle);
  const hits = retrieved
    .filter((c) => isNonEmptyString(c.chunk) && normTitle(c.title) === qt)
    .sort((a, b) => b.score - a.score);
  return hits.slice(0, k);
}

function normalizeRetrieved(
  pc: PineconeQueryResponse,
  wantListing3: boolean
): ContextItem[] {
  const retrievedAll: ContextItem[] = (pc.matches ?? [])
    .map((m: PineconeMatch): ContextItem => {
      const md = (m.metadata ?? {}) as Record<string, unknown>;
      const score = typeof m.score === "number" ? m.score : 0;
      return {
        talk_id: getString(md, "talk_id"),
        title: getString(md, "title"),
        url: getString(md, "url"),
        speaker_1: getString(md, "speaker_1"),
        chunk: extractChunk(md),
        score,
      };
    })
    .filter((c) => {
      if (!isNonEmptyString(c.talk_id)) return false;
      if (!isNonEmptyString(c.title)) return false;
      if (wantListing3) return true; // titles only OK
      return isNonEmptyString(c.chunk); // transcript evidence required
    })
    .sort((a, b) => b.score - a.score);

  return retrievedAll;
}

// -------------------------
// Route
// -------------------------
export async function POST(req: Request) {
  try {
    const body: unknown = await req.json();
    const question = (body as any)?.question;

    if (!isNonEmptyString(question)) {
      return NextResponse.json({ error: "Missing question" }, { status: 400 });
    }

    // Detect capability
    const wantListing3 = isMultiResultListingQ(question);
    const wantRec = isRecommendationQ(question);
    const wantSummary = isKeyIdeaSummaryQ(question);

    // 1) Embed
    const qVec = await embedQuery(question);

    // Optional: if user quotes a title, we can bias retrieval with a metadata filter.
    // IMPORTANT: this is general and safe, but we do NOT rely on it exclusively.
    const quotedTitle = extractQuotedTitle(question);
    const titleFilter =
      quotedTitle && quotedTitle.length >= 3
        ? { title: { $eq: quotedTitle } }
        : undefined;

    // 2) First pass retrieval (normal) — ✅ cap to MAX_TOPK
    const topK1 =
      wantListing3 || wantRec
        ? Math.min(MAX_TOPK, Math.max(LISTING_TOPK, TOP_K * 10))
        : Math.min(MAX_TOPK, Math.max(20, TOP_K * 8));

    // Try with title filter only if we have it (helps pinpoint), otherwise general query.
    const pc1 = await pineconeQuery(qVec, topK1, titleFilter);
    const retrieved1 = normalizeRetrieved(pc1, wantListing3);

    // 3) Decide if we need fallback retrieval
    const distinctTalks1 = dedupeByTalk(retrieved1).length;

    const needsFallback =
      (wantListing3 && distinctTalks1 < 3) ||
      (!wantListing3 && retrieved1.length === 0) ||
      (!wantListing3 &&
        !wantRec &&
        !wantSummary &&
        retrieved1.filter((x) => x.score >= MIN_SCORE).length < 1);

    let retrievedAll = retrieved1;

    if (needsFallback) {
      // 2nd pass retrieval: no filter (avoid exact-match pitfalls), higher topK — ✅ cap to MAX_TOPK
      const pc2 = await pineconeQuery(
        qVec,
        Math.min(MAX_TOPK, FALLBACK_TOPK)
      );
      const retrieved2 = normalizeRetrieved(pc2, wantListing3);

      // Merge results (dedupe by vector id isn't available here, so we dedupe by talk+chunk prefix)
      const seen = new Set<string>();
      const merged: ContextItem[] = [];
      for (const r of [...retrieved1, ...retrieved2]) {
        const key = `${r.talk_id}::${r.title}::${r.chunk.slice(0, 60)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(r);
      }
      merged.sort((a, b) => b.score - a.score);
      retrievedAll = merged;
    }

    // 4) Build modelContext by capability
    let modelContext: ContextItem[] = [];

    if (wantListing3) {
      const filtered = retrievedAll.filter((c) => c.score >= LISTING_MIN_SCORE);
      const deduped = dedupeByTalk(filtered).sort((a, b) => b.score - a.score);
      modelContext = deduped.slice(0, 3);
    } else if (wantRec) {
      const filtered = retrievedAll.filter((c) => c.score >= LISTING_MIN_SCORE);
      const deduped = dedupeByTalk(filtered).sort((a, b) => b.score - a.score);
      modelContext = deduped.slice(0, Math.max(3, MODEL_CONTEXT_K));
    } else if (wantSummary) {
      const min = Math.min(MIN_SCORE, 0.45);
      modelContext = retrievedAll.filter((c) => c.score >= min).slice(0, 1);
      if (modelContext.length === 0) modelContext = retrievedAll.slice(0, 1);
    } else {
      const strict = retrievedAll.filter((c) => c.score >= MIN_SCORE);

      const forced = quotedTitle
        ? pickByQuotedTitle(
            retrievedAll,
            quotedTitle,
            Math.max(1, Math.min(2, MODEL_CONTEXT_K))
          )
        : [];

      const forcedKeys = new Set(
        forced.map((c) => `${c.talk_id}::${c.chunk.slice(0, 60)}`)
      );

      const rest = strict
        .filter((c) => !forcedKeys.has(`${c.talk_id}::${c.chunk.slice(0, 60)}`))
        .slice(0, Math.max(0, MODEL_CONTEXT_K - forced.length));

      modelContext = [...forced, ...rest];

      if (modelContext.length === 0) {
        modelContext = retrievedAll
          .filter((c) => c.score >= FALLBACK_MIN_SCORE)
          .slice(0, MODEL_CONTEXT_K);
      }
    }

    // 5) Prompts (must)
    const systemPrompt =
      `You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). ` +
      `You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. ` +
      `If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data." ` +
      `Always ground your answer in the given context, quoting exact phrases when helpful.`;

    const contextBlock = wantListing3
      ? buildTitleOnlyContextBlock(modelContext)
      : buildTranscriptContextBlock(modelContext, question);

    let extraInstr = "";
    if (wantListing3) {
      extraInstr =
        `- Return EXACTLY 3 distinct TED talk titles.\n` +
        `- Use only titles that appear in the CONTEXT.\n` +
        `- Output format: a numbered list of 3 lines (1..3).\n`;
    } else if (wantRec) {
      extraInstr =
        `- Recommend ONE TED talk from the CONTEXT.\n` +
        `- Justify with evidence-based quotes/paraphrases from the transcript.\n`;
    } else if (wantSummary) {
      extraInstr =
        `- Identify the relevant talk in the CONTEXT.\n` +
        `- Provide the talk title and a concise summary of its main idea.\n` +
        `- Ground the summary in transcript evidence.\n`;
    } else {
      extraInstr =
        `- If the context contains the answer, you MUST answer.\n` +
        `- Prefer the most direct sentence(s) that answer the question.\n` +
        `- Cite the exact phrase(s) that support your answer.\n`;
    }

    const userPrompt =
      `QUESTION:\n${question}\n\n` +
      `CONTEXT:\n${contextBlock}\n\n` +
      `INSTRUCTIONS:\n` +
      `- Use only the context above.\n` +
      extraInstr +
      `- Otherwise say: "I don't know based on the provided TED data."`;

    // 6) If no context, don't call LLM
    if (modelContext.length === 0) {
      return NextResponse.json({
        response: "I don't know based on the provided TED data.",
        context: [],
        Augmented_prompt: { System: systemPrompt, User: userPrompt },
      });
    }

    // 7) Chat
    const response = await callChat(systemPrompt, userPrompt);

    // ✅ Assignment output format: context must contain ONLY talk_id/title/chunk/score
    const responseContext = modelContext.map(({ talk_id, title, chunk, score }) => ({
      talk_id,
      title,
      chunk,
      score,
    }));

    return NextResponse.json({
      response,
      context: responseContext,
      Augmented_prompt: { System: systemPrompt, User: userPrompt },
    });
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Unknown error" },
      { status: 500 }
    );
  }
}
