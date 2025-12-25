// app/api/prompt/route.ts
import { NextResponse } from "next/server";

// -------------------------
// Required models (assignment)
// -------------------------
const EMBEDDING_MODEL =
  process.env.EMBEDDING_MODEL ?? "RPRTHPB-text-embedding-3-small";
const CHAT_MODEL = process.env.CHAT_MODEL ?? "RPRTHPB-gpt-5-mini";

// -------------------------
// Pinecone + Gateway (LLMod)
// -------------------------
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_HOST = process.env.PINECONE_HOST;

const LLMOD_API_KEY = process.env.LLMOD_API_KEY;
const LLMOD_BASE_URL = process.env.LLMOD_BASE_URL ?? "";

// -------------------------
// RAG hyperparameters (must match /api/stats elsewhere)
// -------------------------
const CHUNK_SIZE = 1024; // informational
const OVERLAP_RATIO = 0.2; // informational

// Default baseline
const TOP_K = 5;

// Assignment constraint: Pinecone topK must be <= 30
const MAX_TOPK = 30;

// Thresholds
const MIN_SCORE = Number(process.env.MIN_SCORE ?? "0.22"); // for QA/summary gating
const LISTING_MIN_SCORE = Number(process.env.LISTING_MIN_SCORE ?? "0.25"); // for lists / rec
const META_MIN_SCORE = Number(process.env.META_MIN_SCORE ?? "0.20"); // for deterministic metadata (speaker/title)

// Context controls
const MODEL_CONTEXT_K = Math.max(1, Number(process.env.MODEL_CONTEXT_K ?? "3"));
const MAX_CONTEXT_CHARS_PER_CHUNK = Number(
  process.env.MAX_CONTEXT_CHARS_PER_CHUNK ?? "1800"
);
const MAX_TOTAL_CONTEXT_CHARS = Number(
  process.env.MAX_TOTAL_CONTEXT_CHARS ?? "6500"
);

// Listing
const LISTING_TOPK = Math.min(MAX_TOPK, Number(process.env.LISTING_TOPK ?? "30"));

// Fallback retrieval
const FALLBACK_TOPK = Math.min(MAX_TOPK, Number(process.env.FALLBACK_TOPK ?? "30"));
const FALLBACK_MIN_SCORE = Number(process.env.FALLBACK_MIN_SCORE ?? "0.35");

// -------------------------
// Types
// -------------------------
type ContextItem = {
  talk_id: string;
  title: string;
  chunk: string;
  score: number;

  // internal-only for deterministic metadata answers
  speaker?: string;
  url?: string;
};

type PineconeMatch = {
  id?: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

type PineconeQueryResponse = {
  matches?: PineconeMatch[];
};

// -------------------------
// Helpers
// -------------------------
function mustEnv(name: string, v: string | undefined | null): string {
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

function isNonEmptyString(x: unknown): x is string {
  return typeof x === "string" && x.trim().length > 0;
}

function getString(md: Record<string, unknown>, key: string): string {
  const v = md[key];
  const s = typeof v === "string" ? v : String(v ?? "");
  const t = s.trim();
  if (!t || t === "null" || t === "undefined") return "";
  return t;
}

function extractChunk(md: Record<string, unknown>): string {
  const v = md["chunk_text"] ?? md["chunk"] ?? md["text"] ?? md["passage"] ?? "";
  return String(v ?? "");
}

// speaker can be stored under different keys depending on ingest
function extractSpeaker(md: Record<string, unknown>): string {
  const candidates = [
    getString(md, "speaker"),
    getString(md, "speaker_1"),
    getString(md, "speaker_name"),
    getString(md, "speaker1"),
  ].filter((s) => isNonEmptyString(s));
  return candidates[0] ?? "";
}

function extractUrl(md: Record<string, unknown>): string {
  const candidates = [
    getString(md, "url"),
    getString(md, "talk_url"),
    getString(md, "ted_url"),
  ].filter((s) => isNonEmptyString(s));
  return candidates[0] ?? "";
}

function normalizeText(s: string): string {
  return (s ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function normTitle(s: string): string {
  return (s ?? "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .trim();
}

function extractQuotedTitle(question: string): string | null {
  const q = question ?? "";
  const m1 = q.match(/'([^']{3,160})'/);
  if (m1?.[1]) return m1[1].trim();
  const m2 = q.match(/"([^"]{3,160})"/);
  if (m2?.[1]) return m2[1].trim();
  return null;
}

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
    "speaker",
    "title",
    "url",
    "link",
    "talk_id",
    "chunk_index",
    "metadata",
  ]);

  return (question ?? "")
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length >= 4 && !STOP.has(w))
    .reverse()
    .slice(0, 12);
}

function smartSnippet(text: string, question: string, maxChars: number): string {
  const t = normalizeText(text);
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

function dedupeByTalk(items: ContextItem[]): ContextItem[] {
  const best = new Map<string, ContextItem>();
  for (const it of items) {
    const prev = best.get(it.talk_id);
    if (!prev || it.score > prev.score) best.set(it.talk_id, it);
  }
  return Array.from(best.values());
}

function normalizeRetrieved(pc: PineconeQueryResponse): ContextItem[] {
  return (pc.matches ?? [])
    .map((m: PineconeMatch): ContextItem => {
      const md = (m.metadata ?? {}) as Record<string, unknown>;
      const score = typeof m.score === "number" ? m.score : 0;

      return {
        talk_id: getString(md, "talk_id"),
        title: getString(md, "title"),
        chunk: extractChunk(md),
        speaker: extractSpeaker(md),
        url: extractUrl(md),
        score,
      };
    })
    .filter(
      (c) =>
        isNonEmptyString(c.talk_id) &&
        isNonEmptyString(c.title) &&
        isNonEmptyString(c.chunk)
    )
    .sort((a, b) => b.score - a.score);
}

function buildContextBlock(modelContext: ContextItem[], question: string): string {
  let total = 0;
  const parts: string[] = [];

  for (let i = 0; i < modelContext.length; i++) {
    const c = modelContext[i];
    const snippet = smartSnippet(c.chunk, question, MAX_CONTEXT_CHARS_PER_CHUNK);

    const part =
      `[#${i + 1}] talk_id="${c.talk_id}" title="${c.title}" score=${c.score}\n` +
      `${snippet}`;

    if (total + part.length > MAX_TOTAL_CONTEXT_CHARS) break;

    parts.push(part);
    total += part.length + 8;
  }

  return parts.length ? parts.join("\n\n---\n\n") : "(No context retrieved)";
}

function formatNumberedTitles(titles: string[]): string {
  return titles.map((t, i) => `${i + 1}) ${t}`).join("\n");
}

function answerUnknown(systemPrompt: string, userPrompt: string, ctx: any[] = []) {
  return NextResponse.json({
    response: "I don't know based on the provided TED data.",
    context: ctx,
    Augmented_prompt: { System: systemPrompt, User: userPrompt },
  });
}

// -------------------------
// Question classification
// -------------------------
type ListingSpec = { count: 1 | 2 | 3; mode: "exactly" | "up_to" } | null;

function wordToNum(w: string): 1 | 2 | 3 | null {
  if (w === "1" || w === "one") return 1;
  if (w === "2" || w === "two") return 2;
  if (w === "3" || w === "three") return 3;
  return null;
}

function getListingSpec(q: string): ListingSpec {
  const s = (q ?? "").toLowerCase().replace(/\s+/g, " ").trim();

  const hasTitles = /\btitles?\b/.test(s) && /\btalk\b/.test(s);
  const hasListingVerb =
    /\b(list|return|give|show|provide)\b/.test(s) || /\b(a list of)\b/.test(s);
  const hasSmallNumber = /\b(1|2|3|one|two|three)\b/.test(s);

  if (!(hasTitles && hasListingVerb && hasSmallNumber)) return null;

  const mAny = s.match(/\b(1|2|3|one|two|three)\b/);
  const n = mAny?.[1] ? wordToNum(mAny[1]) : null;
  if (!n) return null;

  if (/\bup\s+to\b/.test(s)) return { count: n, mode: "up_to" };

  const mExactly = s.match(/\bexactly\s+(1|2|3|one|two|three)\b/);
  if (mExactly?.[1]) {
    const count = wordToNum(mExactly[1]);
    return count ? { count, mode: "exactly" } : null;
  }

  return { count: n, mode: "exactly" };
}

// Recommendation intent (type 4)
function isRecommendationQ(q: string): boolean {
  const s = (q ?? "").toLowerCase();
  return (
    s.includes("recommend") ||
    s.includes("which talk should") ||
    s.includes("what talk should") ||
    s.includes("suggest a talk") ||
    s.includes("what should i watch") ||
    s.includes("i’m looking for a ted talk") ||
    s.includes("im looking for a ted talk")
  );
}

function isKeyIdeaSummaryQ(q: string): boolean {
  const s = (q ?? "").toLowerCase();
  return (
    s.includes("main idea") ||
    s.includes("key idea") ||
    s.includes("summarize") ||
    s.includes("summary") ||
    s.includes("what is this talk about")
  );
}

// Speaker question detection (type: metadata)
function isSpeakerQ(q: string): boolean {
  const s = (q ?? "").toLowerCase();
  return s.includes("who is the speaker") || s.includes("speaker of");
}

/**
 * ✅ NEW: Precise Fact Retrieval (Type 1)
 * Detect: "Find a TED talk..." + "Provide the title and speaker"
 * We should answer deterministically from metadata: title + speaker.
 */
function isFactRetrievalTitleSpeakerQ(q: string): boolean {
  const s = (q ?? "").toLowerCase().replace(/\s+/g, " ").trim();

  const wantsFind =
    s.includes("find a ted talk") ||
    s.includes("find a talk") ||
    s.startsWith("find ") ||
    s.includes("locate a ted talk");

  const wantsTitle = /\bprovide\b/.test(s) && /\btitle\b/.test(s);
  const wantsSpeaker = /\bspeaker\b/.test(s);

  // IMPORTANT: do not confuse with recommendation
  const notRec = !isRecommendationQ(s);

  return wantsFind && wantsTitle && wantsSpeaker && notRec;
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
  const body: any = {
    vector: vec,
    topK: Math.min(MAX_TOPK, Math.max(1, topK)),
    includeMetadata: true,
  };
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
// Route: POST /api/prompt
// -------------------------
export async function POST(req: Request) {
  // ✅ "Truth stamp" to verify this file is running
  console.log("🔥 RUNNING app/api/prompt/route.ts");

  try {
    const body: unknown = await req.json();
    const question = (body as any)?.question;

    if (!isNonEmptyString(question)) {
      return NextResponse.json({ error: "Missing question" }, { status: 400 });
    }

    const systemPrompt =
      `You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). ` +
      `You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. ` +
      `If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data." ` +
      `Always ground your answer in the given context, quoting exact phrases when helpful.`;

    // ---------- intent ----------
    const listingSpec = getListingSpec(question);
    const wantListing = listingSpec !== null;

    // Type 4
    const wantRec = isRecommendationQ(question);

    // Type (summary)
    const wantSummary = isKeyIdeaSummaryQ(question);

    // Metadata: speaker-of specific talk
    const wantSpeakerOnly = isSpeakerQ(question);

    // ✅ Type 1: Find talk + provide title and speaker (deterministic metadata answer)
    const wantFactTitleSpeaker = isFactRetrievalTitleSpeakerQ(question);

    const quotedTitle = extractQuotedTitle(question);
    const titleFilter =
      quotedTitle && quotedTitle.length >= 3
        ? { title: { $eq: quotedTitle } }
        : undefined;

    // ---------- retrieval ----------
    const qVec = await embedQuery(question);

    // First pass: if listing -> high topK; else moderate
    const topK1 = wantListing
      ? LISTING_TOPK
      : Math.min(MAX_TOPK, Math.max(12, TOP_K * 3));

    const pc1 = await pineconeQuery(qVec, topK1, titleFilter);
    const retrieved1 = normalizeRetrieved(pc1);

    // Decide fallback
    const enoughForListing = wantListing
      ? dedupeByTalk(retrieved1.filter((c) => c.score >= LISTING_MIN_SCORE)).length >=
        listingSpec!.count
      : true;

    const bestScore1 = retrieved1[0]?.score ?? 0;

    const needsFallback =
      retrieved1.length === 0 ||
      (wantListing && !enoughForListing) ||
      (!wantListing &&
        !wantSpeakerOnly &&
        !wantFactTitleSpeaker &&
        !wantRec &&
        !wantSummary &&
        bestScore1 < MIN_SCORE);

    let retrievedAll = retrieved1;

    if (needsFallback) {
      // Second pass: remove title filter, bigger topK (still capped at 30)
      const pc2 = await pineconeQuery(qVec, FALLBACK_TOPK);
      const retrieved2 = normalizeRetrieved(pc2);

      // Merge (dedupe by talk_id + chunk prefix)
      const seen = new Set<string>();
      const merged: ContextItem[] = [];
      for (const r of [...retrieved1, ...retrieved2]) {
        const key = `${r.talk_id}::${r.title}::${r.chunk.slice(0, 80)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(r);
      }
      merged.sort((a, b) => b.score - a.score);
      retrievedAll = merged;
    }

    // ---------- build modelContext ----------
    let modelContext: ContextItem[] = [];

    if (wantListing) {
      const filtered = retrievedAll.filter((c) => c.score >= LISTING_MIN_SCORE);
      const deduped = dedupeByTalk(filtered).sort((a, b) => b.score - a.score);
      modelContext = deduped.slice(0, listingSpec!.count);
    } else if (wantFactTitleSpeaker || wantSpeakerOnly) {
      // Deterministic metadata answers: pick ONE best talk (dedupe so we don't mix)
      const filtered = retrievedAll.filter((c) => c.score >= META_MIN_SCORE);
      const deduped = dedupeByTalk(filtered).sort((a, b) => b.score - a.score);

      if (quotedTitle) {
        const qt = normTitle(quotedTitle);
        const locked = deduped.filter((c) => normTitle(c.title) === qt);
        modelContext = (locked.length ? locked : deduped).slice(0, 1);
      } else {
        modelContext = deduped.slice(0, 1);
      }
    } else if (wantRec) {
      const filtered = retrievedAll.filter((c) => c.score >= LISTING_MIN_SCORE);
      const deduped = dedupeByTalk(filtered).sort((a, b) => b.score - a.score);
      modelContext = deduped.slice(0, Math.max(3, MODEL_CONTEXT_K));
    } else if (wantSummary) {
      modelContext = retrievedAll.slice(0, 1);
    } else {
      modelContext = retrievedAll.slice(0, MODEL_CONTEXT_K);
    }

    const responseContext = modelContext.map(({ talk_id, title, chunk, score }) => ({
      talk_id,
      title,
      chunk,
      score,
    }));

    // ---------- no context ----------
    if (modelContext.length === 0) {
      return answerUnknown(systemPrompt, question, []);
    }

    // ---------- gating ----------
    const bestScore = modelContext[0]?.score ?? 0;
    if (!wantListing && !wantSpeakerOnly && !wantFactTitleSpeaker && bestScore < MIN_SCORE) {
      return answerUnknown(systemPrompt, question, []);
    }

    // -------------------------
    // 1) LISTING (deterministic)
    // -------------------------
    if (wantListing) {
      const titles = modelContext.map((c) => c.title);
      const n = listingSpec!.count;

      if (listingSpec!.mode === "exactly") {
        if (titles.length < n) return answerUnknown(systemPrompt, question, responseContext);
        return NextResponse.json({
          response: formatNumberedTitles(titles.slice(0, n)),
          context: responseContext,
          Augmented_prompt: { System: systemPrompt, User: question },
        });
      }

      if (titles.length === 0) return answerUnknown(systemPrompt, question, responseContext);

      return NextResponse.json({
        response: formatNumberedTitles(titles.slice(0, Math.min(n, titles.length))),
        context: responseContext,
        Augmented_prompt: { System: systemPrompt, User: question },
      });
    }

    // -------------------------
    // 2) SPEAKER ONLY (deterministic)
    // -------------------------
    if (wantSpeakerOnly) {
      const best = modelContext[0];
      const speaker = (best.speaker ?? "").trim();

      if (!isNonEmptyString(speaker)) {
        return answerUnknown(systemPrompt, question, responseContext);
      }

      return NextResponse.json({
        response: speaker,
        context: responseContext,
        Augmented_prompt: { System: systemPrompt, User: question },
      });
    }

    // -------------------------
    // ✅ 3) TYPE 1: Fact retrieval (Title + Speaker) deterministic
    // -------------------------
    if (wantFactTitleSpeaker) {
      const best = modelContext[0];
      const title = (best.title ?? "").trim();
      const speaker = (best.speaker ?? "").trim();

      // For type-1: must provide both; if missing -> refuse
      if (!isNonEmptyString(title) || !isNonEmptyString(speaker)) {
        return answerUnknown(systemPrompt, question, responseContext);
      }

      return NextResponse.json({
        response: `${title} — ${speaker}`,
        context: responseContext,
        Augmented_prompt: { System: systemPrompt, User: question },
      });
    }

    // -------------------------
    // 4) GPT RAG (summary / recommendation / QA)
    // -------------------------
    const contextBlock = buildContextBlock(modelContext, question);

    let extraInstr = "";
    if (wantRec) {
      extraInstr =
        `- Recommend ONE TED talk from the CONTEXT.\n` +
        `- Justify with evidence-based quotes/paraphrases from the transcript.\n` +
        `- Output must include the chosen title.\n`;
    } else if (wantSummary) {
      extraInstr =
        `- Provide the talk title and a concise summary of its main idea.\n` +
        `- Ground the summary in transcript evidence.\n`;
    } else {
      extraInstr =
        `- If the context contains the answer, you MUST answer.\n` +
        `- Prefer the most direct sentence(s) that answer the question.\n`;
    }

    const userPrompt =
      `QUESTION:\n${question}\n\n` +
      `CONTEXT:\n${contextBlock}\n\n` +
      `INSTRUCTIONS:\n` +
      `- Use only the context above.\n` +
      extraInstr +
      `- Otherwise say: "I don't know based on the provided TED data."`;

    const response = await callChat(systemPrompt, userPrompt);

    return NextResponse.json({
      response,
      context: responseContext,
      Augmented_prompt: { System: systemPrompt, User: userPrompt },
    });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Unknown error" }, { status: 500 });
  }
}
