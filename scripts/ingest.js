/**
 * scripts/ingest.js
 *
 * Reads TED CSV, chunks transcripts, embeds each chunk, and upserts to Pinecone.
 * Loads env vars from .env.local explicitly (important for node scripts).
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";
import dotenv from "dotenv";

// --- Load .env.local for Node scripts (Next.js does this automatically; node scripts don't) ---
dotenv.config({ path: path.join(process.cwd(), ".env.local") });

// -------------------------
// Config
// -------------------------
const CSV_PATH = path.join(process.cwd(), "data", "ted_talks_en.csv");

// Chunking params (match /api/stats)
const CHUNK_SIZE_TOKENS = 1024; // must be <= 2048
const OVERLAP_RATIO = 0.2;      // must be <= 0.3

// Cost controls
const DRY_RUN = process.env.DRY_RUN === "1";
const LIMIT_TALKS = Number(process.env.LIMIT_TALKS ?? "0"); // 0 = no limit
const START_AT_ROW = Number(process.env.START_AT_ROW ?? "1"); // 1 = first data row (after header)

const UPSERT_BATCH_SIZE = 100;

// Required embedding model
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? "RPRTHPB-text-embedding-3-small";
const CHARS_PER_TOKEN_APPROX = 4;

// -------------------------
// Env vars
// -------------------------
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const PINECONE_HOST = process.env.PINECONE_HOST;

const LLMOD_API_KEY = process.env.LLMOD_API_KEY;
const LLMOD_BASE_URL = process.env.LLMOD_BASE_URL; // e.g. https://api.llmod.ai/v1  (NOTE: should end with /v1)

// -------------------------
// Helpers
// -------------------------
function requireEnv(name, value) {
  if (!value) throw new Error(`Missing ${name}. Please set it in .env.local (or shell env).`);
}

function sha1(text) {
  return crypto.createHash("sha1").update(text).digest("hex");
}

// -------------------------
// CSV parser (handles quotes + newlines inside transcript)
// -------------------------
function parseCsvRows(csvText) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < csvText.length; i++) {
    const c = csvText[i];
    const next = csvText[i + 1];

    if (c === '"' && inQuotes && next === '"') {
      field += '"';
      i++;
      continue;
    }
    if (c === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (c === "," && !inQuotes) {
      row.push(field);
      field = "";
      continue;
    }
    if ((c === "\n" || c === "\r") && !inQuotes) {
      if (c === "\r" && next === "\n") i++;
      row.push(field);
      field = "";
      if (!(row.length === 1 && row[0] === "")) rows.push(row);
      row = [];
      continue;
    }
    field += c;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

// -------------------------
// Chunking
// -------------------------
function chunkTranscript(transcript) {
  const targetChars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_APPROX;
  const overlapChars = Math.floor(targetChars * OVERLAP_RATIO);

  const text = (transcript ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  if (!text) return [];

  const paragraphs = text.split("\n\n").map(p => p.trim()).filter(Boolean);

  const chunks = [];
  let current = "";

  for (const p of paragraphs) {
    if (p.length > targetChars) {
      if (current) {
        chunks.push(current.trim());
        current = "";
      }
      for (let i = 0; i < p.length; i += (targetChars - overlapChars)) {
        const piece = p.slice(i, i + targetChars);
        if (piece.trim()) chunks.push(piece.trim());
        if (i + targetChars >= p.length) break;
      }
      continue;
    }

    if ((current ? (current + "\n\n" + p) : p).length <= targetChars) {
      current = current ? (current + "\n\n" + p) : p;
    } else {
      if (current.trim()) chunks.push(current.trim());
      const carry = current.slice(Math.max(0, current.length - overlapChars));
      current = (carry ? carry + "\n\n" : "") + p;
    }
  }

  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

// -------------------------
// Embeddings (OpenAI-compatible)
// -------------------------
async function embedText(text) {
  if (DRY_RUN) return new Array(1536).fill(0);

  requireEnv("LLMOD_API_KEY", LLMOD_API_KEY);
  requireEnv("LLMOD_BASE_URL", LLMOD_BASE_URL);

  const url = `${LLMOD_BASE_URL.replace(/\/$/, "")}/embeddings`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LLMOD_API_KEY}`,
    },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });

  if (!res.ok) throw new Error(`Embeddings API error (${res.status}): ${await res.text()}`);

  const json = await res.json();
  const vec = json?.data?.[0]?.embedding;

  if (!Array.isArray(vec)) {
    throw new Error(`Unexpected embeddings response: ${JSON.stringify(json).slice(0, 500)}`);
  }
  if (vec.length !== 1536) throw new Error(`Embedding dim mismatch: expected 1536, got ${vec.length}`);

  return vec;
}

// -------------------------
// Pinecone upsert (REST)
// -------------------------
async function pineconeUpsert(vectors) {
  if (DRY_RUN) return;

  requireEnv("PINECONE_API_KEY", PINECONE_API_KEY);
  requireEnv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME);
  requireEnv("PINECONE_HOST", PINECONE_HOST);

  const url = `${PINECONE_HOST.replace(/\/$/, "")}/vectors/upsert`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Api-Key": PINECONE_API_KEY,
    },
    body: JSON.stringify({ vectors }),
  });

  if (!res.ok) throw new Error(`Pinecone upsert error (${res.status}): ${await res.text()}`);
  return await res.json();
}

// -------------------------
// Main
// -------------------------
async function main() {
  console.log("=== TED Ingestion Script ===");
  console.log("CSV:", CSV_PATH);
  console.log("chunk_size_tokens:", CHUNK_SIZE_TOKENS);
  console.log("overlap_ratio:", OVERLAP_RATIO);
  console.log("DRY_RUN:", DRY_RUN ? "YES" : "NO");
  if (LIMIT_TALKS) console.log("LIMIT_TALKS:", LIMIT_TALKS);

  // Show which index we are targeting (helps avoid “wrong index” confusion)
  console.log("PINECONE_INDEX_NAME:", PINECONE_INDEX_NAME);
  console.log("PINECONE_HOST:", PINECONE_HOST);

  if (!DRY_RUN) {
    requireEnv("PINECONE_API_KEY", PINECONE_API_KEY);
    requireEnv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME);
    requireEnv("PINECONE_HOST", PINECONE_HOST);
    requireEnv("LLMOD_API_KEY", LLMOD_API_KEY);
    requireEnv("LLMOD_BASE_URL", LLMOD_BASE_URL);
  }

  const csvText = fs.readFileSync(CSV_PATH, "utf8");
  const rows = parseCsvRows(csvText);
  if (rows.length < 2) throw new Error("CSV seems empty/invalid.");

  const header = rows[0];
  const colIndex = Object.fromEntries(header.map((h, i) => [h, i]));

  const requiredCols = ["talk_id", "title", "speaker_1", "url", "transcript"];
  for (const c of requiredCols) {
    if (!(c in colIndex)) throw new Error(`Missing required column "${c}" in CSV header.`);
  }

  console.log("CSV header OK. Rows:", rows.length - 1);

  let processedTalks = 0;
  let vectorBuffer = [];

  for (let r = START_AT_ROW; r < rows.length; r++) {
    const row = rows[r];
    if (!row || row.length === 0) continue;

    const talk_id = row[colIndex.talk_id];
    const title = row[colIndex.title];
    const speaker_1 = row[colIndex.speaker_1];
    const url = row[colIndex.url];
    const transcript = row[colIndex.transcript];

    if (!talk_id || !transcript) continue;

    const chunks = chunkTranscript(transcript);
    if (chunks.length === 0) continue;

    for (let i = 0; i < chunks.length; i++) {
      const chunk_text = chunks[i];

      const chunkHash = sha1(chunk_text).slice(0, 12);
      const id = `${talk_id}#${String(i).padStart(4, "0")}#${chunkHash}`;

      const embedding = await embedText(chunk_text);

      vectorBuffer.push({
        id,
        values: embedding,
        metadata: {
          talk_id: String(talk_id),
          title: title ?? "",
          speaker_1: speaker_1 ?? "",
          url: url ?? "",
          chunk_index: i,
          chunk_text, // IMPORTANT: /api/prompt needs this to answer
        },
      });

      if (vectorBuffer.length >= UPSERT_BATCH_SIZE) {
        console.log(`Upserting batch of ${vectorBuffer.length} vectors...`);
        await pineconeUpsert(vectorBuffer);
        vectorBuffer = [];
      }
    }

    processedTalks++;
    if (processedTalks % 10 === 0) console.log(`Processed talks: ${processedTalks} (row ${r}/${rows.length - 1})`);

    if (LIMIT_TALKS && processedTalks >= LIMIT_TALKS) {
      console.log(`Reached LIMIT_TALKS=${LIMIT_TALKS}. Stopping early.`);
      break;
    }
  }

  if (vectorBuffer.length > 0) {
    console.log(`Upserting final batch of ${vectorBuffer.length} vectors...`);
    await pineconeUpsert(vectorBuffer);
  }

  console.log("✅ Ingestion complete.");
}

main().catch((err) => {
  console.error("❌ Ingestion failed:");
  console.error(err?.stack || err?.message || err);
  process.exit(1);
});
