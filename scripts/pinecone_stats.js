// scripts/pinecone_stats.js
// Reads Pinecone index stats (read-only) to confirm vectors exist.
// IMPORTANT: explicitly loads .env.local (dotenv default is .env, not .env.local).

import dotenv from "dotenv";

dotenv.config({ path: ".env.local" });

const { PINECONE_API_KEY, PINECONE_HOST } = process.env;

if (!PINECONE_API_KEY || !PINECONE_HOST) {
  console.error("Missing PINECONE_API_KEY or PINECONE_HOST in .env.local");
  console.error("Loaded values:", {
    PINECONE_API_KEY: PINECONE_API_KEY ? "SET" : "MISSING",
    PINECONE_HOST: PINECONE_HOST ? "SET" : "MISSING",
  });
  process.exit(1);
}

async function main() {
  const url = `${PINECONE_HOST.replace(/\/$/, "")}/describe_index_stats`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Api-Key": PINECONE_API_KEY,
    },
    body: JSON.stringify({}),
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Pinecone stats error (${res.status}): ${t}`);
  }

  const json = await res.json();
  console.log(JSON.stringify(json, null, 2));
}

main().catch((e) => {
  console.error(e?.stack || e?.message || e);
  process.exit(1);
});
