import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    chunk_size: 1024,
    overlap_ratio: 0.2,
    top_k: 5
  });
}
