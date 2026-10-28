import { timingSafeEqual } from "node:crypto";

const maxRequestBytes = 64 * 1024;

function authorized(request) {
  const expected = process.env.NETLIFY_JEV_BRIDGE_TOKEN;
  const supplied = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "") ?? "";
  if (!expected || !supplied) return false;
  const actualBytes = Buffer.from(supplied);
  const expectedBytes = Buffer.from(expected);
  return actualBytes.length === expectedBytes.length && timingSafeEqual(actualBytes, expectedBytes);
}

export default async function handler(request) {
  if (!authorized(request)) return Response.json({ error: "Unauthorized" }, { status: 401 });

  const raw = await request.text();
  if (Buffer.byteLength(raw) > maxRequestBytes) {
    return Response.json({ error: "Request too large" }, { status: 413 });
  }

  let payload;
  try {
    payload = JSON.parse(raw);
  } catch {
    return Response.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const { state, questions } = payload ?? {};
  if (!((typeof state === "string" || (state && typeof state === "object" && !Array.isArray(state))) &&
      questions && typeof questions === "object" && !Array.isArray(questions) &&
      Object.keys(questions).length > 0 && Object.keys(questions).length <= 16)) {
    return Response.json({ error: "Invalid decision request" }, { status: 400 });
  }

  try {
    // The SDK call follows the provided systemOne example; verify it against the installed SDK before production use.
    const { TypeSafeClient } = await import("@typesafe-ai/sdk");
    const result = await new TypeSafeClient().systemOne({ state, questions });
    if (!result?.answers || typeof result.answers !== "object") {
      return Response.json({ error: "Invalid decision response" }, { status: 502 });
    }
    return Response.json({ answers: result.answers, usage: result.usage ?? {} });
  } catch {
    // Do not expose SDK errors or decision content to callers.
    return Response.json({ error: "Decision service unavailable" }, { status: 502 });
  }
}

export const config = { path: "/api/jev", method: "POST" };
