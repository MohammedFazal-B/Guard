export async function GET() {
  try {
    const response = await fetch(`${process.env.GUARD_API_URL ?? "http://127.0.0.1:8000"}/health`, { cache: "no-store" });
    return new Response(await response.text(), { status: response.status, headers: { "Content-Type": "application/json" } });
  } catch {
    return Response.json({ status: "offline" }, { status: 503 });
  }
}
