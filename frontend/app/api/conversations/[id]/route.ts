export async function GET(_request: Request, context: RouteContext<"/api/conversations/[id]">) {
  const { id } = await context.params;
  try {
    const response = await fetch(`${process.env.GUARD_API_URL ?? "http://127.0.0.1:8000"}/conversations/${encodeURIComponent(id)}`, { cache: "no-store" });
    return new Response(await response.text(), { status: response.status, headers: { "Content-Type": "application/json" } });
  } catch {
    return Response.json({ detail: "The guard API is unavailable." }, { status: 503 });
  }
}
