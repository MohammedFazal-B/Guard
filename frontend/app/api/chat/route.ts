export async function POST(request: Request) {
  try {
    const response = await fetch(`${process.env.GUARD_API_URL ?? "http://127.0.0.1:8000"}/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: await request.text(), cache: "no-store",
    });
    return new Response(await response.text(), { status: response.status, headers: { "Content-Type": "application/json" } });
  } catch {
    return Response.json({ detail: "The guard API is unavailable." }, { status: 503 });
  }
}
