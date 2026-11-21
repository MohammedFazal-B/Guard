"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { Menu, MessageSquare, Plus, Send, ShieldAlert, ShieldCheck, LoaderCircle, PanelLeftClose } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";

type Item = { role: "user" | "assistant"; content: string; blocked?: boolean; probability?: number; handledBy?: "guard" | "gemma" };
type SavedChat = { id: string; title: string };
type ChatReply = { conversation_id: string; reply: string; blocked: boolean; injection_probability: number; handled_by: "guard" | "gemma" };
type Health = { status: "ready" | "training" | "offline"; training?: { percent: number } | null };

const CHAT_KEY = "llm-guard-conversations";

export default function Home() {
  const [items, setItems] = useState<Item[]>([]);
  const [chats, setChats] = useState<SavedChat[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health>({ status: "offline" });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let saved: SavedChat[] = [];
    try { saved = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]") as SavedChat[]; } catch { /* invalid local state */ }
    const active = localStorage.getItem("llm-guard-conversation");
    if (active && !saved.some((chat) => chat.id === active)) saved.unshift({ id: active, title: "Previous chat" });
    queueMicrotask(() => setChats(saved));
    if (!active) return;
    fetch(`/api/conversations/${encodeURIComponent(active)}`)
      .then((response) => response.json())
      .then((data) => {
        if (!Array.isArray(data.messages)) return;
        setConversationId(active);
        setItems(data.messages.map((message: { role: "user" | "assistant"; content: string; blocked: number; probability: number | null; handled_by: "guard" | "gemma" | null }) => ({
          role: message.role, content: message.content, blocked: Boolean(message.blocked), probability: message.probability ?? undefined,
          handledBy: message.handled_by ?? undefined,
        })));
      }).catch(() => {});
  }, []);

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [items, busy]);
  useEffect(() => {
    const check = () => fetch("/api/health").then((response) => response.json()).then(setHealth).catch(() => setHealth({ status: "offline" }));
    check();
    const timer = setInterval(check, 15000);
    return () => clearInterval(timer);
  }, []);

  async function send(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || busy || health.status !== "ready") return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, conversation_id: conversationId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The guard service is unavailable.");
      const result = data as ChatReply;
      localStorage.setItem("llm-guard-conversation", result.conversation_id);
      if (!conversationId) {
        const updated = [{ id: result.conversation_id, title: message.slice(0, 42) }, ...chats];
        setChats(updated);
        localStorage.setItem(CHAT_KEY, JSON.stringify(updated));
      }
      setConversationId(result.conversation_id);
      setItems((current) => [...current,
        { role: "user", content: message, blocked: result.blocked, probability: result.injection_probability },
        { role: "assistant", content: result.reply, blocked: result.blocked, handledBy: result.handled_by },
      ]);
      setInput("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The guard service is unavailable.");
    } finally { setBusy(false); }
  }

  function newChat() {
    localStorage.removeItem("llm-guard-conversation");
    setConversationId(null); setItems([]); setInput(""); setError(""); setSidebarOpen(false);
  }

  async function openChat(id: string) {
    setSidebarOpen(false);
    if (id === conversationId) return;
    try {
      const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error("Could not load this chat.");
      const data = await response.json();
      setItems(data.messages.map((message: { role: "user" | "assistant"; content: string; blocked: number; probability: number | null; handled_by: "guard" | "gemma" | null }) => ({
        role: message.role, content: message.content, blocked: Boolean(message.blocked), probability: message.probability ?? undefined,
        handledBy: message.handled_by ?? undefined,
      })));
      setConversationId(id);
      localStorage.setItem("llm-guard-conversation", id);
      setError("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not load this chat."); }
  }

  const status = health.status === "ready" ? "Guard ready" : health.status === "training"
    ? `Training ${health.training?.percent ?? 0}%` : "Guard offline";
  const composer = <div className="w-full">
    {error && <Alert variant="destructive" className="mb-3 border-0"><ShieldAlert /><AlertDescription>{error}</AlertDescription></Alert>}
    <form onSubmit={send} className="flex w-full items-end gap-3 rounded-[28px] bg-secondary px-4 py-3 shadow-sm">
      <Textarea
        value={input} onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }}
        placeholder={health.status === "ready" ? "Ask anything" : "Chat opens when training finishes"}
        rows={1} disabled={busy || health.status !== "ready"} aria-label="Chat message"
        className="max-h-40 min-h-7 flex-1 resize-none border-0 bg-transparent px-1 py-1 shadow-none focus-visible:border-0 focus-visible:ring-0 disabled:bg-transparent dark:bg-transparent dark:disabled:bg-transparent"
      />
      <Button type="submit" size="icon" className="size-9 shrink-0 rounded-full" disabled={!input.trim() || busy || health.status !== "ready"} aria-label="Send message">
        {busy ? <LoaderCircle className="animate-spin" /> : <Send className="size-4" />}
      </Button>
    </form>
    <p className="mt-3 text-center text-xs text-muted-foreground">Protected by the local Kev guard. Decisions may be imperfect.</p>
  </div>;

  return <main className="flex h-dvh overflow-hidden bg-background text-foreground">
    {sidebarOpen && <button type="button" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)} className="fixed inset-0 z-20 bg-black/60 md:hidden" />}
    <aside className={`${sidebarOpen ? "flex" : "hidden"} fixed inset-y-0 left-0 z-30 w-72 flex-col bg-sidebar px-3 py-5 md:static md:flex md:w-64 md:shrink-0`}>
      <div className="mb-7 flex items-center justify-between px-2">
        <div className="flex items-center gap-2 text-lg font-semibold"><ShieldCheck className="size-5" /> LLM Guard</div>
        <Button variant="ghost" size="icon" className="rounded-full md:hidden" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><PanelLeftClose /></Button>
      </div>
      <Button variant="ghost" className="w-full justify-start gap-3 rounded-xl px-3 text-sm font-medium" onClick={newChat}><Plus className="size-4" /> New chat</Button>
      <div className="mt-9 min-h-0 flex-1 overflow-y-auto px-1">
        <p className="mb-3 px-2 text-xs font-medium text-muted-foreground">Recent chats</p>
        {chats.length === 0 ? <p className="px-2 text-sm text-muted-foreground">Your chats will appear here.</p> : chats.map((chat) =>
          <Button key={chat.id} variant="ghost" onClick={() => openChat(chat.id)} className={`mb-1 w-full justify-start gap-3 overflow-hidden rounded-xl px-2 font-normal ${conversationId === chat.id ? "bg-accent" : ""}`}>
            <MessageSquare className="size-4 shrink-0" /><span className="truncate">{chat.title}</span>
          </Button>)}
      </div>
      <div className="rounded-xl bg-muted/50 px-3 py-3 text-sm">
        <div className="flex items-center gap-2 font-medium"><ShieldCheck className="size-4" /> Local protection</div>
        <p className="mt-1 text-xs text-muted-foreground">Kev checks messages before Gemma 4 responds.</p>
      </div>
    </aside>

    <section className="flex min-w-0 flex-1 flex-col">
      <header className="flex h-16 shrink-0 items-center justify-between px-5 sm:px-8">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" className="rounded-full md:hidden" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Menu /></Button>
          <span className="text-base font-semibold">LLM Guard <span className="font-normal text-muted-foreground">/ Gemma 4</span></span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground"><span className={`size-2 rounded-full ${health.status === "ready" ? "bg-green-500" : "bg-amber-500"}`} />{status}</div>
      </header>

      {items.length === 0 ? <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-5 pb-16">
        <div className="w-full max-w-3xl">
          <h1 className="mb-9 text-center text-3xl font-semibold tracking-tight sm:text-4xl">Where should we begin?</h1>
          {composer}
          {health.status === "ready" && <div className="mt-5 flex justify-center"><Button variant="secondary" className="rounded-full" onClick={() => setInput("What can you help me with?")}>What can I do here?</Button></div>}
          {health.status === "training" && <p className="mt-5 text-center text-sm text-muted-foreground">The guard is training. Chat will unlock automatically when it is ready.</p>}
        </div>
      </div> : <>
        <div className="min-h-0 flex-1 overflow-y-auto px-5">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-7 pb-8 pt-8">
            {items.map((item, index) => <div key={index} className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`${item.role === "user" ? "max-w-[85%] rounded-3xl bg-muted px-5 py-3" : "w-full"} text-sm leading-7`}>
                {item.role === "assistant" && <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-muted-foreground">{item.blocked ? <ShieldAlert className="size-4" /> : <ShieldCheck className="size-4" />}{item.handledBy === "guard" || item.blocked ? "Guard" : "Gemma 4"}</div>}
                <div className="wrap-break-word [&_p]:mb-3 [&_p:last-child]:mb-0 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_pre]:overflow-x-auto [&_pre]:rounded-xl [&_pre]:bg-muted [&_pre]:p-3"><ReactMarkdown remarkPlugins={[remarkGfm]}>{item.content}</ReactMarkdown></div>
                {item.role === "user" && item.blocked && <p className="mt-2 text-xs text-destructive">Blocked · {Math.round((item.probability ?? 0) * 100)}% injection risk</p>}
              </div>
            </div>)}
            {busy && <div className="flex items-center gap-2 text-sm text-muted-foreground"><LoaderCircle className="size-4 animate-spin" /> Checking and responding…</div>}
            <div ref={bottom} />
          </div>
        </div>
        <div className="mx-auto w-full max-w-3xl px-5 pb-5">{composer}</div>
      </>}
    </section>
  </main>;
}
