// Chat Studio — conversations | centered transcript | collapsible settings.
// SSE streaming with 60ms-batched re-renders, markdown bodies, think/tool
// segmentation, message ratings, regenerate + edit-and-resubmit (both via
// server-side truncate), persisted sampler settings.

import { clsx } from 'clsx'
import {
  Brain, Check, ClipboardPaste, MessageSquare, Pencil, Plus, RotateCcw,
  Send, SlidersHorizontal, Sparkles, ThumbsDown, ThumbsUp,
  Trash2, Wrench, X,
} from 'lucide-react'
import {
  useEffect, useMemo, useRef, useState,
} from 'react'
import { useLocation } from 'react-router-dom'
import {
  CopyBtn, SegmentedBody, ThinkCard, ToolActivityCard, UserBubble,
  type ToolResultView,
} from '../components/chat/blocks'
import { Composer } from '../components/chat/Composer'
import { Markdown } from '../components/chat/Markdown'
import { ScrollFeed } from '../components/chat/ScrollFeed'
import {
  Btn, Card, Chip, Check as CheckBox, EmptyState, IconBtn,
  Modal, NumInput, SearchInput, Slider, Spinner, Tag, Textarea,
} from '../components/ui'
import { api, ssePost } from '../lib/api'
import { dayBucket, fmtClock, relTime } from '../lib/format'
import { parseStream, toolCallName } from '../lib/streamParse'
import { useForge } from '../lib/store'
import type { ChatMessage, ChatSummary, Conversation } from '../lib/types'

/* ---------- settings ---------- */

interface ChatSettings {
  temperature: number
  topP: number
  topK: number
  repPenalty: number
  maxTokens: number
  minP: number
  dryMult: number
  dryBase: number
  systemPrompt: string
  useMaster: boolean
  toolsEnabled: boolean
  thinking: boolean
  thinkBudget: number
  showThinking: boolean
  panelOpen: boolean
}

const DEFAULTS: ChatSettings = {
  temperature: 0.7, topP: 0.9, topK: 50, repPenalty: 1.05,
  maxTokens: 2048, minP: 0, dryMult: 0, dryBase: 1.75,
  systemPrompt: '', useMaster: true, toolsEnabled: true,
  thinking: true, thinkBudget: 160, showThinking: true, panelOpen: true,
}

const SETTINGS_KEY = 'forge.chat.settings.v2'

function loadSettings(): ChatSettings {
  try {
    const j = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')
    return { ...DEFAULTS, ...j }
  } catch { return { ...DEFAULTS } }
}

const PRESETS: { name: string; apply: Partial<ChatSettings> }[] = [
  { name: 'Precise', apply: { temperature: 0.2, topP: 0.9, topK: 40, repPenalty: 1.05, minP: 0, dryMult: 0 } },
  { name: 'Balanced', apply: { temperature: 0.7, topP: 0.9, topK: 50, repPenalty: 1.05, minP: 0, dryMult: 0 } },
  { name: 'Creative', apply: { temperature: 1.0, topP: 0.95, topK: 80, repPenalty: 1.02, minP: 0.02, dryMult: 0.8 } },
]

const PROMPT_TEMPLATES = [
  { label: 'Assistant', prompt: 'You are ForgeAI, an expert AI assistant specialized in machine learning and software engineering.' },
  { label: 'Architect', prompt: 'You are a principal software architect. Provide direct, production-grade solutions, complete files, and rigorous technical rationale without preamble.' },
  { label: 'Concise', prompt: 'Answer concisely in bullet points or code snippets. Avoid conversational filler or unnecessary explanations.' },
  { label: 'Researcher', prompt: 'Act as an AI research scientist. Analyze deep architectural tradeoffs, numerical stability, algorithmic complexity, and hardware constraints.' },
]

const STARTERS = [
  { title: 'Architecture Overview', desc: 'High-level structure & modules', prompt: 'Explain what this project does at a high level and detail its main components.' },
  { title: 'Workspace Files', desc: 'Directory tree & assets', prompt: 'List the files in the workspace and explain the directory layout.' },
  { title: 'Mamba-3 Research', desc: 'Web search & paper summary', prompt: 'Search the web for Mamba-3 architecture notes and summarize its key innovations.' },
  { title: 'System Diagnostics', desc: 'GPU & engine check', prompt: 'What time is it? Check system status and verify engine readiness.' },
]

interface StreamState {
  raw: string
  toolResults: ToolResultView[]
  tokPerSec: number
  pendingCalls: number
  gate?: { mode: string; p_easy: number }
}

/* ---------- page ---------- */

export default function Chat() {
  const engine = useForge((s) => s.engine)
  const location = useLocation()
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [search, setSearch] = useState('')
  const [conv, setConv] = useState<Conversation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [draft, setDraft] = useState('')
  const [stream, setStream] = useState<StreamState | null>(null)
  const [editing, setEditing] = useState<number | null>(null)
  const [editDraft, setEditDraft] = useState('')
  const [renaming, setRenaming] = useState(false)
  const [renameDraft, setRenameDraft] = useState('')
  const [importOpen, setImportOpen] = useState(false)
  const [importText, setImportText] = useState('')
  const [importErr, setImportErr] = useState('')

  const [cfg, setCfg] = useState<ChatSettings>(loadSettings)
  const set = <K extends keyof ChatSettings>(k: K, v: ChatSettings[K]) =>
    setCfg((c) => ({ ...c, [k]: v }))

  useEffect(() => {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(cfg)) }
    catch { /* quota */ }
  }, [cfg])

  const abortRef = useRef<AbortController | null>(null)
  const streamRef = useRef<StreamState | null>(null)
  const flushTimer = useRef<number | null>(null)

  /* batched stream render — tokens arrive per-token but we paint ≤ ~16/s */
  const pushStream = (immediate = false) => {
    const render = () => {
      flushTimer.current = null
      const s = streamRef.current
      setStream(s ? { ...s, toolResults: [...s.toolResults] } : null)
    }
    if (immediate) {
      if (flushTimer.current != null) window.clearTimeout(flushTimer.current)
      render()
      return
    }
    if (flushTimer.current == null) {
      flushTimer.current = window.setTimeout(render, 60)
    }
  }

  const loadList = () =>
    api.get<{ conversations: ChatSummary[] }>('/api/chats')
      .then((r) => setChats(
        [...r.conversations].sort((a, b) => b.updated_at - a.updated_at)))
      .catch(() => undefined)

  useEffect(() => { loadList() }, [])

  // deep link from other pages: /chat?open=<id>
  useEffect(() => {
    const open = new URLSearchParams(location.search).get('open')
    if (open) void loadConv(open)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.search])

  const loadConv = async (id: string) => {
    try {
      const c = await api.get<Conversation & { error?: string }>(`/api/chats/${id}`)
      if (c.error || !c.messages) return
      setConv(c)
      setEditing(null)
      setError('')
    } catch (e) { setError(String(e)) }
  }

  const newChat = async () => {
    const c = await api.post<Conversation>('/api/chats', { title: 'New chat' })
    setConv(c)
    setEditing(null)
    loadList()
  }

  const deleteChat = async (id: string) => {
    await api.del(`/api/chats/${id}`).catch(() => undefined)
    if (conv?.id === id) setConv(null)
    loadList()
  }

  const commitRename = async () => {
    setRenaming(false)
    const title = renameDraft.trim()
    if (!conv || !title || title === conv.title) return
    await api.post(`/api/chats/${conv.id}/rename`, { title }).catch(() => undefined)
    setConv({ ...conv, title })
    loadList()
  }

  const exportRated = async () => {
    try {
      const r = await api.post<{ path: string; examples: number }>('/api/chats/export')
      setNotice(`exported ${r.examples} examples → ${r.path}`)
    } catch (e) { setError(String(e)) }
  }

  const doImport = async () => {
    setImportErr('')
    try {
      const r = await api.post<Conversation & { error?: string }>(
        '/api/chats/import', { text: importText })
      if (r.error || !r.id) throw new Error(r.error ?? 'import failed')
      setImportOpen(false)
      setImportText('')
      loadList()
      void loadConv(r.id)
    } catch (e) { setImportErr(String(e)) }
  }

  const rate = async (idx: number, rating: 'good' | 'bad') => {
    if (!conv) return
    const r = await api.post<{ rating: 'good' | 'bad' | null }>(
      `/api/chats/${conv.id}/rate`, { msg_idx: idx, rating })
    setConv({
      ...conv,
      messages: conv.messages.map(
        (m, i) => (i === idx ? { ...m, rating: r.rating } : m)),
    })
    loadList()
  }

  /* ---------- send / stream ---------- */

  /** Core send — assumes guards passed. Holds abortRef for its duration. */
  const doSend = async (text: string) => {
    setError('')
    setNotice('')
    setDraft('')
    setEditing(null)
    setBusy(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl

    const userMsg: ChatMessage = {
      role: 'user', content: text, ts: Date.now() / 1000,
    }
    setConv((c) => (c
      ? { ...c, messages: [...c.messages, userMsg] }
      : { id: '', title: text.slice(0, 48), model: '', created_at: 0,
          updated_at: 0, messages: [userMsg] }))

    streamRef.current = { raw: '', toolResults: [], tokPerSec: 0, pendingCalls: 0 }
    pushStream(true)

    let t0 = 0
    let nTok = 0
    let convId = conv?.id ?? ''
    try {
      await ssePost('/api/chat/send', {
        conv_id: convId,
        message: text,
        system_prompt: cfg.systemPrompt || undefined,
        use_master: cfg.useMaster,
        model: '',
        max_new_tokens: cfg.maxTokens,
        temperature: cfg.temperature,
        top_p: cfg.topP,
        top_k: cfg.topK,
        repetition_penalty: cfg.repPenalty,
        min_p: cfg.minP,
        dry_multiplier: cfg.dryMult,
        dry_base: cfg.dryBase,
        tools_enabled: cfg.toolsEnabled,
        thinking: cfg.thinking,
        think_budget: cfg.thinkBudget,
      }, (evt) => {
        const s = streamRef.current
        if (!s) return
        const tps = nTok && t0 ? nTok / ((Date.now() - t0) / 1000) : 0
        if (evt.type === 'conv') {
          convId = (evt.data as { id: string }).id
        } else if (evt.type === 'token') {
          if (!t0) t0 = Date.now()
          nTok++
          s.raw += evt.data as string
          s.tokPerSec = tps
          s.pendingCalls = 0
          pushStream()
        } else if (evt.type === 'gate') {
          s.gate = evt.data as { mode: string; p_easy: number }
          pushStream(true)
        } else if (evt.type === 'tool_call') {
          s.pendingCalls = 1
          pushStream(true)
        } else if (evt.type === 'tool_result') {
          const d = evt.data as {
            name: string
            ok?: boolean
            elapsed_s?: number
            result: { text?: string }
          }
          s.toolResults.push({
            name: d.name,
            ok: d.ok,
            elapsed_s: d.elapsed_s,
            text: d.result?.text ?? JSON.stringify(d.result),
          })
          s.pendingCalls = 0
          pushStream(true)
        } else if (evt.type === 'saved') {
          convId = (evt.data as { id: string }).id
        } else if (evt.type === 'error') {
          setError(String(evt.data))
        }
      }, ctrl.signal)
    } catch (e) {
      if (!ctrl.signal.aborted) setError(String(e))
    } finally {
      if (flushTimer.current != null) {
        window.clearTimeout(flushTimer.current)
        flushTimer.current = null
      }
      setBusy(false)
      abortRef.current = null
      const finalRaw = streamRef.current?.raw ?? ''
      if (finalRaw) {
        setConv((c) => (c ? {
          ...c,
          messages: [...c.messages, {
            role: 'assistant', content: finalRaw, ts: Date.now() / 1000,
          }],
        } : c))
      }
      setStream(null)
      streamRef.current = null
      if (convId) loadConv(convId).catch(() => undefined)
      loadList()
    }
  }

  const send = (text: string) => {
    text = text.trim()
    if (!text || busy || abortRef.current) return
    if (engine.state !== 'ready') {
      setError('Engine not ready — load a model on the Engine or Models page.')
      return
    }
    void doSend(text)
  }

  /** Drop messages[idx:] server-side, then resend `text`. Backs both
   *  regenerate (idx = last user msg) and edit-and-resubmit. Holds the
   *  guard ref across the truncate POST so a fast Enter can't double-send. */
  const truncateAndSend = async (index: number, text: string) => {
    if (!conv?.id || busy || abortRef.current || engine.state !== 'ready') return
    setBusy(true)
    abortRef.current = new AbortController()
    try {
      const r = await api.post<Conversation & { error?: string }>(
        `/api/chats/${conv.id}/truncate`, { index })
      if (r.error || !r.messages) throw new Error(r.error ?? 'truncate failed')
      setConv(r)
      abortRef.current = null
      await doSend(text)
    } catch (e) {
      setError(String(e))
      abortRef.current = null
      setBusy(false)
    }
  }

  const lastUserIdx = useMemo(() => {
    if (!conv) return -1
    for (let i = conv.messages.length - 1; i >= 0; i--) {
      if (conv.messages[i].role === 'user') return i
    }
    return -1
  }, [conv])

  const regenerate = () => {
    if (lastUserIdx < 0 || !conv) return
    void truncateAndSend(lastUserIdx, conv.messages[lastUserIdx].content)
  }

  /* ---------- display items: pair tool results with their assistant msg -- */

  type Item =
    | { kind: 'user'; msg: ChatMessage; idx: number }
    | { kind: 'assistant'; msg: ChatMessage; idx: number; results: ChatMessage[] }
    | { kind: 'tool'; msg: ChatMessage; idx: number }

  const items = useMemo<Item[]>(() => {
    const out: Item[] = []
    const msgs = conv?.messages ?? []
    for (let i = 0; i < msgs.length; i++) {
      const m = msgs[i]
      if (m.role === 'assistant') {
        const at = i
        const results: ChatMessage[] = []
        while (i + 1 < msgs.length && msgs[i + 1].role === 'tool') {
          results.push(msgs[++i])
        }
        out.push({ kind: 'assistant', msg: m, idx: at, results })
      } else if (m.role === 'user') {
        out.push({ kind: 'user', msg: m, idx: i })
      } else if (m.role === 'tool') {
        out.push({ kind: 'tool', msg: m, idx: i })
      }
    }
    return out
  }, [conv])

  const grouped = useMemo(() => {
    const q = search.toLowerCase()
    const list = chats.filter((c) => c.title.toLowerCase().includes(q))
    const buckets: { label: string; rows: ChatSummary[] }[] = []
    for (const c of list) {
      const b = dayBucket(c.updated_at)
      const cur = buckets[buckets.length - 1]
      if (cur?.label === b) cur.rows.push(c)
      else buckets.push({ label: b, rows: [c] })
    }
    return buckets
  }, [chats, search])

  const ready = engine.state === 'ready'
  const lastAssistantItem = [...items].reverse().find((i) => i.kind === 'assistant')

  return (
    <div className="flex h-full min-h-0">
      {/* ---- conversations ---- */}
      <div className="w-[226px] shrink-0 border-r border-border flex flex-col bg-bg-alt/40">
        <div className="p-2.5 space-y-2">
          <div className="flex gap-1.5">
            <Btn variant="primary" className="flex-1 justify-center" onClick={newChat}>
              <Plus size={14} /> New chat
            </Btn>
            <IconBtn title="Paste a chat transcript as a new conversation"
              className="!h-auto border border-border"
              onClick={() => { setImportOpen(true); setImportErr('') }}>
              <ClipboardPaste size={13} />
            </IconBtn>
          </div>
          <SearchInput
            value={search}
            onChange={setSearch}
            placeholder="Search chats…"
            className="w-full"
          />
        </div>
        <div className="flex-1 overflow-y-auto px-1.5 pb-2">
          {grouped.map((g) => (
            <div key={g.label}>
              <div className="px-2 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-faint">
                {g.label}
              </div>
              {g.rows.map((c) => (
                <button
                  key={c.id}
                  onClick={() => loadConv(c.id)}
                  className={clsx(
                    'w-full text-left px-2.5 py-2 rounded-xl mb-1 group cursor-pointer transition-all duration-150',
                    conv?.id === c.id
                      ? 'bg-accent/15 text-text border border-accent/30 shadow-[0_0_12px_rgba(122,151,255,0.15)]'
                      : 'text-text-dim hover:bg-panel-alt/80 border border-transparent hover:text-text'
                  )}>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[12.5px] font-medium truncate flex-1">{c.title}</span>
                    <IconBtn
                      className="!w-5 !h-5 opacity-0 group-hover:opacity-100 hover:!text-err"
                      title="Delete conversation"
                      onClick={(e) => { e.stopPropagation(); void deleteChat(c.id) }}>
                      <Trash2 size={11} />
                    </IconBtn>
                  </div>
                  <div className="text-[10.5px] text-text-faint mt-1 flex items-center gap-2">
                    <span className="bg-panel px-1.5 py-0.2 rounded border border-border/40 font-mono text-[9.5px]">
                      {c.n_messages}m
                    </span>
                    <span>{relTime(c.updated_at)}</span>
                    {c.good > 0 && <span className="text-ok font-semibold">+{c.good}</span>}
                    {c.bad > 0 && <span className="text-err font-semibold">−{c.bad}</span>}
                  </div>
                </button>
              ))}
            </div>
          ))}
          {!grouped.length && (
            <div className="text-[11.5px] text-text-faint text-center py-8">
              no conversations found
            </div>
          )}
        </div>
      </div>

      {/* ---- transcript ---- */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* chat header */}
        <div className="flex items-center gap-2 px-5 h-12 border-b border-border/80 glass-panel shrink-0">
          {renaming && conv ? (
            <input
              autoFocus
              value={renameDraft}
              onChange={(e) => setRenameDraft(e.target.value)}
              onBlur={commitRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void commitRename()
                if (e.key === 'Escape') setRenaming(false)
              }}
              className="bg-input border border-accent/50 rounded-lg px-2.5 py-1 text-[13px] text-text focus:outline-none w-72" />
          ) : (
            <>
              <span className="text-[13.5px] font-semibold truncate max-w-[40%] text-text">
                {conv?.title ?? 'Chat Studio'}
              </span>
              {conv?.id && (
                <IconBtn title="Rename conversation" className="!w-6 !h-6"
                  onClick={() => { setRenaming(true); setRenameDraft(conv.title) }}>
                  <Pencil size={11} />
                </IconBtn>
              )}
            </>
          )}
          {conv && conv.messages.length > 0 && (
            <span className="text-[11px] text-text-faint bg-panel px-2 py-0.5 rounded-full border border-border/50 shrink-0 font-mono">
              {conv.messages.length} messages
            </span>
          )}

          {/* Quick Preset Selector Chips in Header */}
          <div className="hidden lg:flex items-center gap-1 ml-3 border-l border-border/60 pl-3">
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => setCfg((c) => ({ ...c, ...p.apply }))}
                className={clsx(
                  'px-2 py-0.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer',
                  cfg.temperature === p.apply.temperature && cfg.topP === p.apply.topP
                    ? 'bg-accent/20 text-accent-hi border border-accent/40'
                    : 'text-text-faint hover:text-text hover:bg-panel-alt'
                )}>
                {p.name}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-2">
            {ready && (
              <Tag kind="accent" className="font-mono !text-[10.5px]">
                {engine.info.checkpoint?.split(/[\\/]/).pop() ?? 'model'}
              </Tag>
            )}
            <IconBtn
              title={cfg.panelOpen ? 'Hide settings drawer' : 'Show settings drawer'}
              className={clsx(cfg.panelOpen && '!text-accent-hi bg-panel-alt border border-accent/30')}
              onClick={() => set('panelOpen', !cfg.panelOpen)}>
              <SlidersHorizontal size={13} />
            </IconBtn>
          </div>
        </div>

        <ScrollFeed watch={`${items.length}:${stream?.raw.length ?? 0}:${stream?.toolResults.length ?? 0}`}
          className="px-6 py-5">
          <div className="max-w-[800px] mx-auto space-y-5">
            {!items.length && !stream && (
              <EmptyState
                icon={<MessageSquare size={36} strokeWidth={1.5} />}
                title={ready ? 'Start a conversation' : 'Engine not loaded'}
                desc={ready
                  ? 'Chat directly with your model, inspect its internal thinking pass, or execute live tool harnesses.'
                  : 'Load a checkpoint on the Models page to initialize generation.'}
                action={ready ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-xl text-left mt-3">
                    {STARTERS.map((s) => (
                      <button
                        key={s.title}
                        onClick={() => void doSend(s.prompt)}
                        className="p-3 rounded-xl border border-border/80 bg-panel/60 hover:bg-panel-alt hover:border-accent/50 transition-all text-left group cursor-pointer shadow-sm">
                        <div className="text-[12.5px] font-semibold text-text group-hover:text-accent-hi transition-colors flex items-center justify-between">
                          {s.title}
                          <Sparkles size={12} className="text-text-faint group-hover:text-accent transition-colors" />
                        </div>
                        <div className="text-[11px] text-text-dim mt-0.5 leading-snug">
                          {s.desc}
                        </div>
                      </button>
                    ))}
                  </div>
                ) : undefined}
              />
            )}
            {items.map((it) => (
              <ChatItem
                key={it.idx}
                item={it}
                showThinking={cfg.showThinking}
                isLastAssistant={it === lastAssistantItem}
                busy={busy}
                editing={editing === it.idx}
                editDraft={editDraft}
                onEditDraft={setEditDraft}
                onStartEdit={(idx, text) => { setEditing(idx); setEditDraft(text) }}
                onCancelEdit={() => setEditing(null)}
                onSubmitEdit={(idx, text) => void truncateAndSend(idx, text)}
                onRate={(idx, r) => void rate(idx, r)}
                onRegenerate={regenerate} />
            ))}
            {stream && (
              <StreamRow stream={stream} showThinking={cfg.showThinking}
                thinking={cfg.thinking} />
            )}
            {error && (
              <div className="text-[12.5px] text-err bg-err/10 border border-err/30 rounded-lg px-3 py-2">
                {error}
              </div>
            )}
            {notice && (
              <div className="text-[12px] text-ok bg-ok/10 border border-ok/30 rounded-lg px-3 py-2 flex items-center gap-2">
                <Check size={13} /> {notice}
              </div>
            )}
          </div>
        </ScrollFeed>

        {/* composer */}
        <div className="px-6 pb-4 pt-1.5">
          <div className="max-w-[780px] mx-auto">
            <Composer
              value={draft}
              onChange={setDraft}
              onSubmit={() => void send(draft)}
              onStop={() => abortRef.current?.abort()}
              busy={busy}
              disabled={!ready}
              autoFocus
              placeholder={ready
                ? 'Message — Enter to send, Shift+Enter for newline'
                : 'Load a model first…'}
              chips={<>
                <Chip on={cfg.toolsEnabled} title="Let the model call read-only tools"
                  onClick={() => set('toolsEnabled', !cfg.toolsEnabled)}>
                  <Wrench size={11} /> Tools
                </Chip>
                <Chip on={cfg.thinking} title="Request a thinking pass before the reply"
                  onClick={() => set('thinking', !cfg.thinking)}>
                  <Brain size={11} /> Think
                </Chip>
                <Chip on={cfg.useMaster} title="Use the master system prompt when none is set"
                  onClick={() => set('useMaster', !cfg.useMaster)}>
                  <Sparkles size={11} /> Master
                </Chip>
              </>}
              meta={stream && stream.tokPerSec > 0 ? (
                <span className="text-[11px] text-text-faint tabular-nums">
                  {stream.tokPerSec.toFixed(1)} tok/s
                </span>
              ) : undefined}
            />
          </div>
        </div>
      </div>

      {/* ---- settings ---- */}
      {cfg.panelOpen && (
        <div className="w-[252px] shrink-0 border-l border-border overflow-y-auto p-3 space-y-3 bg-bg-alt/40">
          <Card title="Preset">
            <div className="flex gap-1.5">
              {PRESETS.map((p) => (
                <button key={p.name}
                  onClick={() => setCfg((c) => ({ ...c, ...p.apply }))}
                  className="flex-1 px-2 py-1.5 rounded-lg border border-border text-[11.5px] text-text-dim hover:text-text hover:border-accent/50 hover:bg-panel-alt transition-colors cursor-pointer">
                  {p.name}
                </button>
              ))}
            </div>
          </Card>
          <Card title="Sampling">
            <div className="space-y-2.5">
              <Slider label="Temperature" value={cfg.temperature} min={0} max={2} step={0.05}
                onChange={(v) => set('temperature', v)} fmt={(v) => v.toFixed(2)} />
              <Slider label="Top-p" value={cfg.topP} min={0} max={1} step={0.01}
                onChange={(v) => set('topP', v)} fmt={(v) => v.toFixed(2)} />
              <Slider label="Top-k" value={cfg.topK} min={0} max={200} step={1}
                onChange={(v) => set('topK', v)} />
              <Slider label="Min-p" value={cfg.minP} min={0} max={0.5} step={0.01}
                onChange={(v) => set('minP', v)} fmt={(v) => v.toFixed(2)}
                hint="Min-p sampling — 0 disables" />
              <Slider label="Rep. penalty" value={cfg.repPenalty} min={1} max={1.5} step={0.01}
                onChange={(v) => set('repPenalty', v)} fmt={(v) => v.toFixed(2)} />
              <Slider label="DRY penalty" value={cfg.dryMult} min={0} max={2} step={0.05}
                onChange={(v) => set('dryMult', v)} fmt={(v) => v.toFixed(2)}
                hint="DRY n-gram repetition penalty — 0 disables" />
              <NumInput label="Max tokens" value={cfg.maxTokens} min={1} max={8192} step={64}
                onChange={(v) => set('maxTokens', Math.round(v))} />
            </div>
          </Card>
          <Card title="System prompt">
            <div className="flex flex-wrap gap-1 mb-2">
              {PROMPT_TEMPLATES.map((t) => (
                <button
                  key={t.label}
                  onClick={() => set('systemPrompt', t.prompt)}
                  className="px-2 py-0.5 rounded text-[10.5px] bg-panel border border-border/80 hover:border-accent/40 text-text-dim hover:text-text cursor-pointer transition-colors">
                  {t.label}
                </button>
              ))}
            </div>
            <Textarea
              value={cfg.systemPrompt}
              onChange={(e) => set('systemPrompt', e.target.value)}
              placeholder="You are a helpful assistant…"
              rows={3} className="w-full !text-[12px]" />
            <div className="mt-2">
              <CheckBox label="Master prompt when empty" checked={cfg.useMaster}
                onChange={(v) => set('useMaster', v)} />
            </div>
          </Card>
          <Card title="Session">
            <div className="space-y-2">
              <CheckBox label="Request thinking" checked={cfg.thinking}
                onChange={(v) => set('thinking', v)} />
              <NumInput label="Think budget" value={cfg.thinkBudget} min={0} max={2048} step={16}
                onChange={(v) => set('thinkBudget', Math.round(v))} />
              <CheckBox label="Show thinking" checked={cfg.showThinking}
                onChange={(v) => set('showThinking', v)} />
              <CheckBox label="Tools enabled" checked={cfg.toolsEnabled}
                onChange={(v) => set('toolsEnabled', v)} />
            </div>
            <Btn variant="subtle" className="w-full justify-center mt-3"
              onClick={exportRated}>
              <Send size={12} className="rotate-90" /> Export rated data
            </Btn>
          </Card>
          <Card title="Model">
            <div className="text-[12px] text-text-dim font-mono break-all">
              {ready
                ? (engine.info.checkpoint?.split(/[\\/]/).pop() ?? '—')
                : 'none loaded'}
            </div>
          </Card>
        </div>
      )}

      {/* ---- import transcript modal ---- */}
      <Modal
        open={importOpen}
        onClose={() => setImportOpen(false)}
        title="Import Chat Transcript"
        maxWidth="max-w-xl">
        <div className="space-y-3">
          <Textarea
            autoFocus
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            rows={12}
            placeholder={
              'Paste a transcript — accepts:\n'
              + '  • "User:" / "Assistant:" markers (also **User:**, ChatGPT said:, ## Assistant)\n'
              + '  • ChatML  <|im_start|>user … <|im_end|>\n'
              + '  • JSON  [{"role": "user", "content": "…"}]\n'
              + '  • plain text — blank-line paragraphs alternate user/assistant'}
            className="w-full !text-[12px] font-mono resize-y" />
          {importErr && (
            <div className="px-3 py-2 text-[12px] text-err border border-err/30 bg-err/10 rounded-lg">
              {importErr}
            </div>
          )}
          <div className="flex items-center justify-between pt-2">
            <span className="text-[11px] text-text-faint">
              Import creates a new conversation you can steer and continue
            </span>
            <div className="flex gap-2">
              <Btn variant="ghost" size="sm" onClick={() => setImportOpen(false)}>Cancel</Btn>
              <Btn variant="primary" size="sm" disabled={!importText.trim()}
                onClick={() => void doImport()}>
                <ClipboardPaste size={13} /> Import Chat
              </Btn>
            </div>
          </div>
        </div>
      </Modal>
    </div>
  )
}

/* ---------- one transcript item ---------- */

function ChatItem({ item, showThinking, isLastAssistant, busy, editing,
  editDraft, onEditDraft, onStartEdit, onCancelEdit, onSubmitEdit,
  onRate, onRegenerate }: {
  item:
    | { kind: 'user'; msg: ChatMessage; idx: number }
    | { kind: 'assistant'; msg: ChatMessage; idx: number; results: ChatMessage[] }
    | { kind: 'tool'; msg: ChatMessage; idx: number }
  showThinking: boolean
  isLastAssistant: boolean
  busy: boolean
  editing: boolean
  editDraft: string
  onEditDraft: (v: string) => void
  onStartEdit: (idx: number, text: string) => void
  onCancelEdit: () => void
  onSubmitEdit: (idx: number, text: string) => void
  onRate: (idx: number, r: 'good' | 'bad') => void
  onRegenerate: () => void
}) {
  const { msg, idx } = item

  if (item.kind === 'tool') {
    // orphan tool message (not consumed by an assistant msg)
    return (
      <div className="max-w-[85%]">
        <ToolActivityCard
          name={msg.name ?? 'tool'} args=""
          result={{ text: msg.content, ok: true }} />
      </div>
    )
  }

  if (item.kind === 'user') {
    if (editing) {
      return (
        <div className="flex justify-end">
          <div className="w-full max-w-[75%] bg-panel border border-accent/40 rounded-xl p-2.5">
            <Textarea
              autoFocus
              value={editDraft}
              onChange={(e) => onEditDraft(e.target.value)}
              rows={3}
              className="w-full !text-[13px] resize-none" />
            <div className="flex justify-end gap-1.5 mt-2">
              <Btn variant="ghost" onClick={onCancelEdit}>
                <X size={12} /> Cancel
              </Btn>
              <Btn variant="primary" disabled={!editDraft.trim()}
                onClick={() => onSubmitEdit(idx, editDraft)}>
                <Send size={12} /> Save & resend
              </Btn>
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="group flex flex-col items-end">
        <UserBubble text={msg.content} />
        <div className="flex items-center gap-0.5 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <span className="text-[10px] text-text-faint mr-1">{fmtClock(msg.ts)}</span>
          <CopyBtn text={msg.content} />
          {!busy && (
            <IconBtn title="Edit & resend" className="!w-6 !h-6"
              onClick={() => onStartEdit(idx, msg.content)}>
              <Pencil size={11} />
            </IconBtn>
          )}
        </div>
      </div>
    )
  }

  // assistant
  const results: ToolResultView[] = item.results.map((r) => ({
    ok: true,
    text: r.content,
  }))
  return (
    <div className="group flex flex-col gap-1">
      <div className="max-w-[92%] min-w-0 space-y-2">
        {showThinking && msg.reasoning_content && (
          <ThinkCard text={msg.reasoning_content} streaming={false} />
        )}
        <SegmentedBody raw={msg.content} showThinking={showThinking}
          results={results} />
        {msg.tool_calls?.map((tc, i) => (
          <ToolActivityCard key={`tc${i}`} name={tc.name}
            args={JSON.stringify(tc.arguments ?? {}, null, 2)}
            result={results[i]} />
        ))}
      </div>
      <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="text-[10px] text-text-faint mr-1">{fmtClock(msg.ts)}</span>
        <CopyBtn text={msg.content} />
        <IconBtn title="Good response"
          className={clsx('!w-6 !h-6', msg.rating === 'good' && '!text-ok')}
          onClick={() => onRate(idx, 'good')}>
          <ThumbsUp size={11} />
        </IconBtn>
        <IconBtn title="Bad response"
          className={clsx('!w-6 !h-6', msg.rating === 'bad' && '!text-err')}
          onClick={() => onRate(idx, 'bad')}>
          <ThumbsDown size={11} />
        </IconBtn>
        {isLastAssistant && !busy && (
          <IconBtn title="Regenerate" className="!w-6 !h-6" onClick={onRegenerate}>
            <RotateCcw size={11} />
          </IconBtn>
        )}
      </div>
    </div>
  )
}

/* ---------- live stream row ---------- */

function StreamRow({ stream, showThinking, thinking }: {
  stream: StreamState
  showThinking: boolean
  thinking: boolean
}) {
  // thinking requests leave the prompt's <think> block open — raw stream
  // starts mid-reasoning, so mark the lead segment as think (implicit).
  // Exception: gate-routed direct mode closes the block server-side.
  const implicit = thinking && stream.gate?.mode !== 'direct'
  const segs = useMemo(
    () => parseStream(stream.raw, { implicitThink: implicit }),
    [stream.raw, implicit])
  let toolIdx = 0
  return (
    <div className="max-w-[92%] min-w-0 space-y-2">
      {stream.gate && (
        <div className="flex items-center gap-1.5 text-[10.5px] text-text-faint">
          <Brain size={10} />
          gate → {stream.gate.mode} (p={stream.gate.p_easy.toFixed(2)})
        </div>
      )}
      {segs.map((s, i) => {
        if (s.kind === 'think') {
          if (!showThinking) return null
          return <ThinkCard key={i} text={s.text} streaming={!s.closed} />
        }
        if (s.kind === 'tool_call') {
          const res = stream.toolResults[toolIdx++]
          return (
            <ToolActivityCard key={i}
              name={s.closed ? toolCallName(s.text) : 'calling tool…'}
              args={s.text} result={res} running={!s.closed} />
          )
        }
        return <Markdown key={i} text={s.text} />
      })}
      {!segs.length && (
        <div className="flex items-center gap-2 text-text-faint text-[12.5px]">
          <Spinner /> generating…
        </div>
      )}
      {stream.pendingCalls > 0 && (
        <div className="flex items-center gap-2 text-text-faint text-[12px]">
          <Spinner /> executing tool…
        </div>
      )}
    </div>
  )
}
