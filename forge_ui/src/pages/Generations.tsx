// Generations — decoding playground: stream / adaptive / batch / raw.
// All modes run against the one shared resident engine.

import { clsx } from 'clsx'
import {
  Check as CheckIcon, ChevronDown, ChevronRight, Copy, Download, Play,
  Sparkles, Square, Trash2,
} from 'lucide-react'
import type { ReactNode, RefObject } from 'react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ssePost } from '../lib/api'
import { useForge } from '../lib/store'
import {
  Btn, Card, Check as CheckBox, EmptyState, NumInput,
  Spinner, Tag, Textarea,
} from '../components/ui'

type Mode = 'stream' | 'adaptive' | 'batch' | 'raw'

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: 'stream', label: 'Stream', hint: 'token-by-token SSE' },
  { id: 'adaptive', label: 'Adaptive', hint: 'auto think / no-think' },
  { id: 'batch', label: 'Batch', hint: 'one prompt per line' },
  { id: 'raw', label: 'Raw', hint: 'full sampling controls' },
]

const fmtVal = (v: unknown): string => {
  if (v == null) return '—'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

export default function Generations() {
  const engine = useForge((s) => s.engine)
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>('stream')
  const ready = engine.state === 'ready'

  return (
    <div className="h-full min-h-0 flex flex-col">
      {/* mode tabs */}
      <div className="flex items-center gap-1.5 px-4 pt-3">
        {MODES.map((m) => (
          <button
            key={m.id} title={m.hint}
            onClick={() => setMode(m.id)}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-[12.5px] border transition-colors cursor-pointer',
              mode === m.id
                ? 'bg-accent/15 text-accent-hi border-accent/25'
                : 'text-text-dim border-transparent hover:bg-panel-alt hover:text-text')}>
            {m.label}
          </button>
        ))}
      </div>

      {!ready ? (
        <EmptyState
          icon={<Sparkles size={40} strokeWidth={1.2} />}
          title="No model loaded"
          desc="Generation uses the one shared engine — load a checkpoint on the Models page first."
          action={
            <Btn variant="primary" onClick={() => navigate('/models')}>
              Load model
            </Btn>
          } />
      ) : (
        <div className="flex-1 min-h-0">
          {mode === 'stream' && <StreamMode />}
          {mode === 'adaptive' && <AdaptiveMode />}
          {mode === 'batch' && <BatchMode />}
          {mode === 'raw' && <RawMode />}
        </div>
      )}
    </div>
  )
}

/* ---------- shared layout bits ---------- */

function GenLayout({ side, meta, contentRef, children }: {
  side: ReactNode
  meta?: ReactNode
  contentRef?: RefObject<HTMLDivElement | null>
  children: ReactNode
}) {
  return (
    <div className="h-full min-h-0 flex gap-4 p-4">
      <div className="w-[330px] shrink-0 overflow-y-auto space-y-3 pr-0.5">
        {side}
      </div>
      <div className="flex-1 min-w-0 bg-panel border border-border rounded-card shadow-card flex flex-col min-h-0">
        <div className="flex items-center justify-between px-4 pt-3 pb-1">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-text-faint">
            Output
          </div>
          <div className="flex items-center gap-2">{meta}</div>
        </div>
        <div ref={contentRef} className="flex-1 min-h-0 overflow-y-auto px-4 pb-4">
          {children}
        </div>
      </div>
    </div>
  )
}

function ParamsCard({ children }: { children: ReactNode }) {
  return (
    <Card title="Parameters">
      <div className="space-y-2">{children}</div>
    </Card>
  )
}

function Num({ label, value, onChange, min, max, step }: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
}) {
  return (
    <NumInput label={label} value={value} onChange={onChange}
      min={min} max={max} step={step} className="justify-between w-full" />
  )
}

function ErrLine({ msg }: { msg: string }) {
  if (!msg) return null
  return (
    <div className="text-[11.5px] text-err bg-err/10 border border-err/30 rounded-lg px-2.5 py-1.5 mt-2">
      {msg}
    </div>
  )
}

function JsonResult({ data }: { data: Record<string, unknown> }) {
  const [open, setOpen] = useState(false)
  const longText = Object.entries(data).filter(
    ([, v]) => typeof v === 'string' && v.length > 48)
  const scalars = Object.entries(data).filter(
    ([, v]) => !(typeof v === 'string' && v.length > 48))
  return (
    <div className="space-y-2.5">
      {scalars.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {scalars.map(([k, v]) => (
            <Tag key={k} kind={k === 'did_think' ? 'think' : 'idle'}>
              {k}: {fmtVal(v)}
            </Tag>
          ))}
        </div>
      )}
      {longText.map(([k, v]) => (
        <div key={k}>
          <div className="text-[10.5px] font-semibold uppercase tracking-wider text-text-faint mb-1">
            {k}
          </div>
          <div className="text-[13px] whitespace-pre-wrap leading-relaxed">
            {String(v)}
          </div>
        </div>
      ))}
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[11px] text-text-faint hover:text-text cursor-pointer">
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />} raw json
      </button>
      {open && (
        <pre className="text-[11.5px] font-mono text-text-dim bg-bg-alt border border-border rounded-lg p-3 whitespace-pre-wrap break-all max-h-72 overflow-y-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}

/* ---------- (a) stream ---------- */

function StreamMode() {
  const [prompt, setPrompt] = useState('')
  const [maxTok, setMaxTok] = useState(128)
  const [temp, setTemp] = useState(0.7)
  const [topK, setTopK] = useState(50)
  const [topP, setTopP] = useState(0.95)
  const [out, setOut] = useState('')
  const [nTok, setNTok] = useState(0)
  const [tokS, setTokS] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const abortRef = useRef<AbortController | null>(null)
  const outRef = useRef<HTMLDivElement>(null)

  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => {
    outRef.current?.scrollTo({ top: outRef.current.scrollHeight })
  }, [out])

  const run = async () => {
    const p = prompt.trim()
    if (!p || busy) return
    setErr(''); setOut(''); setNTok(0); setTokS(null); setBusy(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    let acc = ''
    try {
      await ssePost('/api/gen/stream', {
        prompt: p, max_new_tokens: maxTok,
        temperature: temp, top_k: topK, top_p: topP,
      }, (evt) => {
        if (evt.type === 'token') {
          acc += evt.data as string
          setOut(acc)
          setNTok((n) => n + 1)
        } else if (evt.type === 'done') {
          const d = evt.data as { tok_s?: number }
          if (typeof d.tok_s === 'number') setTokS(d.tok_s)
        } else if (evt.type === 'error') {
          setErr(String(evt.data))
        }
      }, ctrl.signal)
    } catch (e) {
      if (!ctrl.signal.aborted) setErr(String(e))
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  const [copied, setCopied] = useState(false)
  const copyOut = () => {
    if (!out) return
    void navigator.clipboard.writeText(out).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    })
  }

  const STREAM_SUGGESTIONS = [
    { label: 'Fibonacci', p: 'def fibonacci(n):\n    """Compute the n-th Fibonacci number efficiently."""' },
    { label: 'Mamba-3', p: 'Explain the core difference between Mamba-3 state-space duality and multi-head attention:' },
    { label: 'SQL Optimizer', p: 'Write an optimized PostgreSQL query with indexed CTEs to find monthly active users:' },
  ]

  return (
    <GenLayout
      contentRef={outRef}
      meta={<>
        {(out.length > 0 || nTok > 0) && (
          <span className="text-[10.5px] text-text-faint font-mono">
            {out.length} chars · {nTok} tok
          </span>
        )}
        {tokS != null && <Tag kind="accent">{tokS.toFixed(1)} tok/s</Tag>}
        {busy && <Tag kind="accent" pulse><Spinner /> streaming</Tag>}
        {out && !busy && (
          <div className="flex items-center gap-1 border-l border-border/60 pl-2">
            <button
              onClick={copyOut}
              title="Copy output"
              className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-panel-alt hover:bg-panel-hover border border-border text-text-dim hover:text-text cursor-pointer transition-colors">
              {copied ? <CheckIcon size={11} className="text-ok" /> : <Copy size={11} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
            <button
              onClick={() => { setOut(''); setNTok(0); setTokS(null) }}
              title="Clear output"
              className="p-1 rounded text-text-faint hover:text-err hover:bg-err/10 transition-colors cursor-pointer">
              <Trash2 size={12} />
            </button>
          </div>
        )}
      </>}
      side={<>
        <Card title="Prompt">
          <div className="flex flex-wrap gap-1 mb-2">
            {STREAM_SUGGESTIONS.map((s) => (
              <button
                key={s.label}
                onClick={() => setPrompt(s.p)}
                className="px-2 py-0.5 rounded text-[10.5px] bg-panel border border-border/80 hover:border-accent/40 text-text-dim hover:text-text cursor-pointer transition-colors">
                {s.label}
              </button>
            ))}
          </div>
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6} placeholder="Prompt text…"
            className="w-full !text-[13px]" />
          <div className="flex gap-2 mt-2.5">
            {busy ? (
              <Btn variant="danger" className="flex-1 justify-center"
                onClick={() => abortRef.current?.abort()}>
                <Square size={12} /> Stop
              </Btn>
            ) : (
              <Btn variant="primary" className="flex-1 justify-center"
                disabled={!prompt.trim()} onClick={run}>
                <Play size={13} /> Generate
              </Btn>
            )}
            {prompt && !busy && (
              <Btn variant="ghost" size="sm" onClick={() => setPrompt('')} title="Clear prompt">
                Clear
              </Btn>
            )}
          </div>
          <ErrLine msg={err} />
        </Card>
        <ParamsCard>
          <Num label="Max new tokens" value={maxTok} onChange={setMaxTok}
            min={1} max={8192} step={32} />
          <Num label="Temperature" value={temp} onChange={setTemp}
            min={0} max={2} step={0.05} />
          <Num label="Top-k" value={topK} onChange={setTopK}
            min={0} max={500} step={1} />
          <Num label="Top-p" value={topP} onChange={setTopP}
            min={0} max={1} step={0.05} />
        </ParamsCard>
      </>}>
      {!out && !busy && !err && (
        <EmptyState
          icon={<Sparkles size={32} strokeWidth={1.2} />}
          title="Nothing yet"
          desc="Tokens stream in here as the engine generates them." />
      )}
      <div className="text-[13px] whitespace-pre-wrap leading-relaxed">
        {out}
        {busy && (
          <span className="inline-block w-1.5 h-4 bg-accent animate-pulse rounded-sm align-text-bottom ml-0.5" />
        )}
      </div>
    </GenLayout>
  )
}

/* ---------- (b) adaptive ---------- */

function AdaptiveMode() {
  const [prompt, setPrompt] = useState('')
  const [thinkTok, setThinkTok] = useState(512)
  const [noThinkTok, setNoThinkTok] = useState(256)
  const [temp, setTemp] = useState(0)
  const [topP, setTopP] = useState(1.0)
  const [topK, setTopK] = useState(80)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const run = async () => {
    const p = prompt.trim()
    if (!p || busy) return
    setBusy(true); setErr('')
    try {
      const r = await api.post<Record<string, unknown>>('/api/gen/adaptive', {
        prompt: p,
        think_max_tokens: thinkTok, no_think_max_tokens: noThinkTok,
        temperature: temp, top_p: topP, top_k: topK,
      })
      if (typeof r.error === 'string') setErr(r.error)
      else setResult(r)
    } catch (e) { setErr(String(e)) }
    finally { setBusy(false) }
  }

  const tokS = result && typeof result.tok_s === 'number' ? result.tok_s : null

  return (
    <GenLayout
      meta={<>
        {tokS != null && <Tag kind="idle">{tokS.toFixed(1)} tok/s</Tag>}
        {busy && <Tag kind="accent"><Spinner /> generating</Tag>}
      </>}
      side={<>
        <Card title="Prompt">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6} placeholder="Prompt — the engine decides whether to think…"
            className="w-full !text-[13px]" />
          <Btn variant="primary" className="w-full justify-center mt-2.5"
            disabled={!prompt.trim() || busy} onClick={run}>
            {busy ? <Spinner /> : <Play size={13} />} Generate
          </Btn>
          <ErrLine msg={err} />
        </Card>
        <ParamsCard>
          <Num label="Think max tok" value={thinkTok} onChange={setThinkTok}
            min={1} max={8192} step={32} />
          <Num label="No-think max tok" value={noThinkTok} onChange={setNoThinkTok}
            min={1} max={8192} step={32} />
          <Num label="Temperature" value={temp} onChange={setTemp}
            min={0} max={2} step={0.05} />
          <Num label="Top-p" value={topP} onChange={setTopP}
            min={0} max={1} step={0.05} />
          <Num label="Top-k" value={topK} onChange={setTopK}
            min={0} max={500} step={1} />
        </ParamsCard>
      </>}>
      {result ? <JsonResult data={result} /> : (
        <EmptyState
          icon={<Sparkles size={32} strokeWidth={1.2} />}
          title="Nothing yet"
          desc="Adaptive generation picks a thinking or direct answer budget per prompt." />
      )}
    </GenLayout>
  )
}

/* ---------- (c) batch ---------- */

function BatchMode() {
  const [text, setText] = useState('')
  const [maxTok, setMaxTok] = useState(256)
  const [temp, setTemp] = useState(0)
  const [topP, setTopP] = useState(1.0)
  const [topK, setTopK] = useState(80)
  const [pairs, setPairs] = useState<{ prompt: string; output: string }[]>([])
  const [tokS, setTokS] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const prompts = text.split('\n').map((s) => s.trim()).filter(Boolean)

  const run = async () => {
    if (!prompts.length || busy) return
    setBusy(true); setErr('')
    try {
      const r = await api.post<{
        results?: string[]; tok_s?: number; error?: string
      }>('/api/gen/batch', {
        prompts, max_new_tokens: maxTok,
        temperature: temp, top_p: topP, top_k: topK,
      })
      if (typeof r.error === 'string') {
        setErr(r.error)
      } else {
        const res = r.results ?? []
        setPairs(prompts.map((p, i) => ({ prompt: p, output: res[i] ?? '' })))
        if (typeof r.tok_s === 'number') setTokS(r.tok_s)
      }
    } catch (e) { setErr(String(e)) }
    finally { setBusy(false) }
  }

  const exportJson = () => {
    if (!pairs.length) return
    const blob = new Blob([JSON.stringify(pairs, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `batch_generations_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <GenLayout
      meta={<>
        {pairs.length > 0 && (
          <span className="text-[10.5px] text-text-faint font-mono">
            {pairs.length} result{pairs.length > 1 ? 's' : ''}
          </span>
        )}
        {tokS != null && <Tag kind="accent">{tokS.toFixed(1)} tok/s</Tag>}
        {busy && <Tag kind="accent" pulse><Spinner /> generating</Tag>}
        {pairs.length > 0 && !busy && (
          <div className="flex items-center gap-1 border-l border-border/60 pl-2">
            <button
              onClick={exportJson}
              title="Download results as JSON"
              className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-panel-alt hover:bg-panel-hover border border-border text-text-dim hover:text-text cursor-pointer transition-colors">
              <Download size={11} /> Export JSON
            </button>
            <button
              onClick={() => { setPairs([]); setTokS(null) }}
              title="Clear results"
              className="p-1 rounded text-text-faint hover:text-err hover:bg-err/10 transition-colors cursor-pointer">
              <Trash2 size={12} />
            </button>
          </div>
        )}
      </>}
      side={<>
        <Card title="Prompts">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8} placeholder={'One prompt per line…'}
            className="w-full !text-[13px] font-mono" />
          <Btn variant="primary" className="w-full justify-center mt-2.5"
            disabled={!prompts.length || busy} onClick={run}>
            {busy ? <Spinner /> : <Play size={13} />}
            Run batch ({prompts.length})
          </Btn>
          <ErrLine msg={err} />
        </Card>
        <ParamsCard>
          <Num label="Max new tokens" value={maxTok} onChange={setMaxTok}
            min={1} max={8192} step={32} />
          <Num label="Temperature" value={temp} onChange={setTemp}
            min={0} max={2} step={0.05} />
          <Num label="Top-p" value={topP} onChange={setTopP}
            min={0} max={1} step={0.05} />
          <Num label="Top-k" value={topK} onChange={setTopK}
            min={0} max={500} step={1} />
        </ParamsCard>
      </>}>
      {pairs.length ? (
        <div className="space-y-2">
          {pairs.map((p, i) => (
            <div key={i} className="border border-border rounded-lg overflow-hidden">
              <div className="px-3 py-1.5 text-[11.5px] text-text-faint bg-panel-alt border-b border-border truncate"
                title={p.prompt}>
                {p.prompt}
              </div>
              <pre className="px-3 py-2 text-[12.5px] font-mono text-text whitespace-pre-wrap">
                {p.output}
              </pre>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<Sparkles size={32} strokeWidth={1.2} />}
          title="Nothing yet"
          desc="Each line of the prompt box is generated independently in one batch call." />
      )}
    </GenLayout>
  )
}

/* ---------- (d) raw ---------- */

function RawMode() {
  const [prompt, setPrompt] = useState('')
  const [maxTok, setMaxTok] = useState(256)
  const [temp, setTemp] = useState(0.2)
  const [topP, setTopP] = useState(1.0)
  const [topK, setTopK] = useState(80)
  const [repPen, setRepPen] = useState(1.05)
  const [minP, setMinP] = useState(0)
  const [minK, setMinK] = useState(0)
  const [skipSpecial, setSkipSpecial] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const run = async () => {
    const p = prompt.trim()
    if (!p || busy) return
    setBusy(true); setErr('')
    try {
      const r = await api.post<Record<string, unknown>>('/api/gen/raw', {
        prompt: p, max_new_tokens: maxTok,
        temperature: temp, top_p: topP, top_k: topK,
        repetition_penalty: repPen, min_p: minP, min_k: minK,
        skip_special_tokens: skipSpecial,
      })
      if (typeof r.error === 'string') setErr(r.error)
      else setResult(r)
    } catch (e) { setErr(String(e)) }
    finally { setBusy(false) }
  }

  const tokS = result && typeof result.tok_s === 'number' ? result.tok_s : null

  return (
    <GenLayout
      meta={<>
        {tokS != null && <Tag kind="idle">{tokS.toFixed(1)} tok/s</Tag>}
        {busy && <Tag kind="accent"><Spinner /> generating</Tag>}
      </>}
      side={<>
        <Card title="Prompt">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6} placeholder="Raw prompt — no chat template applied…"
            className="w-full !text-[13px]" />
          <Btn variant="primary" className="w-full justify-center mt-2.5"
            disabled={!prompt.trim() || busy} onClick={run}>
            {busy ? <Spinner /> : <Play size={13} />} Generate
          </Btn>
          <ErrLine msg={err} />
        </Card>
        <ParamsCard>
          <Num label="Max new tokens" value={maxTok} onChange={setMaxTok}
            min={1} max={8192} step={32} />
          <Num label="Temperature" value={temp} onChange={setTemp}
            min={0} max={2} step={0.05} />
          <Num label="Top-p" value={topP} onChange={setTopP}
            min={0} max={1} step={0.05} />
          <Num label="Top-k" value={topK} onChange={setTopK}
            min={0} max={500} step={1} />
          <Num label="Rep. penalty" value={repPen} onChange={setRepPen}
            min={1} max={2} step={0.01} />
          <Num label="Min-p" value={minP} onChange={setMinP}
            min={0} max={1} step={0.01} />
          <Num label="Min-k" value={minK} onChange={setMinK}
            min={0} max={500} step={1} />
          <CheckBox label="Skip special tokens" checked={skipSpecial}
            onChange={setSkipSpecial} />
        </ParamsCard>
      </>}>
      {result ? <JsonResult data={result} /> : (
        <EmptyState
          icon={<Sparkles size={32} strokeWidth={1.2} />}
          title="Nothing yet"
          desc="Raw decode with the full sampler stack — penalty, min-p and min-k filters included." />
      )}
    </GenLayout>
  )
}
