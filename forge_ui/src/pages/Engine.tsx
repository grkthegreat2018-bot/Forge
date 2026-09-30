// Engine — control + tuning: status/power, activation editor,
// maintenance ops, logs, sessions, prompt library.

import { clsx } from 'clsx'
import {
  BookOpen, ChevronDown, ChevronRight, FileText, Moon,
  Play, RefreshCw, ScrollText, Square, Sun,
  Wrench, X, Zap,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Dispatch, ReactNode, SetStateAction } from 'react'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import type {
  ActivationCatalog, ActivationField, EngineSnapshot,
} from '../lib/types'
import {
  Btn, Card, Check, IconBtn, Input, KvRow, NumInput,
  SearchInput, Select, Spinner, StatusDot, Tag, Textarea,
} from '../components/ui'

type TabId = 'status' | 'activation' | 'maint' | 'log' | 'sessions' | 'library'

const TABS: { id: TabId; label: string; icon: ReactNode }[] = [
  { id: 'status', label: 'Status', icon: <Zap size={12} /> },
  { id: 'activation', label: 'Activation', icon: <Wrench size={12} /> },
  { id: 'maint', label: 'Maintenance', icon: <Play size={12} /> },
  { id: 'log', label: 'Log & outputs', icon: <ScrollText size={12} /> },
  { id: 'sessions', label: 'Sessions', icon: <FileText size={12} /> },
  { id: 'library', label: 'Library', icon: <BookOpen size={12} /> },
]

interface EngineStats {
  ready: boolean
  stats?: Record<string, unknown>
  vram?: Record<string, unknown>
  lora?: Record<string, unknown>
  awake?: boolean
  info?: Record<string, unknown>
}

interface OpResult {
  ok: boolean
  result?: unknown
  error?: string
}

type SetCfg = Dispatch<SetStateAction<Record<string, unknown>>>

function fmtVal(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function num(v: unknown, d: number): number {
  return typeof v === 'number' && !Number.isNaN(v) ? v : d
}

function RecRows({ rec }: { rec: Record<string, unknown> }) {
  const entries = Object.entries(rec)
  if (!entries.length) {
    return <div className="text-[12px] text-text-faint py-1">—</div>
  }
  return (
    <>
      {entries.slice(0, 60).map(([k, v]) => (
        <KvRow key={k} k={k} v={fmtVal(v)}
          mono={typeof v === 'string' && v.length > 28} />
      ))}
    </>
  )
}

function Group({ title, rec }: {
  title: string
  rec?: Record<string, unknown> | null
}) {
  return (
    <div className="min-w-0">
      <div className="text-[10.5px] font-semibold uppercase tracking-wider text-text-faint mb-1">
        {title}
      </div>
      {rec ? <RecRows rec={rec} />
        : <div className="text-[12px] text-text-faint py-1">—</div>}
    </div>
  )
}

function errOf(r: unknown): string | null {
  const o = r as OpResult | null
  return o && o.ok === false ? (o.error ?? 'request failed') : null
}

export default function Engine() {
  const engine = useForge((s) => s.engine)
  const [tab, setTab] = useState<TabId>('status')
  const [visited, setVisited] = useState<Set<TabId>>(new Set<TabId>(['status']))
  const [err, setErr] = useState('')
  // activation edits live here so the Status tab can reactivate with them
  const [actCfg, setActCfg] = useState<Record<string, unknown>>({})

  const open = (t: TabId) => {
    setTab(t)
    setVisited((v) => (v.has(t) ? v : new Set(v).add(t)))
  }

  const show = (t: TabId) => visited.has(t) && (
    <div hidden={tab !== t} className={tab === t ? 'space-y-4' : undefined}>
      {t === 'status' && (
        <StatusPanel engine={engine} actCfg={actCfg} setErr={setErr} />)}
      {t === 'activation' && (
        <ActivationPanel engine={engine} actCfg={actCfg}
          setActCfg={setActCfg} setErr={setErr} />)}
      {t === 'maint' && <MaintPanel engine={engine} setErr={setErr} />}
      {t === 'log' && <LogPanel setErr={setErr} />}
      {t === 'sessions' && <SessionsPanel engine={engine} setErr={setErr} />}
      {t === 'library' && <LibraryPanel engine={engine} setErr={setErr} />}
    </div>
  )

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1 px-4 pt-3 pb-2 border-b border-border shrink-0">
        {TABS.map((t) => (
          <button key={t.id} onClick={() => open(t.id)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12.5px] transition-colors cursor-pointer',
              tab === t.id
                ? 'bg-accent/15 text-accent-hi border border-accent/25'
                : 'text-text-dim border border-transparent hover:bg-panel-alt hover:text-text')}>
            {t.icon}{t.label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {err && (
          <div className="text-[12px] text-err bg-err/10 border border-err/30 rounded-lg px-3 py-1.5 mb-3 flex items-center gap-2">
            <span className="flex-1">{err}</span>
            <button onClick={() => setErr('')}
              className="text-text-faint hover:text-text cursor-pointer">
              <X size={11} />
            </button>
          </div>
        )}
        {TABS.map((t) => <div key={t.id}>{show(t.id)}</div>)}
      </div>
    </div>
  )
}

/* ---------- (a) status & power ---------- */

function StatusPanel({ engine, actCfg, setErr }: {
  engine: EngineSnapshot
  actCfg: Record<string, unknown>
  setErr: (e: string) => void
}) {
  const [stats, setStats] = useState<EngineStats | null>(null)
  const [sleepLevel, setSleepLevel] = useState(1)
  const [busy, setBusy] = useState('')
  const ready = engine.state === 'ready'

  const load = () => {
    api.get<EngineStats>('/api/engine/stats')
      .then(setStats)
      .catch((e) => setErr(String(e)))
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { load() }, [])

  const power = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    setErr('')
    try {
      const e = errOf(await fn())
      if (e) setErr(e)
      await useForge.getState().refreshStatus().catch(() => undefined)
      load()
    } catch (e) { setErr(String(e)) } finally { setBusy('') }
  }

  const kind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  return (
    <>
      <Card title="Engine" actions={
        <IconBtn title="Refresh stats" onClick={load}>
          <RefreshCw size={12} />
        </IconBtn>
      }>
        <div className="flex items-center gap-2 mb-3">
          <StatusDot kind={kind} pulse={engine.state === 'loading'} />
          <span className="text-[14px] font-semibold capitalize">
            {engine.state}
          </span>
          {engine.busy && <Tag kind="accent">busy</Tag>}
          {stats?.awake != null && (
            <Tag kind={stats.awake ? 'ok' : 'idle'}>
              {stats.awake ? 'awake' : 'asleep'}
            </Tag>
          )}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-6 gap-y-4">
          <Group title="info" rec={stats?.info ?? engine.info} />
          <Group title="stats" rec={stats?.stats} />
          <Group title="vram" rec={stats?.vram} />
          <Group title="lora" rec={stats?.lora} />
        </div>
      </Card>

      <Card title="Power">
        <div className="flex items-center gap-2 flex-wrap">
          <NumInput label="sleep level" value={sleepLevel}
            onChange={setSleepLevel} min={0} max={3} step={1} />
          <Btn variant="subtle" disabled={!ready || !!busy}
            onClick={() => power('sleep',
              () => api.post('/api/engine/sleep', { level: sleepLevel }))}>
            {busy === 'sleep' ? <Spinner /> : <Moon size={12} />} Sleep
          </Btn>
          <Btn variant="subtle" disabled={!ready || !!busy}
            onClick={() => power('wake', () => api.post('/api/engine/wake'))}>
            {busy === 'wake' ? <Spinner /> : <Sun size={12} />} Wake
          </Btn>
          <Btn variant="subtle" disabled={!ready || !!busy}
            title={Object.keys(actCfg).length
              ? 'reactivate with the Activation tab edits'
              : 'reactivate with engine defaults'}
            onClick={() => power('reactivate',
              () => api.post('/api/engine/reactivate',
                Object.keys(actCfg).length ? { activation: actCfg } : {}))}>
            {busy === 'reactivate' ? <Spinner /> : <RefreshCw size={12} />}
            Reactivate
          </Btn>
          <Btn variant="danger" disabled={!ready || !!busy}
            onClick={() => power('unload', () => api.post('/api/engine/unload'))}>
            {busy === 'unload' ? <Spinner /> : <Square size={12} />} Unload
          </Btn>
        </div>
        {!ready && (
          <div className="text-[11.5px] text-text-faint mt-2">
            power controls need a resident engine — load one on the Models page.
          </div>
        )}
      </Card>
    </>
  )
}

/* ---------- (b) activation editor ---------- */

function ActivationPanel({ engine, actCfg, setActCfg, setErr }: {
  engine: EngineSnapshot
  actCfg: Record<string, unknown>
  setActCfg: SetCfg
  setErr: (e: string) => void
}) {
  const [catalog, setCatalog] = useState<ActivationCatalog | null>(null)
  const [open, setOpen] = useState<Set<string>>(new Set())
  const [errors, setErrors] = useState<string[] | null>(null)
  const [diff, setDiff] = useState<string[] | null>(null)
  const [busy, setBusy] = useState('')
  const ready = engine.state === 'ready'

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    api.get<ActivationCatalog>('/api/activation/catalog')
      .then((c) => {
        setCatalog(c)
        setActCfg((prev) => Object.keys(prev).length ? prev : {
          ...(c.default ?? {}),
          ...(engine.info.activation ?? {}),
        })
        if (c.categories?.length) setOpen(new Set([c.categories[0].name]))
      })
      .catch((e) => setErr(String(e)))
  }, [])

  const setField = (name: string, v: unknown) =>
    setActCfg((c) => ({ ...c, [name]: v }))

  const loadPreset = async (name: string) => {
    setErr('')
    try {
      const r = await api.get<{ ok: boolean; config?: Record<string, unknown>; error?: string }>(
        `/api/activation/preset/${encodeURIComponent(name)}`)
      if (r.ok && r.config) setActCfg(r.config)
      else setErr(r.error ?? `unknown preset ${name}`)
    } catch (e) { setErr(String(e)) }
  }

  const doValidate = async () => {
    setBusy('validate')
    setErr('')
    try {
      const r = await api.post<{ errors: string[] }>(
        '/api/activation/validate', { config: actCfg })
      setErrors(r.errors ?? [])
      setDiff(null)
    } catch (e) { setErr(String(e)) } finally { setBusy('') }
  }

  const doDiff = async () => {
    setBusy('diff')
    setErr('')
    try {
      const r = await api.post<{ diff: unknown }>(
        '/api/activation/diff', { config: actCfg })
      const d = r.diff
      setDiff(Array.isArray(d)
        ? d.map(String)
        : d ? [JSON.stringify(d, null, 2)] : [])
      setErrors(null)
    } catch (e) { setErr(String(e)) } finally { setBusy('') }
  }

  const doApply = async () => {
    setBusy('apply')
    setErr('')
    try {
      await api.post('/api/engine/reactivate', { activation: actCfg })
      useForge.getState().refreshStatus().catch(() => undefined)
    } catch (e) { setErr(String(e)) } finally { setBusy('') }
  }

  if (!catalog) {
    return (
      <div className="flex items-center gap-2 text-[12.5px] text-text-dim py-8 justify-center">
        <Spinner /> loading catalog…
      </div>
    )
  }

  return (
    <>
      <Card title="Presets">
        <div className="flex items-center gap-1.5 flex-wrap">
          {catalog.presets.map((p) => (
            <Btn key={p.name} variant="outline" title={p.desc}
              onClick={() => loadPreset(p.name)}>
              {p.name}
            </Btn>
          ))}
        </div>
      </Card>

      <Card title="Features" actions={
        <div className="flex items-center gap-1.5">
          <Btn variant="subtle" disabled={!!busy} onClick={doValidate}>
            {busy === 'validate' ? <Spinner /> : null} Validate
          </Btn>
          <Btn variant="subtle" disabled={!!busy} onClick={doDiff}>
            {busy === 'diff' ? <Spinner /> : null} Show diff
          </Btn>
          <Btn variant="primary" disabled={!ready || !!busy} onClick={doApply}
            title={ready ? 'apply to resident engine' : 'engine not ready'}>
            {busy === 'apply' ? <Spinner /> : null} Apply
          </Btn>
        </div>
      }>
        {errors !== null && (
          <div className={clsx(
            'rounded-lg border px-3 py-2 mb-2 text-[12px]',
            errors.length
              ? 'border-err/40 bg-err/10 text-err'
              : 'border-ok/40 bg-ok/10 text-ok')}>
            {errors.length
              ? errors.map((e, i) => <div key={i}>{e}</div>)
              : 'config valid'}
          </div>
        )}
        {diff !== null && (
          <div className="rounded-lg border border-border bg-deep px-3 py-2 mb-2 text-[11.5px] font-mono text-text-dim max-h-40 overflow-y-auto">
            {diff.length
              ? diff.map((d, i) => <div key={i}>{d}</div>)
              : 'no differences vs active config'}
          </div>
        )}
        <div className="space-y-1">
          {catalog.categories.map((cat) => {
            const isOpen = open.has(cat.name)
            return (
              <div key={cat.name} className="border border-border rounded-lg overflow-hidden">
                <button
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-left cursor-pointer hover:bg-panel-alt"
                  onClick={() => setOpen((s) => {
                    const n = new Set(s)
                    if (n.has(cat.name)) n.delete(cat.name); else n.add(cat.name)
                    return n
                  })}>
                  {isOpen
                    ? <ChevronDown size={12} className="text-text-faint" />
                    : <ChevronRight size={12} className="text-text-faint" />}
                  <span className="text-[12px] font-medium">{cat.name}</span>
                  <span className="text-[10.5px] text-text-faint ml-auto">
                    {cat.fields.length} fields
                  </span>
                </button>
                {isOpen && (
                  <div className="px-3 py-1.5 border-t border-border bg-panel-alt/40">
                    {cat.fields.map((f) => (
                      <div key={f.name} title={f.tooltip}
                        className="grid grid-cols-[190px_1fr] items-center gap-3 py-1">
                        <div className="text-[12px] text-text-dim truncate">
                          {f.label}
                          {f.suffix && (
                            <span className="text-text-faint"> ({f.suffix})</span>
                          )}
                        </div>
                        <FieldEditor f={f} value={actCfg[f.name]}
                          onChange={(v) => setField(f.name, v)} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </Card>
    </>
  )
}

function FieldEditor({ f, value, onChange }: {
  f: ActivationField
  value: unknown
  onChange: (v: unknown) => void
}) {
  if (f.kind === 'combo') {
    const hasNone = f.options.some((o) => o.value === 'none')
    const val = value == null
      ? (hasNone ? 'none' : (f.options[0]?.value ?? ''))
      : String(value)
    return (
      <Select value={val} className="w-full !py-1 !text-[12px]"
        onChange={(e) =>
          onChange(e.target.value === 'none' ? null : e.target.value)}>
        {f.options.map((o) => (
          <option key={o.value} value={o.value} title={o.tip}>{o.label}</option>
        ))}
      </Select>
    )
  }
  if (f.kind === 'bool') {
    return <Check label="enabled" checked={Boolean(value)} onChange={onChange} />
  }
  if (f.kind === 'int') {
    return (
      <NumInput value={num(value, num(f.default, 0))}
        onChange={(v) => onChange(Math.round(v))}
        min={f.lo} max={f.hi} step={f.step || 1} />
    )
  }
  // opt_int / opt_float — Check toggles between null and a value
  const enabled = value !== null && value !== undefined
  const isInt = f.kind === 'opt_int'
  return (
    <div className="flex items-center gap-2">
      <Check label="set" checked={enabled}
        onChange={(b) => onChange(b ? num(f.default ?? f.lo, 0) : null)} />
      {enabled ? (
        <NumInput value={num(value, 0)}
          onChange={(v) => onChange(isInt ? Math.round(v) : v)}
          min={f.lo} max={f.hi}
          step={f.step || (isInt ? 1 : 0.1)} />
      ) : (
        <span className="text-[11px] text-text-faint">off</span>
      )}
    </div>
  )
}

/* ---------- (c) maintenance ---------- */

function MaintPanel({ engine, setErr }: {
  engine: EngineSnapshot
  setErr: (e: string) => void
}) {
  const [prompt, setPrompt] = useState('The quick brown fox')
  const [maxTokens, setMaxTokens] = useState(64)
  const [runs, setRuns] = useState(3)
  const [chunkSize, setChunkSize] = useState(512)
  const [maxChunks, setMaxChunks] = useState(16)
  const [blendText, setBlendText] = useState('')
  const [out, setOut] = useState('')
  const [busy, setBusy] = useState('')
  const ready = engine.state === 'ready'

  const run = async (label: string, path: string, body?: unknown) => {
    setBusy(label)
    setErr('')
    try {
      const r = await api.post<OpResult>(path, body)
      setOut(r.ok
        ? JSON.stringify(r.result ?? { ok: true }, null, 2)
        : `error: ${r.error ?? 'request failed'}`)
      useForge.getState().refreshStatus().catch(() => undefined)
    } catch (e) {
      setOut(String(e))
      setErr(String(e))
    } finally { setBusy('') }
  }

  const opBtn = (label: string, path: string,
               body?: unknown, primary?: boolean) => (
    <Btn key={label} variant={primary ? 'primary' : 'subtle'}
      disabled={!ready || !!busy}
      onClick={() => run(label, path, body)}>
      {busy === label ? <Spinner /> : null}{label}
    </Btn>
  )

  return (
    <>
      <Card title="Maintenance ops">
        <div className="flex items-center gap-2 flex-wrap mb-3">
          <Input value={prompt} onChange={(e) => setPrompt(e.target.value)}
            placeholder="benchmark prompt" className="flex-1 min-w-[180px] font-mono !text-[12px]" />
          <NumInput label="tok" value={maxTokens} onChange={setMaxTokens}
            min={1} max={2048} step={16} />
          <NumInput label="runs" value={runs} onChange={setRuns}
            min={1} max={20} step={1} />
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {opBtn('Benchmark', '/api/engine/benchmark',
            { prompt, max_tokens: maxTokens, runs }, true)}
          {opBtn('Bottleneck', '/api/engine/bottleneck')}
          {opBtn('Diagnose', '/api/engine/diagnose')}
          {opBtn('Recover', '/api/engine/recover')}
          {opBtn('Clear recovery', '/api/engine/clear_recovery')}
          {opBtn('Reset stats', '/api/engine/reset_stats')}
        </div>
        {!ready && (
          <div className="text-[11.5px] text-text-faint mt-2">
            maintenance ops need a resident engine.
          </div>
        )}
      </Card>

      <Card title="Cache blend">
        <div className="flex items-center gap-2 flex-wrap mb-2">
          <NumInput label="chunk" value={chunkSize} onChange={setChunkSize}
            min={16} max={8192} step={16} />
          <NumInput label="max chunks" value={maxChunks} onChange={setMaxChunks}
            min={1} max={256} step={1} />
          {opBtn('Enable', '/api/engine/cache_blend/enable',
            { chunk_size: chunkSize, max_chunks: maxChunks })}
        </div>
        <div className="flex items-center gap-2">
          <Input value={blendText} onChange={(e) => setBlendText(e.target.value)}
            placeholder="chunk text to register…"
            className="flex-1 !text-[12px]" />
          {opBtn('Register', '/api/engine/cache_blend/register',
            { text: blendText })}
        </div>
      </Card>

      <Card title="Output">
        <pre className="font-mono text-[11.5px] text-text-dim whitespace-pre-wrap max-h-80 overflow-y-auto bg-deep rounded-lg p-3 border border-border min-h-[80px]">
          {out || 'results appear here'}
        </pre>
      </Card>
    </>
  )
}

/* ---------- (d) log & outputs ---------- */

function LogPanel({ setErr }: { setErr: (e: string) => void }) {
  const [lines, setLines] = useState<string[]>([])
  const [outputs, setOutputs] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)

  const load = async () => {
    setBusy(true)
    try {
      const [l, o] = await Promise.all([
        api.get<{ lines: string[] }>('/api/engine/log?lines=200'),
        api.get<{ outputs: unknown }>('/api/engine/outputs?n=10'),
      ])
      setLines(l.lines ?? [])
      setOutputs(o.outputs)
    } catch (e) { setErr(String(e)) } finally { setBusy(false) }
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { load() }, [])

  return (
    <>
      <Card title="Engine log" actions={
        <IconBtn title="Refresh" onClick={load}>
          {busy ? <Spinner /> : <RefreshCw size={12} />}
        </IconBtn>
      }>
        <pre className="font-mono text-[11px] text-text-dim whitespace-pre-wrap max-h-72 overflow-y-auto bg-deep rounded-lg p-3 border border-border min-h-[60px]">
          {lines.length ? lines.join('\n') : 'no log lines'}
        </pre>
      </Card>
      <Card title="Recent outputs">
        <pre className="font-mono text-[11px] text-text-dim whitespace-pre-wrap max-h-72 overflow-y-auto bg-deep rounded-lg p-3 border border-border min-h-[60px]">
          {outputs == null
            ? 'none'
            : JSON.stringify(outputs, null, 2)}
        </pre>
      </Card>
    </>
  )
}

/* ---------- (e) sessions ---------- */

function SessionsPanel({ engine, setErr }: {
  engine: EngineSnapshot
  setErr: (e: string) => void
}) {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [sessId, setSessId] = useState('')
  const [ttl, setTtl] = useState(300)
  const [prompt, setPrompt] = useState('')
  const [maxTok, setMaxTok] = useState(256)
  const [temp, setTemp] = useState(0.7)
  const [topP, setTopP] = useState(0.95)
  const [out, setOut] = useState('')
  const [busy, setBusy] = useState('')
  const ready = engine.state === 'ready'

  const loadStats = () => {
    api.get<{ ok: boolean; stats?: Record<string, unknown>; error?: string }>(
      '/api/engine/sessions')
      .then((r) => setStats(r.stats ?? null))
      .catch((e) => setErr(String(e)))
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { loadStats() }, [])

  const op = async (label: string, path: string, body: unknown) => {
    setBusy(label)
    setErr('')
    try {
      const r = await api.post<OpResult>(path, body)
      setOut(r.ok
        ? JSON.stringify(r.result ?? { ok: true }, null, 2)
        : `error: ${r.error ?? 'request failed'}`)
      // begin returns the new session id — grab it for follow-up calls
      const res = r.result
      if (res && typeof res === 'object') {
        const id = (res as Record<string, unknown>).session_id
          ?? (res as Record<string, unknown>).id
        if (typeof id === 'string' && id) setSessId(id)
      }
      loadStats()
    } catch (e) {
      setOut(String(e))
      setErr(String(e))
    } finally { setBusy('') }
  }

  return (
    <>
      <Card title="Session stats" actions={
        <IconBtn title="Refresh" onClick={loadStats}><RefreshCw size={12} /></IconBtn>
      }>
        {stats ? <RecRows rec={stats} />
          : <div className="text-[12px] text-text-faint">—</div>}
      </Card>

      <Card title="Begin session">
        <div className="flex items-center gap-2 flex-wrap">
          <Input value={sessId} onChange={(e) => setSessId(e.target.value)}
            placeholder="session id (blank = auto)"
            className="flex-1 min-w-[160px] font-mono !text-[12px]" />
          <NumInput label="ttl s" value={ttl} onChange={setTtl}
            min={10} max={86400} step={30} />
          <Btn variant="primary" disabled={!ready || !!busy}
            onClick={() => op('begin', '/api/engine/sessions/begin',
              { session_id: sessId, ttl_s: ttl })}>
            {busy === 'begin' ? <Spinner /> : <Play size={12} />} Begin
          </Btn>
        </div>
      </Card>

      <Card title="Continue session">
        <div className="space-y-2">
          <Input value={prompt} onChange={(e) => setPrompt(e.target.value)}
            placeholder="prompt for the pinned session…"
            className="w-full !text-[12px]" />
          <div className="flex items-center gap-3 flex-wrap">
            <NumInput label="max tok" value={maxTok} onChange={setMaxTok}
              min={1} max={8192} step={64} />
            <NumInput label="temp" value={temp} onChange={setTemp}
              min={0} max={2} step={0.05} />
            <NumInput label="top-p" value={topP} onChange={setTopP}
              min={0} max={1} step={0.05} />
            <Btn variant="subtle" disabled={!ready || !sessId || !prompt || !!busy}
              onClick={() => op('continue', '/api/engine/sessions/continue', {
                session_id: sessId, prompt, max_tokens: maxTok,
                temperature: temp, top_p: topP,
              })}>
              {busy === 'continue' ? <Spinner /> : null} Continue
            </Btn>
          </div>
        </div>
      </Card>

      <Card title="Pin / end by id">
        <div className="flex items-center gap-2 flex-wrap">
          <Btn variant="subtle" disabled={!ready || !sessId || !!busy}
            onClick={() => op('pin', '/api/engine/sessions/pin',
              { session_id: sessId, ttl_s: ttl })}>
            Pin
          </Btn>
          <Btn variant="subtle" disabled={!ready || !sessId || !!busy}
            onClick={() => op('unpin', '/api/engine/sessions/unpin',
              { session_id: sessId })}>
            Unpin
          </Btn>
          <Btn variant="danger" disabled={!ready || !sessId || !!busy}
            onClick={() => op('end', '/api/engine/sessions/end',
              { session_id: sessId })}>
            End
          </Btn>
        </div>
      </Card>

      <Card title="Output">
        <pre className="font-mono text-[11.5px] text-text-dim whitespace-pre-wrap max-h-64 overflow-y-auto bg-deep rounded-lg p-3 border border-border min-h-[60px]">
          {out || 'results appear here'}
        </pre>
      </Card>
    </>
  )
}

/* ---------- (f) prompt library ---------- */

function LibraryPanel({ engine, setErr }: {
  engine: EngineSnapshot
  setErr: (e: string) => void
}) {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [entries, setEntries] = useState<unknown[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<unknown[] | null>(null)
  const [enabled, setEnabled] = useState(true)
  const [budget, setBudget] = useState(256)
  const [content, setContent] = useState('')
  const [category, setCategory] = useState('general')
  const [priority, setPriority] = useState(5)
  const [desc, setDesc] = useState('')
  const [tags, setTags] = useState('')
  const [busy, setBusy] = useState('')
  const ready = engine.state === 'ready'

  const load = () => {
    api.get<{ ok: boolean; stats?: Record<string, unknown> }>(
      '/api/engine/library/stats')
      .then((r) => setStats(r.stats ?? null))
      .catch((e) => setErr(String(e)))
    api.get<{ ok: boolean; entries?: unknown }>('/api/engine/library/list')
      .then((r) => setEntries(Array.isArray(r.entries) ? r.entries : []))
      .catch(() => setEntries([]))
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { load() }, [])

  const op = async (label: string, path: string, body?: unknown) => {
    setBusy(label)
    setErr('')
    try {
      const r = await api.post<OpResult & { results?: unknown }>(path, body)
      if (r.ok === false) setErr(r.error ?? 'request failed')
      return r
    } catch (e) { setErr(String(e)); return null } finally { setBusy('') }
  }

  const doSearch = async () => {
    const r = await op('search', '/api/engine/library/search', { query })
    const res = r?.results
    setResults(Array.isArray(res) ? res : res ? [res] : [])
  }

  const doSave = async () => {
    if (!content.trim()) return
    const r = await op('save', '/api/engine/library/save', {
      content, category, priority,
      tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
      description: desc, triggers: [],
    })
    if (r?.ok) { setContent(''); setDesc(''); load() }
  }

  return (
    <>
      <Card title="Library" actions={
        <div className="flex items-center gap-3">
          <Check label="enabled" checked={enabled}
            onChange={(v) => {
              setEnabled(v)
              void op('enabled', '/api/engine/library/enabled', { enabled: v })
            }} />
          <NumInput label="budget" value={budget} onChange={setBudget} min={0}
            max={8192} step={32} />
          <Btn variant="subtle" disabled={!ready || !!busy}
            onClick={() => op('budget', '/api/engine/library/budget',
              { budget })}>
            Set
          </Btn>
          <Btn variant="subtle" disabled={!ready || !!busy}
            onClick={() => op('optimize', '/api/engine/library/optimize')
              .then(() => load())}>
            {busy === 'optimize' ? <Spinner /> : null} Optimize
          </Btn>
        </div>
      }>
        {stats ? <RecRows rec={stats} />
          : <div className="text-[12px] text-text-faint">—</div>}
      </Card>

      <Card title="Search">
        <div className="flex items-center gap-2 mb-2">
          <SearchInput
            value={query}
            onChange={setQuery}
            placeholder="Search prompt library…"
            className="flex-1"
          />
          <Btn variant="subtle" disabled={!ready || !!busy} onClick={doSearch}>
            {busy === 'search' ? <Spinner /> : null} Search
          </Btn>
        </div>
        {results !== null && (
          <div className="space-y-1.5">
            {results.map((e, i) => <LibEntry key={i} e={e} />)}
            {!results.length && (
              <div className="text-[12px] text-text-faint py-1">no results</div>
            )}
          </div>
        )}
      </Card>

      <Card title="Save entry">
        <div className="space-y-2">
          <Textarea value={content} onChange={(e) => setContent(e.target.value)}
            placeholder="library content — facts, instructions, snippets…"
            rows={3} className="w-full !text-[12px]" />
          <div className="flex items-center gap-2 flex-wrap">
            <Input value={category} onChange={(e) => setCategory(e.target.value)}
              placeholder="category" className="w-32 !text-[12px]" />
            <NumInput label="priority" value={priority} onChange={setPriority}
              min={0} max={10} step={1} />
            <Input value={tags} onChange={(e) => setTags(e.target.value)}
              placeholder="tags, comma-sep" className="flex-1 min-w-[120px] !text-[12px]" />
          </div>
          <div className="flex items-center gap-2">
            <Input value={desc} onChange={(e) => setDesc(e.target.value)}
              placeholder="description (optional)"
              className="flex-1 !text-[12px]" />
            <Btn variant="primary" disabled={!ready || !content.trim() || !!busy}
              onClick={doSave}>
              {busy === 'save' ? <Spinner /> : null} Save
            </Btn>
          </div>
        </div>
      </Card>

      <Card title={`Entries (${entries.length})`} actions={
        <IconBtn title="Refresh" onClick={load}><RefreshCw size={12} /></IconBtn>
      }>
        <div className="space-y-1.5 max-h-80 overflow-y-auto">
          {entries.map((e, i) => <LibEntry key={i} e={e} />)}
          {!entries.length && (
            <div className="text-[12px] text-text-faint py-1">no entries</div>
          )}
        </div>
      </Card>
    </>
  )
}

function LibEntry({ e }: { e: unknown }) {
  if (typeof e === 'string') {
    return <div className="text-[12px] text-text-dim">{e}</div>
  }
  const r = (e ?? {}) as Record<string, unknown>
  const content = r.content ?? r.text
  return (
    <div className="border border-border rounded-lg px-2.5 py-1.5">
      <div className="text-[12px] text-text whitespace-pre-wrap line-clamp-3">
        {content != null ? String(content) : JSON.stringify(r)}
      </div>
      <div className="flex items-center gap-1.5 mt-1 flex-wrap">
        {r.category != null && <Tag kind="idle">{String(r.category)}</Tag>}
        {r.priority != null && (
          <span className="text-[10.5px] text-text-faint">
            p{String(r.priority)}
          </span>
        )}
        {Array.isArray(r.tags) &&
          (r.tags as unknown[]).slice(0, 6).map((t, i) => (
            <Tag key={i} kind="accent">{String(t)}</Tag>
          ))}
      </div>
    </div>
  )
}
