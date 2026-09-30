// LoRA — adapter index, live harness (load/pin/mode), merge into base.

import { clsx } from 'clsx'
import {
  GitMerge, Layers, Pin, PinOff, RefreshCw, Sparkles,
  Square, Upload,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import type { LoraStatus, ModelEntry } from '../lib/types'
import {
  Btn, Card, Check, EmptyState, IconBtn, Input, KvRow,
  NumInput, SearchInput, Select, Spinner, StatusDot, Tag,
} from '../components/ui'

// adapter records are defensive-typed — the scan may produce
// different key sets for broken/legacy files
interface LoraRec {
  name?: string
  path?: string
  size_label?: string
  modified?: number
  rank?: number | null
  n_tensors?: number
  n_params?: number
  dtype?: string
  base_hint?: string
  header_error?: string
  category?: string
  [k: string]: unknown
}

interface OpResult {
  ok: boolean
  error?: string
  info?: Record<string, unknown>
  out?: string
}

function shortName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

function fmtParams(n: unknown): string {
  if (typeof n !== 'number' || !n) return '—'
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}

function fmtDate(ts: unknown): string {
  if (typeof ts !== 'number' || !ts) return '—'
  return new Date(ts * 1000).toLocaleDateString(
    undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

const FALLBACK_MODES = ['chat', 'agent', 'self_play', 'training']
const FALLBACK_TARGETS = ['default', 'ffn', 'attention']

export default function Lora() {
  const lora = useForge((s) => s.lora)
  const engine = useForge((s) => s.engine)

  const [adapters, setAdapters] = useState<LoraRec[]>([])
  const [models, setModels] = useState<ModelEntry[]>([])
  const [search, setSearch] = useState('')
  const [info, setInfo] = useState<Record<string, unknown> | null>(null)
  const [selPath, setSelPath] = useState<string | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState('')

  // mode harness
  const [mode, setMode] = useState('')
  const [rec, setRec] = useState<LoraRec | null>(null)

  // load panel
  const [rank, setRank] = useState(32)
  const [alphaOn, setAlphaOn] = useState(false)
  const [alpha, setAlpha] = useState(64)
  const [targetKey, setTargetKey] = useState('default')

  // merge panel
  const [mergeBase, setMergeBase] = useState('')
  const [mergeCfg, setMergeCfg] = useState('forgelm_v2')
  const [mergeRank, setMergeRank] = useState(32)
  const [mergeAlphaOn, setMergeAlphaOn] = useState(false)
  const [mergeAlpha, setMergeAlpha] = useState(64)
  const [mergeOut, setMergeOut] = useState(
    'research/checkpoints/merged.safetensors')
  const [mergeRes, setMergeRes] = useState('')

  const refresh = () => {
    api.get<{ adapters: LoraRec[]; status: LoraStatus }>('/api/lora')
      .then((r) => setAdapters(Array.isArray(r.adapters) ? r.adapters : []))
      .catch((e) => setErr(String(e)))
    api.get<{ models: ModelEntry[] }>('/api/models')
      .then((r) => setModels(
        r.models.filter((m) => m.is_safetensors && !m.is_lora)))
      .catch(() => undefined)
  }

  const loadInfo = () => {
    api.get<OpResult>('/api/lora/info')
      .then((r) => setInfo(r.info ?? null))
      .catch(() => setInfo(null))
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { refresh(); loadInfo() }, [])

  const curMode = mode || lora.mode || 'chat'
  const modes = lora.modes.length ? lora.modes : FALLBACK_MODES
  const targets = lora.target_presets.length
    ? lora.target_presets : FALLBACK_TARGETS

  // recommended adapter for the active mode
  useEffect(() => {
    api.get<{ adapter: LoraRec | null }>(
      `/api/lora/recommend?mode=${encodeURIComponent(curMode)}`)
      .then((r) => setRec(r.adapter ?? null))
      .catch(() => setRec(null))
  }, [curMode])

  const filteredAdapters = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return adapters
    return adapters.filter((a) =>
      (a.name && a.name.toLowerCase().includes(q)) ||
      (a.path && a.path.toLowerCase().includes(q)) ||
      (a.category && a.category.toLowerCase().includes(q))
    )
  }, [adapters, search])

  const sel = adapters.find((a) => a.path === selPath) ?? null
  const ready = engine.state === 'ready'

  const op = async (label: string, fn: () => Promise<unknown>,
    after?: () => void) => {
    setBusy(label)
    setErr('')
    try {
      const r = await fn()
      const o = r as OpResult | null
      if (o && o.ok === false) setErr(o.error ?? 'request failed')
      useForge.getState().refreshStatus().catch(() => undefined)
      after?.()
    } catch (e) { setErr(String(e)) } finally { setBusy('') }
  }

  const doSetMode = () =>
    op('mode', () => api.post('/api/lora/mode', { mode: curMode }),
      () => { refresh(); loadInfo() })

  const loadPath = (path: string) =>
    op('load', () => api.post('/api/lora/load', {
      path, rank,
      alpha: alphaOn ? alpha : null,
      target_key: targetKey,
    }), loadInfo)

  const doLoad = () => {
    if (sel?.path) void loadPath(sel.path)
  }

  const doUnload = () =>
    op('unload', () => api.post('/api/lora/unload'), loadInfo)

  const doPin = (path: string) =>
    op('pin', () => api.post('/api/lora/pin', { path }),
      () => { refresh(); loadInfo() })

  const doUnpin = () =>
    op('unpin', () => api.post('/api/lora/unpin'),
      () => { refresh(); loadInfo() })

  const doMerge = () => {
    if (!sel?.path || !mergeBase) return
    op('merge', async () => {
      const r = await api.post<OpResult>('/api/lora/merge', {
        base: mergeBase,
        config_name: mergeCfg || 'forgelm_v2',
        adapter: sel.path,
        rank: mergeRank,
        alpha: mergeAlphaOn ? mergeAlpha : null,
        out: mergeOut,
      })
      setMergeRes(r.ok
        ? `merged → ${r.out ?? mergeOut}`
        : `error: ${r.error ?? 'merge failed'}`)
      return r
    }, refresh)
  }

  return (
    <div className="flex h-full min-h-0">
      {/* ---- main: status + adapter table ---- */}
      <div className="flex-1 min-w-0 overflow-y-auto p-4 space-y-4">
        <Card title="LoRA harness" actions={
          <IconBtn title="Refresh"
            onClick={() => { refresh(); loadInfo() }}>
            <RefreshCw size={12} />
          </IconBtn>
        }>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2 min-w-0">
              <StatusDot
                kind={lora.busy ? 'warn' : lora.current ? 'ok' : 'idle'}
                pulse={lora.busy} />
              <div className="min-w-0">
                <div className="text-[13px] font-medium truncate">
                  {lora.current ? shortName(lora.current) : 'no adapter loaded'}
                </div>
                <div className="text-[11px] text-text-faint truncate">
                  {lora.pinned
                    ? `pinned: ${shortName(lora.pinned)}`
                    : 'auto-selected by mode'}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2 ml-auto flex-wrap">
              {lora.busy && <Tag kind="accent"><Spinner /> busy</Tag>}
              <span className="text-[12px] text-text-dim">mode</span>
              <Select value={curMode}
                onChange={(e) => setMode(e.target.value)}>
                {modes.map((m) => <option key={m} value={m}>{m}</option>)}
              </Select>
              <Btn variant="subtle" disabled={!!busy} onClick={doSetMode}>
                {busy === 'mode' ? <Spinner /> : null} Set
              </Btn>
              {lora.pinned && (
                <Btn variant="subtle" disabled={!!busy} onClick={doUnpin}>
                  {busy === 'unpin' ? <Spinner /> : <PinOff size={12} />} Unpin
                </Btn>
              )}
            </div>
          </div>
          {rec?.name && (
            <div className="mt-2.5 flex items-center gap-1.5 text-[11.5px] text-text-dim">
              <Sparkles size={11} className="text-accent shrink-0" />
              recommended for <span className="text-text">{curMode}</span>:
              <span className="text-text">{rec.name}</span>
              {rec.path && rec.path !== lora.current && (
                <button
                  className="text-accent-hi hover:underline cursor-pointer"
                  onClick={() => setSelPath(rec.path ?? null)}>
                  select
                </button>
              )}
            </div>
          )}
          {err && (
            <div className="mt-2 text-[12px] text-err bg-err/10 border border-err/30 rounded-lg px-3 py-1.5">
              {err}
            </div>
          )}
        </Card>

        <Card title={`Adapters (${filteredAdapters.length}${adapters.length !== filteredAdapters.length ? ` / ${adapters.length}` : ''})`}>
          <SearchInput
            value={search}
            onChange={setSearch}
            placeholder="Search adapters by name, path, or category…"
            className="mb-3"
          />
          {filteredAdapters.length ? (
            <div className="overflow-x-auto -mx-1">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="text-left text-text-faint">
                    <th className="px-1.5 py-1 font-medium">name</th>
                    <th className="px-1.5 py-1 font-medium">rank</th>
                    <th className="px-1.5 py-1 font-medium">params</th>
                    <th className="px-1.5 py-1 font-medium">size</th>
                    <th className="px-1.5 py-1 font-medium">category</th>
                    <th className="px-1.5 py-1 font-medium">modified</th>
                    <th className="px-1.5 py-1 font-medium text-right">actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAdapters.map((a: LoraRec) => {
                    const open = selPath === a.path
                    const pinned = !!a.path && lora.pinned === a.path
                    const loaded = !!a.path && lora.current === a.path
                    return (
                      <tr key={a.path ?? a.name}
                        onClick={() => setSelPath(open ? null : (a.path ?? null))}
                        className={clsx(
                          'border-t border-border cursor-pointer transition-colors',
                          open ? 'bg-accent/10' : 'hover:bg-panel-alt')}>
                        <td className="px-1.5 py-1.5 max-w-[220px]">
                          <div className="truncate text-text">
                            {a.name ?? shortName(a.path ?? '?')}
                          </div>
                          {a.header_error && (
                            <div className="text-[10.5px] text-err truncate"
                              title={a.header_error}>
                              {a.header_error}
                            </div>
                          )}
                        </td>
                        <td className="px-1.5 py-1.5 text-text-dim">
                          {a.rank ?? '—'}
                        </td>
                        <td className="px-1.5 py-1.5 text-text-dim">
                          {fmtParams(a.n_params)}
                        </td>
                        <td className="px-1.5 py-1.5 text-text-dim whitespace-nowrap">
                          {a.size_label ?? '—'}
                        </td>
                        <td className="px-1.5 py-1.5">
                          {a.category
                            ? <Tag kind={a.category === 'uncategorized' ? 'idle' : 'accent'}>{a.category}</Tag>
                            : '—'}
                        </td>
                        <td className="px-1.5 py-1.5 text-text-dim whitespace-nowrap">
                          {fmtDate(a.modified)}
                        </td>
                        <td className="px-1.5 py-1.5"
                          onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center gap-1 justify-end">
                            {loaded && <Tag kind="ok">loaded</Tag>}
                            <IconBtn title="Load on engine"
                              disabled={!ready || !!busy || !a.path}
                              onClick={() => {
                                setSelPath(a.path ?? null)
                                if (a.path) void loadPath(a.path)
                              }}>
                              <Upload size={12} />
                            </IconBtn>
                            {pinned ? (
                              <IconBtn title="Unpin" className="!text-warn"
                                disabled={!!busy} onClick={doUnpin}>
                                <PinOff size={12} />
                              </IconBtn>
                            ) : (
                              <IconBtn title="Pin adapter"
                                disabled={!!busy || !a.path}
                                onClick={() => a.path && doPin(a.path)}>
                                <Pin size={12} />
                              </IconBtn>
                            )}
                            <IconBtn title="Select for merge"
                              className={open ? '!text-accent-hi' : undefined}
                              onClick={() => setSelPath(open ? null : (a.path ?? null))}>
                              <GitMerge size={12} />
                            </IconBtn>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState icon={<Layers size={32} strokeWidth={1.2} />}
              title="No adapters"
              desc="Train a LoRA on the Fine-Tune page — adapters are scanned from research/checkpoints." />
          )}
        </Card>
      </div>

      {/* ---- right: load + merge + engine info ---- */}
      <div className="w-[330px] shrink-0 border-l border-border overflow-y-auto p-3 space-y-3">
        <Card title="Load adapter">
          <div className="space-y-2.5">
            <div className="font-mono text-[11px] break-all bg-input border border-border rounded-lg px-2.5 py-1.5 min-h-[30px]">
              {sel
                ? <span className="text-text">{sel.path}</span>
                : <span className="text-text-faint">select an adapter…</span>}
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <NumInput label="rank" value={rank} onChange={setRank}
                min={1} max={512} step={1} />
              <Select value={targetKey}
                onChange={(e) => setTargetKey(e.target.value)}
                title="target modules">
                {targets.map((t) => <option key={t} value={t}>{t}</option>)}
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <Check label="alpha" checked={alphaOn} onChange={setAlphaOn} />
              {alphaOn
                ? <NumInput value={alpha} onChange={setAlpha} min={1} max={1024} step={1} />
                : <span className="text-[11px] text-text-faint">engine default</span>}
            </div>
            <div className="flex gap-2">
              <Btn variant="primary" className="flex-1 justify-center"
                disabled={!ready || !sel?.path || !!busy} onClick={doLoad}
                title={ready ? 'load on resident engine' : 'engine not ready'}>
                {busy === 'load' ? <Spinner /> : <Upload size={12} />} Load
              </Btn>
              <Btn variant="subtle" className="flex-1 justify-center"
                disabled={!ready || !lora.current || !!busy} onClick={doUnload}>
                {busy === 'unload' ? <Spinner /> : <Square size={12} />} Unload
              </Btn>
            </div>
            {!ready && (
              <div className="text-[11px] text-text-faint">
                load/unload needs a resident engine.
              </div>
            )}
          </div>
        </Card>

        <Card title="Merge into base">
          <div className="space-y-2.5">
            <div>
              <div className="text-[11px] text-text-faint mb-1">adapter</div>
              <div className="text-[12px] text-text truncate">
                {sel ? (sel.name ?? shortName(sel.path ?? '')) : '— select a row —'}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[12px] text-text-dim shrink-0">base</span>
              <Select value={mergeBase} className="flex-1"
                onChange={(e) => {
                  setMergeBase(e.target.value)
                  const m = models.find((mm) => mm.path === e.target.value)
                  if (m?.config_name) setMergeCfg(m.config_name)
                }}>
                <option value="">— checkpoint —</option>
                {models.map((m) => (
                  <option key={m.path} value={m.path}>{m.name}</option>
                ))}
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[12px] text-text-dim shrink-0">config</span>
              <Input value={mergeCfg} onChange={(e) => setMergeCfg(e.target.value)}
                className="flex-1 !py-1 !text-[11.5px] font-mono" />
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <NumInput label="rank" value={mergeRank} onChange={setMergeRank}
                min={1} max={512} step={1} />
              <Check label="alpha" checked={mergeAlphaOn} onChange={setMergeAlphaOn} />
              {mergeAlphaOn && (
                <NumInput value={mergeAlpha} onChange={setMergeAlpha}
                  min={1} max={1024} step={1} />
              )}
            </div>
            <Input value={mergeOut} onChange={(e) => setMergeOut(e.target.value)}
              placeholder="out checkpoint path"
              className="w-full !text-[11.5px] font-mono" />
            <Btn variant="primary" className="w-full justify-center"
              disabled={!sel?.path || !mergeBase || !!busy} onClick={doMerge}>
              {busy === 'merge' ? <Spinner /> : <GitMerge size={12} />}
              Merge (CPU)
            </Btn>
            {mergeRes && (
              <pre className="font-mono text-[11px] text-text-dim whitespace-pre-wrap max-h-32 overflow-y-auto bg-deep rounded-lg p-2.5 border border-border">
                {mergeRes}
              </pre>
            )}
          </div>
        </Card>

        <Card title="Engine LoRA info" actions={
          <IconBtn title="Refresh" onClick={loadInfo}><RefreshCw size={12} /></IconBtn>
        }>
          {info && Object.keys(info).length ? (
            Object.entries(info).slice(0, 40).map(([k, v]) => (
              <KvRow key={k} k={k}
                v={v == null ? '—'
                  : typeof v === 'object' ? JSON.stringify(v) : String(v)}
                mono={typeof v === 'string' && v.length > 28} />
            ))
          ) : (
            <div className="text-[12px] text-text-faint py-1">
              {ready ? 'no adapter info' : 'engine not loaded'}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
