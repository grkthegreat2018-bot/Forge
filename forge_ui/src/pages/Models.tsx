// Models — checkpoint index, config presets, engine load/boot, merging.

import { clsx } from 'clsx'
import {
  Box, ChevronDown, ChevronRight, Dna, GitMerge, Play,
  RefreshCw, Square, Trash2, Upload,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import type { ModelEntry } from '../lib/types'
import {
  Btn, Card, Check, EmptyState, Input, KvRow, NumInput,
  SearchInput, Select, Spinner, StatusDot, Tag,
} from '../components/ui'

type ConfigRec = Record<string, unknown>

interface BootResult {
  ok: boolean
  output?: string
  tokens?: number
  time_s?: number
  tps?: number
  error?: string
}

interface OpResult {
  ok: boolean
  result?: unknown
  error?: string
}

const MERGE_METHODS = ['blockwise_crossover', 'linear', 'slerp', 'evolve']

// preferred column order for the config preset table — anything else
// scalar that shows up in the payload gets appended after these
const CFG_PREFERRED = [
  'name', 'params_label', 'd_model', 'n_layers', 'n_heads',
  'n_kv_heads', 'vocab_size', 'attn_type', 'ffn_type', 'max_seq_len',
]

function fmtVal(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function fmtDate(ts: number): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleDateString(
    undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

function shortName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

export default function Models() {
  const engine = useForge((s) => s.engine)
  const progress = useForge((s) => s.engineProgress)
  const bootEvt = useForge((s) => s.boot)

  const [models, setModels] = useState<ModelEntry[]>([])
  const [configs, setConfigs] = useState<ConfigRec[]>([])
  const [search, setSearch] = useState('')
  const [selPath, setSelPath] = useState<string | null>(null)
  const [err, setErr] = useState('')

  // load panel
  const [loadCfg, setLoadCfg] = useState('')
  const [useCompile, setUseCompile] = useState(false)

  // boot & test
  const [booting, setBooting] = useState(false)
  const [bootPrompt, setBootPrompt] = useState('def fibonacci(n):')
  const [bootOut, setBootOut] = useState<BootResult | null>(null)

  // merge panel
  const [parents, setParents] = useState<Set<string>>(new Set())
  const [method, setMethod] = useState('blockwise_crossover')
  const [outPath, setOutPath] = useState('')
  const [hotSwap, setHotSwap] = useState(false)
  const [mergeBusy, setMergeBusy] = useState('')
  const [mergeRes, setMergeRes] = useState('')
  const [showEvolve, setShowEvolve] = useState(false)
  const [generations, setGenerations] = useState(4)
  const [population, setPopulation] = useState(6)
  const [elitism, setElitism] = useState(2)
  const [crossover, setCrossover] = useState(0.5)
  const [mutation, setMutation] = useState(0.1)
  const [mutRate, setMutRate] = useState(0.05)
  const [benchPrompt, setBenchPrompt] = useState('def sort_list(x):')
  const [benchTokens, setBenchTokens] = useState(32)

  const refresh = () => {
    api.get<{ models: ModelEntry[] }>('/api/models')
      .then((r) => setModels(r.models))
      .catch((e) => setErr(String(e)))
    api.get<{ configs: ConfigRec[] }>('/api/models/configs')
      .then((r) => setConfigs(r.configs))
      .catch(() => undefined)
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { refresh() }, [])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return models
    return models.filter((m) =>
      m.name.toLowerCase().includes(q) || m.path.toLowerCase().includes(q))
  }, [models, search])

  const sel = models.find((m) => m.path === selPath) ?? null

  const cfgNames = useMemo(() => {
    const out: string[] = []
    for (const c of configs) {
      const n = c.name
      if (typeof n === 'string' && n && !out.includes(n)) out.push(n)
    }
    return out
  }, [configs])

  const cfgCols = useMemo(() => {
    const scalar = (v: unknown) =>
      v == null || typeof v === 'string' || typeof v === 'number'
      || typeof v === 'boolean'
    const keys = CFG_PREFERRED.filter((k) =>
      configs.some((c) => c[k] !== undefined))
    for (const c of configs) {
      for (const k of Object.keys(c)) {
        if (!keys.includes(k) && scalar(c[k])) keys.push(k)
      }
    }
    return keys.slice(0, 7)
  }, [configs])

  const engKind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  const toggleRow = (m: ModelEntry) => {
    if (selPath === m.path) { setSelPath(null); return }
    setSelPath(m.path)
    if (m.config_name) setLoadCfg(m.config_name)
  }

  const toggleParent = (path: string) => {
    setParents((s) => {
      const n = new Set(s)
      if (n.has(path)) n.delete(path); else n.add(path)
      return n
    })
  }

  const cfgFor = (m: ModelEntry) => loadCfg || m.config_name || 'forgelm_v2'

  const doLoad = async () => {
    if (!sel) return
    setErr('')
    try {
      await api.post('/api/engine/load', {
        checkpoint: sel.path,
        config_name: cfgFor(sel),
        use_compile: useCompile,
      })
      useForge.getState().refreshStatus().catch(() => undefined)
    } catch (e) { setErr(String(e)) }
  }

  const doUnload = () => {
    api.post('/api/engine/unload')
      .then(() => useForge.getState().refreshStatus())
      .catch((e) => setErr(String(e)))
  }

  const doReactivate = () => {
    api.post('/api/engine/reactivate')
      .then(() => useForge.getState().refreshStatus())
      .catch((e) => setErr(String(e)))
  }

  const doBoot = async () => {
    if (!sel || booting) return
    setBooting(true)
    setBootOut(null)
    setErr('')
    try {
      const r = await api.post<BootResult>('/api/models/boot', {
        checkpoint: sel.path,
        config_name: cfgFor(sel),
        prompt: bootPrompt,
        max_tokens: 64,
      })
      setBootOut(r)
      if (!r.ok) setErr(r.error ?? 'boot test failed')
      useForge.getState().refreshStatus().catch(() => undefined)
    } catch (e) { setErr(String(e)) } finally { setBooting(false) }
  }

  const cancelBoot = () =>
    api.post('/api/models/boot/cancel').catch(() => undefined)

  const doDelete = async (m: ModelEntry) => {
    if (!window.confirm(`Delete checkpoint ${m.name}?\n${m.path}`)) return
    try {
      const r = await api.del<OpResult>(
        `/api/models?path=${encodeURIComponent(m.path)}`)
      if (r.ok === false) { setErr(r.error ?? 'delete failed'); return }
      if (selPath === m.path) setSelPath(null)
      setParents((s) => { const n = new Set(s); n.delete(m.path); return n })
      refresh()
    } catch (e) { setErr(String(e)) }
  }

  const doMerge = async () => {
    if (parents.size < 2 || mergeBusy) return
    setMergeBusy('merge')
    setMergeRes('')
    setErr('')
    try {
      const r = await api.post<OpResult>('/api/engine/merge', {
        parents: [...parents],
        method,
        out_path: outPath,
        hot_swap: hotSwap,
      })
      setMergeRes(r.ok
        ? JSON.stringify(r.result ?? { ok: true }, null, 2)
        : `error: ${r.error ?? 'merge failed'}`)
      useForge.getState().refreshStatus().catch(() => undefined)
      refresh()
    } catch (e) { setErr(String(e)) } finally { setMergeBusy('') }
  }

  const doEvolve = async () => {
    if (parents.size < 2 || mergeBusy) return
    setMergeBusy('evolve')
    setMergeRes('')
    setErr('')
    try {
      const r = await api.post<OpResult>('/api/engine/evolve_merge', {
        parents: [...parents],
        out_path: outPath,
        generations, population, elitism, crossover,
        mutation, mut_rate: mutRate,
        bench_prompt: benchPrompt, bench_tokens: benchTokens,
      })
      setMergeRes(r.ok
        ? JSON.stringify(r.result ?? { ok: true }, null, 2)
        : `error: ${r.error ?? 'evolve merge failed'}`)
      useForge.getState().refreshStatus().catch(() => undefined)
      refresh()
    } catch (e) { setErr(String(e)) } finally { setMergeBusy('') }
  }

  return (
    <div className="flex h-full min-h-0">
      {/* ---- left: engine strip + checkpoint list ---- */}
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border">
          <StatusDot kind={engKind} pulse={engine.state === 'loading'} />
          <span className="text-[12.5px] capitalize font-medium">
            {engine.state}
          </span>
          {engine.state === 'ready' && (
            <span className="text-[12px] text-text-dim font-mono truncate">
              {engine.info.checkpoint
                ? shortName(engine.info.checkpoint) : 'resident'}
            </span>
          )}
          {engine.state === 'loading' && (
            <span className="text-[12px] text-warn truncate">
              {progress ?? 'loading weights…'}
            </span>
          )}
          {engine.state === 'error' && (
            <span className="text-[12px] text-err truncate">{engine.error}</span>
          )}
          {engine.busy && <Tag kind="accent">busy</Tag>}
          <div className="ml-auto shrink-0 flex gap-1.5">
            {engine.state === 'ready' && (
              <>
                <Btn variant="subtle" onClick={doReactivate}>
                  <RefreshCw size={11} /> Reactivate
                </Btn>
                <Btn variant="subtle" onClick={doUnload}>
                  <Square size={11} /> Unload
                </Btn>
              </>
            )}
          </div>
        </div>

        <div className="px-4 pt-3 pb-2">
          <SearchInput
            value={search}
            onChange={setSearch}
            placeholder="Search checkpoints by name or path…"
            className="w-full"
          />
        </div>

        <div className="flex-1 overflow-y-auto px-4 pb-4">
          {err && (
            <div className="text-[12px] text-err bg-err/10 border border-err/30 rounded-lg px-3 py-1.5 mb-2">
              {err}
            </div>
          )}

          {(booting || bootOut) && (
            <Card title="Boot test" className="mb-2" actions={booting && (
              <Btn variant="danger" onClick={cancelBoot}>
                <Square size={11} /> Cancel
              </Btn>
            )}>
              {booting && (
                <div className="flex items-center gap-2 text-[12px] text-text-dim">
                  <Spinner /> {bootEvt.status || 'running boot test…'}
                </div>
              )}
              {bootOut && (
                <>
                  <div className="text-[11px] text-text-faint mb-1.5">
                    {bootOut.ok
                      ? `${bootOut.tokens ?? '?'} tokens · ${bootOut.time_s ?? '?'}s · ${bootOut.tps ?? '?'} tok/s`
                      : `failed: ${bootOut.error ?? 'unknown'}`}
                  </div>
                  {bootOut.output && (
                    <pre className="font-mono text-[11.5px] text-text-dim whitespace-pre-wrap max-h-44 overflow-y-auto bg-deep rounded-lg p-2.5 border border-border">
                      {bootOut.output}
                    </pre>
                  )}
                </>
              )}
            </Card>
          )}

          {filtered.map((m) => {
            const open = selPath === m.path
            return (
              <div key={m.path}
                className={clsx(
                  'border rounded-lg mb-1.5 transition-colors',
                  open
                    ? 'border-accent/40 bg-panel-alt'
                    : 'border-border bg-panel hover:border-border-hi')}>
                <button
                  className="w-full flex items-center gap-2.5 px-3 py-2 text-left cursor-pointer"
                  onClick={() => toggleRow(m)}>
                  {open
                    ? <ChevronDown size={13} className="text-text-faint shrink-0" />
                    : <ChevronRight size={13} className="text-text-faint shrink-0" />}
                  <span className="text-[13px] truncate flex-1">{m.name}</span>
                  {m.is_lora && <Tag kind="think">lora</Tag>}
                  {m.config_name && <Tag kind="accent">{m.config_name}</Tag>}
                  <span className="text-[11px] text-text-faint w-16 text-right shrink-0">
                    {m.size_label}
                  </span>
                  <span className="text-[11px] text-text-faint w-[76px] text-right shrink-0">
                    {fmtDate(m.modified)}
                  </span>
                </button>
                {open && (
                  <div className="px-3 pb-3 pt-2 border-t border-border/60 fade-in">
                    <div className="font-mono text-[11px] text-text-dim break-all mb-2">
                      {m.path}
                    </div>
                    <div className="grid grid-cols-2 gap-x-6">
                      <KvRow k="size" v={`${m.size_label} (${m.ext})`} />
                      <KvRow k="modified"
                        v={new Date(m.modified * 1000).toLocaleString()} />
                      <KvRow k="safetensors" v={m.is_safetensors ? 'yes' : 'no'} />
                      <KvRow k="config" v={m.config_name ?? '—'} />
                    </div>
                    {m.meta && Object.keys(m.meta).length > 0 && (
                      <div className="mt-1.5">
                        <div className="text-[10.5px] font-semibold uppercase tracking-wider text-text-faint mb-0.5">
                          meta
                        </div>
                        <div className="grid grid-cols-2 gap-x-6">
                          {Object.entries(m.meta).slice(0, 12).map(([k, v]) => (
                            <KvRow key={k} k={k} v={fmtVal(v)} />
                          ))}
                        </div>
                      </div>
                    )}
                    <div className="flex items-center gap-2 mt-2.5">
                      <Btn variant="primary" onClick={doLoad}
                        disabled={engine.state === 'loading'}>
                        {engine.state === 'loading' ? <Spinner /> : <Upload size={12} />}
                        Load
                      </Btn>
                      <Btn variant="subtle" onClick={doBoot} disabled={booting}>
                        {booting ? <Spinner /> : <Play size={12} />} Boot test
                      </Btn>
                      <Input value={bootPrompt}
                        onChange={(e) => setBootPrompt(e.target.value)}
                        placeholder="boot prompt"
                        className="flex-1 !py-1 !text-[11.5px] font-mono" />
                      <Btn variant="danger" className="ml-auto shrink-0"
                        onClick={() => doDelete(m)}>
                        <Trash2 size={12} /> Delete
                      </Btn>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
          {!filtered.length && (
            <EmptyState icon={<Box size={36} strokeWidth={1.2} />}
              title="No checkpoints"
              desc={search
                ? 'Nothing matches the search.'
                : 'Drop .safetensors files under research/checkpoints.'} />
          )}
        </div>
      </div>

      {/* ---- right: load / presets / merge ---- */}
      <div className="w-[340px] shrink-0 border-l border-border overflow-y-auto p-3 space-y-3">
        <Card title="Load checkpoint">
          <div className="space-y-2.5">
            <div className="font-mono text-[11px] break-all bg-input border border-border rounded-lg px-2.5 py-1.5 min-h-[30px]">
              {sel
                ? <span className="text-text">{sel.path}</span>
                : <span className="text-text-faint">select a checkpoint…</span>}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[12px] text-text-dim shrink-0">config</span>
              <Select value={loadCfg}
                onChange={(e) => setLoadCfg(e.target.value)} className="flex-1">
                <option value="">— auto —</option>
                {cfgNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </Select>
            </div>
            <Check label="use compile (torch.compile)"
              checked={useCompile} onChange={setUseCompile} />
            <Btn variant="primary" className="w-full justify-center"
              disabled={!sel || engine.state === 'loading'} onClick={doLoad}>
              {engine.state === 'loading' ? <Spinner /> : <Upload size={13} />}
              Load into engine
            </Btn>
          </div>
        </Card>

        <Card title="Config presets">
          {configs.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr>
                    {cfgCols.map((k) => (
                      <th key={k}
                        className="text-left font-medium text-text-faint px-1 py-1 whitespace-nowrap">
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {configs.map((c, i) => (
                    <tr key={i}
                      className="border-t border-border hover:bg-panel-alt cursor-pointer"
                      title="click to use for load"
                      onClick={() => {
                        if (typeof c.name === 'string') setLoadCfg(c.name)
                      }}>
                      {cfgCols.map((k) => (
                        <td key={k}
                          className={clsx('px-1 py-1.5 whitespace-nowrap',
                            k === 'name' ? 'text-accent-hi' : 'text-text-dim')}>
                          {fmtVal(c[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-[12px] text-text-faint py-1">no presets</div>
          )}
        </Card>

        <Card title="Merge checkpoints">
          <div className="space-y-2.5">
            <div className="max-h-44 overflow-y-auto border border-border rounded-lg divide-y divide-border">
              {models.map((m) => (
                <label key={m.path}
                  className="flex items-center gap-2 px-2.5 py-1.5 text-[12px] text-text-dim cursor-pointer hover:bg-panel-alt">
                  <input type="checkbox" checked={parents.has(m.path)}
                    onChange={() => toggleParent(m.path)}
                    className="w-3.5 h-3.5 rounded accent-accent cursor-pointer" />
                  <span className="truncate flex-1">{m.name}</span>
                  <span className="text-[10px] text-text-faint">{m.size_label}</span>
                </label>
              ))}
              {!models.length && (
                <div className="text-[11.5px] text-text-faint px-2.5 py-3 text-center">
                  no checkpoints
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[12px] text-text-dim shrink-0">method</span>
              <Select value={method} onChange={(e) => setMethod(e.target.value)}
                className="flex-1">
                {MERGE_METHODS.map((mm) => (
                  <option key={mm} value={mm}>{mm}</option>
                ))}
              </Select>
            </div>
            <Input value={outPath} onChange={(e) => setOutPath(e.target.value)}
              placeholder="out path (optional)"
              className="w-full !text-[11.5px] font-mono" />
            <Check label="hot swap into resident engine"
              checked={hotSwap} onChange={setHotSwap} />
            <div className="flex gap-2">
              <Btn variant="primary" className="flex-1 justify-center"
                disabled={parents.size < 2 || !!mergeBusy} onClick={doMerge}>
                {mergeBusy === 'merge' ? <Spinner /> : <GitMerge size={12} />}
                Merge
              </Btn>
              <Btn variant="subtle" className="flex-1 justify-center"
                disabled={parents.size < 2 || !!mergeBusy} onClick={doEvolve}>
                {mergeBusy === 'evolve' ? <Spinner /> : <Dna size={12} />}
                Evolve
              </Btn>
            </div>
            <button onClick={() => setShowEvolve((v) => !v)}
              className="flex items-center gap-1 text-[11px] text-text-faint hover:text-text-dim cursor-pointer">
              {showEvolve ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
              evolve settings
            </button>
            {showEvolve && (
              <div className="bg-panel-alt border border-border rounded-lg p-2.5 space-y-2 fade-in">
                <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                  <NumInput label="gens" value={generations}
                    onChange={setGenerations} min={1} max={64} step={1} />
                  <NumInput label="pop" value={population}
                    onChange={setPopulation} min={2} max={64} step={1} />
                  <NumInput label="elitism" value={elitism}
                    onChange={setElitism} min={0} max={16} step={1} />
                  <NumInput label="xover" value={crossover}
                    onChange={setCrossover} min={0} max={1} step={0.05} />
                  <NumInput label="mut" value={mutation}
                    onChange={setMutation} min={0} max={1} step={0.05} />
                  <NumInput label="mut rate" value={mutRate}
                    onChange={setMutRate} min={0} max={1} step={0.01} />
                </div>
                <Input value={benchPrompt}
                  onChange={(e) => setBenchPrompt(e.target.value)}
                  placeholder="bench prompt"
                  className="w-full !py-1 !text-[11.5px] font-mono" />
                <NumInput label="bench tok" value={benchTokens}
                  onChange={setBenchTokens} min={1} max={512} step={8} />
              </div>
            )}
            {mergeRes && (
              <pre className="font-mono text-[11px] text-text-dim whitespace-pre-wrap max-h-48 overflow-y-auto bg-deep rounded-lg p-2.5 border border-border">
                {mergeRes}
              </pre>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
