// Training — live run telemetry: status.json runs, loss chart, log tail.
// Primary feed is the hub store (WS); a 3s REST poll is the fallback.

import { clsx } from 'clsx'
import { Check as CheckIcon, Copy, RefreshCw, Square, TrendingUp, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import type { RunSnapshot } from '../lib/types'
import { LineChart } from '../components/charts'
import {
  Btn, Card, EmptyState, IconBtn, KvRow, Progress, SearchInput, StatusDot, Tag,
} from '../components/ui'

type TagKind = 'ok' | 'warn' | 'err' | 'idle' | 'accent' | 'think'

const statusKind = (r: RunSnapshot): TagKind => {
  if (r.is_live) return 'accent'
  const s = r.status.toLowerCase()
  if (/done|finish|complete/.test(s)) return 'ok'
  if (/err|fail|crash/.test(s)) return 'err'
  if (/stall|stop|kill/.test(s)) return 'warn'
  return 'idle'
}

const fmtScalar = (v: unknown): string => {
  if (v == null) return '—'
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'number') {
    return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/\.?0+$/, '')
  }
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

const fmtTime = (ts?: number) => {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

const GRID =
  'grid grid-cols-[16px_minmax(0,1fr)_80px_84px_64px_66px_52px_48px_28px] items-center gap-x-2'

export default function Training() {
  const storeRuns = useForge((s) => s.runs)
  const refreshStatus = useForge((s) => s.refreshStatus)
  const [runs, setRuns] = useState<RunSnapshot[]>(storeRuns)
  const [runSearch, setRunSearch] = useState('')
  const [copiedLog, setCopiedLog] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [logLines, setLogLines] = useState<string[]>([])
  const [logErr, setLogErr] = useState('')
  const [error, setError] = useState('')
  const seenRef = useRef(new Map<string, RunSnapshot>())
  const histRef = useRef<Record<string, number[]>>({})
  const logRef = useRef<HTMLDivElement>(null)

  // Merge a fresh snapshot list: remember every run ever seen by status_file
  // so a run that vanishes (finished, file cleaned up) stays listed.
  const merge = useCallback((list: RunSnapshot[]) => {
    const seen = seenRef.current
    for (const r of list) {
      seen.set(r.status_file, r)
      if (Number.isFinite(r.loss)) {
        const h = (histRef.current[r.id] ??= [])
        if (h[h.length - 1] !== r.loss) {
          h.push(r.loss)
          if (h.length > 600) h.splice(0, h.length - 600)
        }
      }
    }
    const present = new Set(list.map((r) => r.status_file))
    const gone: RunSnapshot[] = []
    for (const [sf, prev] of seen) {
      if (!present.has(sf)) {
        gone.push({
          ...prev, is_live: false,
          status: prev.is_live || /run|train/i.test(prev.status)
            ? 'finished' : prev.status,
        })
      }
    }
    setRuns([...list, ...gone])
  }, [])

  useEffect(() => { merge(storeRuns) }, [storeRuns, merge])

  useEffect(() => {
    refreshStatus().catch(() => undefined)
    let dead = false
    const poll = async () => {
      try {
        const r = await api.get<{ runs: RunSnapshot[] }>('/api/runs')
        if (!dead) merge(r.runs)
      } catch { /* WS-fed store is the fallback source */ }
    }
    poll()
    const t = setInterval(poll, 3000)
    return () => { dead = true; clearInterval(t) }
  }, [merge, refreshStatus])

  const fetchLog = useCallback(async (sf: string) => {
    try {
      const r = await api.get<{ lines: string[] }>(
        `/api/runs/log_tail?status_file=${encodeURIComponent(sf)}&lines=200`)
      setLogLines(r.lines)
      setLogErr('')
    } catch (e) { setLogErr(String(e)) }
  }, [])

  // auto-refresh the log tail while a run is selected
  useEffect(() => {
    if (!selected) { setLogLines([]); setLogErr(''); return }
    fetchLog(selected)
    const t = setInterval(() => fetchLog(selected), 3000)
    return () => clearInterval(t)
  }, [selected, fetchLog])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [logLines])

  const stopRun = async (sf: string) => {
    setError('')
    try {
      await api.post('/api/runs/stop', { status_file: sf })
      await refreshStatus()
    } catch (e) { setError(String(e)) }
  }

  const filteredRuns = useMemo(() => {
    const q = runSearch.trim().toLowerCase()
    if (!q) return runs
    return runs.filter((r) =>
      r.name.toLowerCase().includes(q) ||
      r.status.toLowerCase().includes(q) ||
      (r.method && r.method.toLowerCase().includes(q))
    )
  }, [runs, runSearch])

  const sel = runs.find((r) => r.status_file === selected) ?? null
  const extras = sel
    ? Object.entries(sel.extra ?? {}).filter(([, v]) =>
        v == null || ['string', 'number', 'boolean'].includes(typeof v))
    : []
  const lossHist = sel ? (histRef.current[sel.id] ?? []) : []

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <Card title={`Training runs (${filteredRuns.length}${runs.length !== filteredRuns.length ? ` / ${runs.length}` : ''})`} actions={
        <div className="flex items-center gap-2">
          <SearchInput
            value={runSearch}
            onChange={setRunSearch}
            placeholder="Filter runs…"
            className="w-48 !text-[11.5px]"
          />
          <IconBtn title="Refresh"
            onClick={() => refreshStatus().catch(() => undefined)}>
            <RefreshCw size={12} />
          </IconBtn>
        </div>
      }>
        {error && <div className="text-[11.5px] text-err mb-1">{error}</div>}
        {filteredRuns.length ? (
          <div>
            <div className={clsx(GRID,
              'px-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-text-faint')}>
              <span /><span>run</span><span>status</span>
              <span className="text-right">step</span>
              <span className="text-right">loss</span>
              <span className="text-right">lr</span>
              <span className="text-right">vram</span>
              <span className="text-right">hb</span>
              <span />
            </div>
            {filteredRuns.map((r) => (
              <div
                key={r.status_file}
                onClick={() => setSelected(
                  selected === r.status_file ? null : r.status_file)}
                className={clsx(GRID,
                  'w-full px-2 py-1.5 rounded-lg cursor-pointer transition-colors',
                  selected === r.status_file
                    ? 'bg-accent/10 border border-accent/25'
                    : 'hover:bg-panel-alt border border-transparent')}>
                <StatusDot kind={r.is_live ? 'ok' : 'idle'} pulse={r.is_live} />
                <span className="text-[12.5px] truncate" title={r.name}>
                  {r.name}
                </span>
                <Tag kind={statusKind(r)}>{r.status}</Tag>
                <span className="text-[11.5px] font-mono text-right text-text-dim">
                  {r.step}/{r.max_steps}
                </span>
                <span className="text-[11.5px] font-mono text-right text-text-dim">
                  {r.loss ? r.loss.toFixed(3) : '—'}
                </span>
                <span className="text-[11.5px] font-mono text-right text-text-dim">
                  {r.lr ? r.lr.toExponential(1) : '—'}
                </span>
                <span className="text-[11.5px] font-mono text-right text-text-dim">
                  {r.vram_gb ? r.vram_gb.toFixed(1) : '—'}
                </span>
                <span className="text-[11.5px] font-mono text-right text-text-faint">
                  {r.heartbeat_age_s >= 0 ? `${Math.round(r.heartbeat_age_s)}s` : '—'}
                </span>
                <span onClick={(e) => e.stopPropagation()}>
                  {r.is_live && (
                    <IconBtn title="Stop run"
                      className="!w-6 !h-6 hover:!text-err"
                      onClick={() => stopRun(r.status_file)}>
                      <Square size={11} />
                    </IconBtn>
                  )}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            icon={<TrendingUp size={34} strokeWidth={1.2} />}
            title="No training runs"
            desc="status.json telemetry from sft_train / self-play appears here while a run is active." />
        )}
      </Card>

      {sel && (
        <Card
          title={
            <span className="flex items-center gap-2">
              <StatusDot kind={sel.is_live ? 'ok' : 'idle'} pulse={sel.is_live} />
              <span className="text-[12px] font-semibold text-text normal-case tracking-normal">
                {sel.name}
              </span>
              <Tag kind={statusKind(sel)} className="normal-case">{sel.status}</Tag>
            </span>
          }
          actions={
            <div className="flex items-center gap-1">
              {sel.is_live && (
                <Btn variant="danger" className="!py-1"
                  onClick={() => stopRun(sel.status_file)}>
                  <Square size={11} /> Stop
                </Btn>
              )}
              <IconBtn title="Close" onClick={() => setSelected(null)}>
                <X size={13} />
              </IconBtn>
            </div>
          }>
          <div className="mb-3">
            <div className="flex justify-between text-[10.5px] text-text-faint mb-1">
              <span>{sel.method || 'training'}</span>
              <span>
                {sel.progress_pct.toFixed(0)}% · {sel.step}/{sel.max_steps}
              </span>
            </div>
            <Progress value={sel.progress_pct} />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <KvRow k="status file" v={sel.status_file} mono />
              <KvRow k="method" v={sel.method || '—'} />
              <KvRow k="vram"
                v={sel.vram_gb ? `${sel.vram_gb.toFixed(1)} GB` : '—'} />
              <KvRow k="heartbeat"
                v={sel.heartbeat_age_s >= 0
                  ? `${sel.heartbeat_age_s.toFixed(0)}s ago` : '—'} />
              <KvRow k="updated" v={fmtTime(sel.updated_at)} />
              {extras.map(([k, v]) => <KvRow key={k} k={k} v={fmtScalar(v)} />)}
            </div>
            <div>
              {lossHist.length > 1 ? (
                <LineChart height={150} yLabel="loss"
                  series={[{ name: 'loss', color: '#8aa3ff', data: lossHist }]} />
              ) : (
                <div className="text-[11.5px] text-text-faint py-6 text-center">
                  loss history builds as the run reports
                </div>
              )}
            </div>
          </div>
          <div className="mt-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10.5px] font-semibold uppercase tracking-wider text-text-faint">
                Run Log Tail
              </span>
              <div className="flex items-center gap-1">
                {logLines.length > 0 && (
                  <button
                    onClick={() => {
                      void navigator.clipboard.writeText(logLines.join('\n')).then(() => {
                        setCopiedLog(true)
                        setTimeout(() => setCopiedLog(false), 1200)
                      })
                    }}
                    title="Copy log lines"
                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-panel-alt hover:bg-panel-hover border border-border text-text-dim hover:text-text cursor-pointer transition-colors">
                    {copiedLog ? <CheckIcon size={11} className="text-ok" /> : <Copy size={11} />}
                    {copiedLog ? 'Copied' : 'Copy'}
                  </button>
                )}
                <IconBtn title="Refresh log"
                  onClick={() => fetchLog(sel.status_file)}>
                  <RefreshCw size={12} />
                </IconBtn>
              </div>
            </div>
            <div ref={logRef}
              className="h-52 overflow-y-auto bg-deep border border-border rounded-lg p-3">
              {logErr && <div className="text-[11.5px] text-err">{logErr}</div>}
              {!logLines.length && !logErr && (
                <div className="text-[11.5px] text-text-faint">no log lines</div>
              )}
              <pre className="text-[11px] font-mono text-text-dim whitespace-pre-wrap leading-relaxed">
                {logLines.join('\n')}
              </pre>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
