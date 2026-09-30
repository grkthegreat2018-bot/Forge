// Logs — unified log viewer across all discovered sources.
// Filters (query/levels/sources) map to /api/logs query params; the pane
// auto-scrolls unless the user scrolls up (follow is re-armed via button).

import { clsx } from 'clsx'
import {
  ArrowDownToLine, Check as CheckIcon, Copy, Download, RefreshCw, ScrollText, Trash2,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import {
  Btn, Check, EmptyState, IconBtn, SearchInput, Select, Tag,
} from '../components/ui'

const LEVELS = ['ERROR', 'WARN', 'INFO', 'DEBUG'] as const
const LEVEL_COLOR: Record<string, string> = {
  ERROR: 'text-err',
  WARN: 'text-warn',
  INFO: 'text-accent-hi',
  DEBUG: 'text-text-dim',
}
const MAX_SRC_CHIPS = 6

interface LogsResp {
  lines: string[]
  total: number
  errors: number
  warnings: number
  sources: string[]
}

function lineClass(l: string): string {
  if (l.includes('ERROR') || l.includes('FATAL')) return 'text-err'
  if (l.includes('WARN')) return 'text-warn'
  if (l.includes('DEBUG')) return 'text-text-faint'
  return 'text-text-dim'
}

export default function Logs() {
  const [query, setQuery] = useState('')
  const [levels, setLevels] = useState<Set<string>>(() => new Set(LEVELS))
  const [srcSel, setSrcSel] = useState<Set<string>>(() => new Set<string>())
  const [sources, setSources] = useState<string[]>([])
  const [lines, setLines] = useState<string[]>([])
  const [counts, setCounts] = useState({ total: 0, errors: 0, warnings: 0 })
  const [err, setErr] = useState('')
  const [auto, setAuto] = useState(true)
  const [following, setFollowing] = useState(true)
  const paneRef = useRef<HTMLDivElement>(null)
  const followRef = useRef(true)

  const load = useCallback(async () => {
    const params = new URLSearchParams()
    if (query.trim()) params.set('query', query.trim())
    if (levels.size === 0) params.set('levels', 'NONE')
    else if (levels.size < LEVELS.length)
      params.set('levels', [...levels].join(','))
    if (srcSel.size) params.set('sources', [...srcSel].join(','))
    params.set('limit', '3000')
    try {
      const r = await api.get<LogsResp>(`/api/logs?${params.toString()}`)
      setLines(r.lines)
      setCounts({ total: r.total, errors: r.errors, warnings: r.warnings })
      if (r.sources?.length) {
        setSources((prev) => {
          const s = new Set(prev)
          r.sources.forEach((x) => s.add(x))
          return [...s].sort()
        })
      }
      setErr('')
    } catch (e) { setErr(String(e)) }
  }, [query, levels, srcSel])

  // keep a stable handle for intervals
  const loadRef = useRef(load)
  useEffect(() => { loadRef.current = load }, [load])

  // discover sources once
  useEffect(() => {
    api.get<{ sources: string[] }>('/api/logs/sources')
      .then((r) => setSources(r.sources))
      .catch(() => undefined)
  }, [])

  // debounced auto-apply on any filter change (incl. first mount)
  useEffect(() => {
    const t = setTimeout(() => void loadRef.current(), 300)
    return () => clearTimeout(t)
  }, [load])

  // auto-refresh
  useEffect(() => {
    if (!auto) return
    const t = setInterval(() => void loadRef.current(), 3000)
    return () => clearInterval(t)
  }, [auto])

  // auto-scroll while following
  useEffect(() => {
    if (followRef.current) {
      paneRef.current?.scrollTo({ top: paneRef.current.scrollHeight })
    }
  }, [lines])

  const onScroll = () => {
    const el = paneRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    followRef.current = atBottom
    setFollowing(atBottom)
  }

  const resumeFollow = () => {
    followRef.current = true
    setFollowing(true)
    paneRef.current?.scrollTo({
      top: paneRef.current.scrollHeight, behavior: 'smooth' })
  }

  const toggleLevel = (lv: string) => setLevels((s) => {
    const n = new Set(s)
    if (n.has(lv)) n.delete(lv); else n.add(lv)
    return n
  })

  const toggleSource = (s: string) => setSrcSel((prev) => {
    const n = new Set(prev)
    if (n.has(s)) n.delete(s); else n.add(s)
    return n
  })

  const clear = async () => {
    try {
      await api.post('/api/logs/clear')
      await load()
    } catch (e) { setErr(String(e)) }
  }

  const [copied, setCopied] = useState(false)
  const copyLogs = () => {
    if (!lines.length) return
    void navigator.clipboard.writeText(lines.join('\n')).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    })
  }

  const downloadLogs = () => {
    if (!lines.length) return
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `forge_logs_${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const filtered = query.trim() !== '' || levels.size < LEVELS.length
    || srcSel.size > 0

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* toolbar */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 px-3 py-2 border-b border-border">
        <SearchInput
          value={query}
          onChange={setQuery}
          placeholder="Search logs…"
          className="w-60"
        />

        <div className="flex items-center gap-1">
          {LEVELS.map((lv) => (
            <button
              key={lv}
              onClick={() => toggleLevel(lv)}
              className={clsx(
                'px-2 py-0.5 rounded-full border text-[10.5px] font-mono cursor-pointer transition-colors',
                levels.has(lv)
                  ? clsx('border-border-hi bg-panel-alt', LEVEL_COLOR[lv])
                  : 'border-border text-text-faint opacity-50 hover:opacity-80')}>
              {lv}
            </button>
          ))}
        </div>

        {sources.length > 0 && sources.length <= MAX_SRC_CHIPS ? (
          <div className="flex items-center gap-1 flex-wrap">
            {srcSel.size > 0 && (
              <button
                onClick={() => setSrcSel(new Set<string>())}
                className="text-[10px] text-accent-hi hover:underline cursor-pointer">
                all
              </button>
            )}
            {sources.map((s) => (
              <button
                key={s}
                onClick={() => toggleSource(s)}
                title={s}
                className={clsx(
                  'px-2 py-0.5 rounded-full border text-[10.5px] font-mono cursor-pointer transition-colors max-w-[140px] truncate',
                  srcSel.size === 0 || srcSel.has(s)
                    ? 'border-border-hi bg-panel-alt text-text-dim'
                    : 'border-border text-text-faint opacity-50 hover:opacity-80')}>
                {s}
              </button>
            ))}
          </div>
        ) : sources.length > MAX_SRC_CHIPS ? (
          <Select
            value={srcSel.size === 1 ? [...srcSel][0] : ''}
            onChange={(e) => setSrcSel(
              e.target.value
                ? new Set<string>([e.target.value])
                : new Set<string>())}
            className="!py-1 !text-[12px]">
            <option value="">all sources</option>
            {sources.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </Select>
        ) : null}

        <div className="ml-auto flex items-center gap-2">
          <Tag kind="err">err {counts.errors}</Tag>
          <Tag kind="warn">warn {counts.warnings}</Tag>
          <Tag kind="idle">{lines.length}/{counts.total}</Tag>
          {lines.length > 0 && (
            <div className="flex items-center gap-1 border-r border-border/60 pr-2">
              <button
                onClick={copyLogs}
                title="Copy all filtered logs"
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-panel-alt hover:bg-panel-hover border border-border text-text-dim hover:text-text cursor-pointer transition-colors">
                {copied ? <CheckIcon size={11} className="text-ok" /> : <Copy size={11} />}
                {copied ? 'Copied' : 'Copy'}
              </button>
              <button
                onClick={downloadLogs}
                title="Download logs as text file"
                className="p-1 rounded text-text-faint hover:text-text hover:bg-panel-alt transition-colors cursor-pointer">
                <Download size={12} />
              </button>
            </div>
          )}
          <IconBtn title="Refresh" onClick={() => void load()}>
            <RefreshCw size={13} />
          </IconBtn>
          <Btn variant="danger" size="xs" onClick={clear}>
            <Trash2 size={11} /> Clear
          </Btn>
          <Check label="auto" checked={auto} onChange={setAuto} />
        </div>
      </div>
      {err && (
        <div className="px-3 py-1.5 text-[11.5px] text-err border-b border-border">
          {err}
        </div>
      )}

      {/* log pane */}
      <div className="relative flex-1 min-h-0">
        <div
          ref={paneRef}
          onScroll={onScroll}
          className="h-full overflow-y-auto px-4 py-2 font-mono text-[11.5px] leading-[1.6]">
          {lines.map((l, i) => (
            <div key={i}
              className={clsx('whitespace-pre-wrap break-all', lineClass(l))}>
              {l}
            </div>
          ))}
          {!lines.length && (
            <EmptyState
              icon={<ScrollText size={40} strokeWidth={1.2} />}
              title="No log lines"
              desc={filtered
                ? 'No lines match the current filters.'
                : 'Log sources are discovered automatically — nothing written yet.'} />
          )}
        </div>
        {!following && (
          <button
            onClick={resumeFollow}
            className="absolute bottom-4 right-6 flex items-center gap-1.5 bg-panel-alt border border-border-hi rounded-full px-3 py-1.5 text-[11.5px] text-text-dim hover:text-text hover:border-accent shadow-card cursor-pointer transition-colors">
            <ArrowDownToLine size={12} /> follow
          </button>
        )}
      </div>
    </div>
  )
}
