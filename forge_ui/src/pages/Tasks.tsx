// Tasks — process manager. Master-detail: live task list on the left,
// detail + merged stored/streamed output pane on the right.
// Live lines arrive over the hub WS as taskLines[taskId].

import { clsx } from 'clsx'
import {
  Check as CheckIcon, Copy, Play, RefreshCw, Square, Terminal, Trash2, X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import type { TaskInfo } from '../lib/types'
import {
  Btn, Check, EmptyState, IconBtn, Input, KvRow, SearchInput, StatusDot, Tag,
} from '../components/ui'

type TaskDetail = TaskInfo & { error?: string }

const STATUS_KIND = {
  starting: 'warn',
  running: 'accent',
  done: 'ok',
  crashed: 'err',
  killed: 'err',
} as const

function fmtDur(s: number): string {
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

function taskElapsed(t: TaskInfo, now: number): number {
  if (!t.started_at) return t.elapsed_s
  const end = t.ended_at ? t.ended_at * 1000 : now
  return Math.max(0, (end - t.started_at * 1000) / 1000)
}

/**
 * Merge the stored tail (from GET /tasks/{id}?tail=500) with live WS lines.
 * Both are suffix-slices of the same output stream: drop the leading live
 * lines already present at the end of `stored`. If there is no overlap the
 * streams are disjoint (WS reconnect gap) or live fully covers stored.
 */
function mergeLines(stored: string[], live: string[]): string[] {
  if (!live.length) return stored
  if (!stored.length) return live
  const max = Math.min(stored.length, live.length)
  for (let k = max; k > 0; k--) {
    let match = true
    for (let i = 0; i < k; i++) {
      if (stored[stored.length - k + i] !== live[i]) { match = false; break }
    }
    if (match) return stored.concat(live.slice(k))
  }
  return live.length >= stored.length ? live : stored.concat(live)
}

export default function Tasks() {
  const tasks = useForge((s) => s.tasks)
  const taskLines = useForge((s) => s.taskLines)
  const refresh = useForge((s) => s.refreshStatus)

  const [sel, setSel] = useState<string | null>(null)
  const [detail, setDetail] = useState<TaskDetail | null>(null)
  const [err, setErr] = useState('')
  const [name, setName] = useState('')
  const [command, setCommand] = useState('')
  const [follow, setFollow] = useState(true)
  const [now, setNow] = useState(() => Date.now())
  const outRef = useRef<HTMLDivElement>(null)

  const [taskSearch, setTaskSearch] = useState('')
  const [copied, setCopied] = useState(false)

  const ordered = useMemo(() => [...tasks].reverse(), [tasks])
  const filteredTasks = useMemo(() => {
    const q = taskSearch.trim().toLowerCase()
    if (!q) return ordered
    return ordered.filter((t) =>
      t.name.toLowerCase().includes(q) ||
      t.id.toLowerCase().includes(q) ||
      t.command.join(' ').toLowerCase().includes(q)
    )
  }, [ordered, taskSearch])

  const selTask = tasks.find((t) => t.id === sel) ?? null
  const anyLive = tasks.some((t) => t.is_live)

  // keep selection valid; auto-select newest task
  useEffect(() => {
    if (sel && tasks.some((t) => t.id === sel)) return
    setSel(ordered[0]?.id ?? null)
  }, [tasks, sel, ordered])

  // 1s tick so elapsed stays live
  useEffect(() => {
    if (!anyLive) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [anyLive])

  const loadDetail = useCallback((id: string) => {
    api.get<TaskDetail>(`/api/tasks/${id}?tail=500`)
      .then((d) => setDetail(d.error ? null : d))
      .catch(() => setDetail(null))
  }, [])

  // fetch stored lines on select and when the task stops being live
  useEffect(() => {
    if (!sel) { setDetail(null); return }
    loadDetail(sel)
  }, [sel, selTask?.is_live, loadDetail])

  // merge stored tail + live stream
  const liveLines = useMemo(
    () => (sel ? (taskLines[sel] ?? []) : []),
    [taskLines, sel])
  const stored = detail?.lines ?? selTask?.lines ?? []
  const merged = useMemo(
    () => mergeLines(stored, liveLines), [stored, liveLines])

  // auto-scroll output
  useEffect(() => {
    if (follow) {
      outRef.current?.scrollTo({ top: outRef.current.scrollHeight })
    }
  }, [merged.length, follow, sel])

  const kill = async (id: string) => {
    await api.post(`/api/tasks/${id}/kill`).catch((e) => setErr(String(e)))
    refresh().catch(() => undefined)
  }

  const remove = async (id: string) => {
    await api.del(`/api/tasks/${id}`).catch((e) => setErr(String(e)))
    if (sel === id) setSel(null)
    refresh().catch(() => undefined)
  }

  const clearFinished = async () => {
    try {
      await api.post<{ removed: number }>('/api/tasks/clear_finished')
    } catch (e) { setErr(String(e)) }
    refresh().catch(() => undefined)
  }

  const runCustom = async () => {
    const cmd = command.trim()
    if (!cmd) return
    try {
      const r = await api.post<{ ok: boolean; task_id?: string; error?: string }>(
        '/api/tasks/launch', { command: cmd, name: name.trim() })
      if (!r.ok) {
        setErr(r.error ?? 'launch failed')
      } else {
        setErr('')
        setCommand('')
        if (r.task_id) setSel(r.task_id)
      }
    } catch (e) { setErr(String(e)) }
    refresh().catch(() => undefined)
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* toolbar: clear + custom launch */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
        <Btn variant="subtle" onClick={clearFinished}>
          <Trash2 size={12} /> Clear finished
        </Btn>
        <div className="w-px h-5 bg-border mx-1" />
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="name"
          className="w-40 !py-1 !text-[12px]" />
        <Input
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void runCustom() }}
          placeholder="command to run…"
          className="flex-1 !py-1 !text-[12px] font-mono" />
        <Btn variant="primary" onClick={runCustom} disabled={!command.trim()}>
          <Play size={12} /> Run
        </Btn>
      </div>
      {err && (
        <div className="px-3 py-1.5 text-[11.5px] text-err border-b border-border">
          {err}
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        {/* left: task list */}
        <div className="w-[300px] shrink-0 border-r border-border flex flex-col">
          <div className="p-2 border-b border-border/70">
            <SearchInput
              value={taskSearch}
              onChange={setTaskSearch}
              placeholder="Search processes…"
              className="w-full"
            />
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {filteredTasks.map((t) => (
              <TaskRow key={t.id} task={t} sel={t.id === sel} now={now}
                onSelect={() => setSel(t.id)} onKill={() => void kill(t.id)} />
            ))}
            {!filteredTasks.length && (
              <div className="text-[11.5px] text-text-faint text-center py-8">
                {taskSearch ? 'no matching processes' : 'no processes'}
              </div>
            )}
          </div>
        </div>

        {/* right: detail */}
        <div className="flex-1 min-w-0 flex flex-col">
          {selTask ? (
            <>
              <div className="flex items-center gap-2 px-4 py-2 border-b border-border">
                <span className="text-[13px] font-medium truncate">
                  {selTask.name}
                </span>
                <Tag kind={STATUS_KIND[selTask.status]}>
                  {selTask.is_live && (
                    <StatusDot kind={STATUS_KIND[selTask.status]} pulse />)}
                  {selTask.status}
                </Tag>
                <div className="ml-auto flex items-center gap-1.5">
                  {merged.length > 0 && (
                    <button
                      onClick={() => {
                        void navigator.clipboard.writeText(merged.join('\n')).then(() => {
                          setCopied(true)
                          setTimeout(() => setCopied(false), 1200)
                        })
                      }}
                      title="Copy task output"
                      className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-panel-alt hover:bg-panel-hover border border-border text-text-dim hover:text-text cursor-pointer transition-colors">
                      {copied ? <CheckIcon size={11} className="text-ok" /> : <Copy size={11} />}
                      {copied ? 'Copied' : 'Copy log'}
                    </button>
                  )}
                  <IconBtn title="Reload output"
                    onClick={() => sel && loadDetail(sel)}>
                    <RefreshCw size={12} />
                  </IconBtn>
                  {selTask.is_live ? (
                    <Btn variant="danger" size="xs" onClick={() => void kill(selTask.id)}>
                      <Square size={11} /> Kill
                    </Btn>
                  ) : (
                    <Btn variant="subtle" size="xs" onClick={() => void remove(selTask.id)}>
                      <Trash2 size={11} /> Remove
                    </Btn>
                  )}
                </div>
              </div>

              <div className="px-4 py-2 border-b border-border">
                <div
                  className="font-mono text-[11px] text-text-dim bg-bg-alt border border-border rounded-lg px-2.5 py-1.5 truncate"
                  title={selTask.command.join(' ')}>
                  {selTask.command.join(' ')}
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-x-6">
                  <KvRow k="pid" mono v={selTask.pid || '—'} />
                  <KvRow k="exit" mono v={selTask.exit_code ?? '—'} />
                  <KvRow k="elapsed" v={fmtDur(taskElapsed(selTask, now))} />
                  <KvRow k="started"
                    v={selTask.started_at
                      ? new Date(selTask.started_at * 1000).toLocaleTimeString()
                      : '—'} />
                </div>
              </div>

              <div ref={outRef}
                className="flex-1 overflow-y-auto px-4 py-2 font-mono text-[11.5px] leading-[1.55]">
                {merged.map((l, i) => (
                  <div key={i}
                    className="whitespace-pre-wrap break-all text-text-dim">
                    {l}
                  </div>
                ))}
                {!merged.length && (
                  <div className="text-text-faint text-[12px] text-center py-8">
                    no output yet
                  </div>
                )}
              </div>

              <div className="flex items-center gap-3 px-4 py-1.5 border-t border-border">
                <Check label="follow" checked={follow} onChange={setFollow} />
                <span className="text-[10.5px] text-text-faint ml-auto">
                  {merged.length} lines
                </span>
                {selTask.log_path && (
                  <span
                    className="text-[10.5px] text-text-faint font-mono truncate max-w-[40%]"
                    title={selTask.log_path}>
                    {selTask.log_path}
                  </span>
                )}
              </div>
            </>
          ) : (
            <EmptyState
              icon={<Terminal size={40} strokeWidth={1.2} />}
              title="No task selected"
              desc="Launch a preset from the Launch page or run a custom command above." />
          )}
        </div>
      </div>
    </div>
  )
}

/* ---------- task row ---------- */

function TaskRow({ task, sel, now, onSelect, onKill }: {
  task: TaskInfo
  sel: boolean
  now: number
  onSelect: () => void
  onKill: () => void
}) {
  const kind = STATUS_KIND[task.status]
  return (
    <button
      onClick={onSelect}
      className={clsx(
        'w-full text-left px-2.5 py-2 rounded-lg mb-0.5 group cursor-pointer transition-colors border',
        sel
          ? 'bg-accent/15 border-accent/25 text-text'
          : 'border-transparent text-text-dim hover:bg-panel-alt')}>
      <div className="flex items-center gap-1.5">
        <span className="text-[12.5px] truncate flex-1 text-text">
          {task.name}
        </span>
        {task.is_live && (
          <IconBtn
            title="Kill"
            className="!w-5 !h-5 hover:!text-err"
            onClick={(e) => { e.stopPropagation(); onKill() }}>
            <X size={11} />
          </IconBtn>
        )}
      </div>
      <div className="flex items-center gap-2 mt-1">
        <Tag kind={kind} className="!px-1.5">
          {task.is_live && <StatusDot kind={kind} pulse />}
          {task.status}
        </Tag>
        <span className="text-[10.5px] text-text-faint">
          {fmtDur(taskElapsed(task, now))}
        </span>
        {task.exit_code != null && (
          <span className="text-[10.5px] text-text-faint font-mono">
            exit {task.exit_code}
          </span>
        )}
      </div>
    </button>
  )
}
