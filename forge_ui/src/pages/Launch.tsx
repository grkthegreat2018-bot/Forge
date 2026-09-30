// Launch — process preset launcher + ad-hoc command runner.
// Presets come from /api/launch/presets; expanding a card exposes the
// preset's arg_defaults as editable rows collected into `overrides`.

import { ArrowRight, ChevronDown, ChevronRight, Play, Rocket } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import type { ProcessPreset } from '../lib/types'
import { Btn, Card, EmptyState, Input, SearchInput, Spinner, Tag } from '../components/ui'

interface LaunchResult {
  ok: boolean
  task_id?: string
  cmd?: string[]
  error?: string
}

export default function Launch() {
  const [presets, setPresets] = useState<ProcessPreset[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    api.get<{ presets: ProcessPreset[] }>('/api/launch/presets')
      .then((r) => setPresets(r.presets))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return presets
    return presets.filter((p) =>
      p.name.toLowerCase().includes(q) ||
      (p.description && p.description.toLowerCase().includes(q)) ||
      p.script.toLowerCase().includes(q)
    )
  }, [presets, search])

  return (
    <div className="h-full overflow-y-auto p-5 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <SearchInput
          value={search}
          onChange={setSearch}
          placeholder="Filter process launch presets…"
          className="w-full max-w-md"
        />
        <Btn variant="subtle" size="sm" onClick={() => navigate('/tasks')}>
          Manage Running Tasks <ArrowRight size={12} />
        </Btn>
      </div>

      {loading && (
        <div className="flex items-center justify-center gap-2 py-10 text-[12.5px] text-text-dim">
          <Spinner /> loading presets…
        </div>
      )}
      {error && (
        <div className="text-[12.5px] text-err bg-err/10 border border-err/30 rounded-lg px-3 py-2">
          {error}
        </div>
      )}
      {!loading && !error && !filtered.length && (
        <EmptyState
          icon={<Rocket size={40} strokeWidth={1.2} />}
          title={search ? 'No matching presets' : 'No launch presets'}
          desc={search ? 'Try adjusting your search filter.' : 'No launchable process presets are defined on the server.'} />
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filtered.map((p) => (
          <PresetCard key={p.name} preset={p}
            onOpenTasks={() => navigate('/tasks')} />
        ))}
      </div>
      <CustomCommand onOpenTasks={() => navigate('/tasks')} />
    </div>
  )
}

/* ---------- preset card ---------- */

function PresetCard({ preset, onOpenTasks }: {
  preset: ProcessPreset
  onOpenTasks: () => void
}) {
  const [open, setOpen] = useState(false)
  const [args, setArgs] = useState<Record<string, string>>(
    () => ({ ...preset.arg_defaults }))
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<LaunchResult | null>(null)

  const argEntries = Object.entries(preset.arg_defaults ?? {})

  const launch = async () => {
    if (busy) return
    setBusy(true)
    setResult(null)
    try {
      const r = await api.post<LaunchResult>('/api/launch/preset', {
        preset_name: preset.name,
        overrides: args,
      })
      setResult(r)
    } catch (e) {
      setResult({ ok: false, error: String(e) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-start gap-2 text-left cursor-pointer group">
        <span className="text-text-faint mt-0.5 group-hover:text-text transition-colors">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <span className="flex-1 min-w-0">
          <span className="block text-[13.5px] font-medium">{preset.name}</span>
          <span className="block text-[11.5px] text-text-faint mt-0.5">
            {preset.description}
          </span>
          <span className="block font-mono text-[11px] text-text-dim mt-1.5 truncate">
            {preset.script}
          </span>
        </span>
      </button>

      {open && (
        <div className="mt-3 pt-3 border-t border-border space-y-2 fade-in">
          {argEntries.length ? argEntries.map(([flag, def]) => (
            <label key={flag} className="flex items-center gap-2">
              <span
                className="w-32 shrink-0 font-mono text-[11px] text-text-faint truncate"
                title={flag}>
                {flag}
              </span>
              <Input
                value={args[flag] ?? def}
                onChange={(e) =>
                  setArgs((a) => ({ ...a, [flag]: e.target.value }))}
                className="flex-1 !py-1 !text-[12px] font-mono" />
            </label>
          )) : (
            <div className="text-[11.5px] text-text-faint">
              No configurable arguments.
            </div>
          )}

          <div className="flex items-center gap-2 pt-1">
            <Btn variant="primary" onClick={launch} disabled={busy}>
              {busy ? <Spinner /> : <Play size={12} />} Launch
            </Btn>
            {result?.ok && result.task_id && (
              <Tag kind="ok" className="font-mono">{result.task_id}</Tag>
            )}
          </div>

          {result && !result.ok && (
            <div className="text-[11.5px] text-err">
              {result.error ?? 'launch failed'}
            </div>
          )}
          {result?.ok && (
            <div className="space-y-2">
              {!!result.cmd?.length && (
                <div className="font-mono text-[10.5px] text-text-faint break-all bg-bg-alt border border-border rounded-lg px-2.5 py-1.5">
                  {result.cmd.join(' ')}
                </div>
              )}
              <Btn variant="outline" onClick={onOpenTasks}>
                Open Tasks <ArrowRight size={12} />
              </Btn>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

/* ---------- custom command ---------- */

function CustomCommand({ onOpenTasks }: { onOpenTasks: () => void }) {
  const [name, setName] = useState('')
  const [command, setCommand] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<LaunchResult | null>(null)

  const run = async () => {
    const cmd = command.trim()
    if (!cmd || busy) return
    setBusy(true)
    setResult(null)
    try {
      const r = await api.post<LaunchResult>('/api/tasks/launch', {
        command: cmd,
        name: name.trim(),
      })
      setResult(r)
      if (r.ok) setCommand('')
    } catch (e) {
      setResult({ ok: false, error: String(e) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card title="Custom command">
      <div className="flex items-center gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="name (optional)"
          className="w-44 !py-1 !text-[12px]" />
        <Input
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void run() }}
          placeholder="command — e.g. python scripts/train_expert.py --epochs 3"
          className="flex-1 !py-1 !text-[12px] font-mono" />
        <Btn variant="primary" onClick={run} disabled={!command.trim() || busy}>
          {busy ? <Spinner /> : <Play size={12} />} Launch
        </Btn>
      </div>
      {result && (
        <div className="mt-2.5 flex items-center gap-2 flex-wrap">
          {result.ok ? (
            <>
              <Tag kind="ok" className="font-mono">
                {result.task_id ?? 'launched'}
              </Tag>
              <span className="text-[11.5px] text-text-dim">process started</span>
              <Btn variant="outline" onClick={onOpenTasks}>
                Open Tasks <ArrowRight size={12} />
              </Btn>
            </>
          ) : (
            <span className="text-[11.5px] text-err">
              {result.error ?? 'launch failed'}
            </span>
          )}
        </div>
      )}
    </Card>
  )
}
