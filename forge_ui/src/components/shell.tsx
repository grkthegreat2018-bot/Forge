// Shell chrome: PageHeader, ApprovalPopups, CommandPalette.

import { clsx } from 'clsx'
import {
  AlertTriangle, Bot, Check, ChevronDown, Cpu, FileText, Gauge,
  Layers, Moon, Play, RefreshCw, Search, Square, Sun,
  Terminal, X, Zap,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import { useUi } from '../lib/ui'
import { Btn, StatusDot, Tag } from './ui'
import { PAGES } from './Sidebar'

export function PageHeader({ title, subtitle, actions }: {
  title: string
  subtitle?: string
  actions?: React.ReactNode
}) {
  const navigate = useNavigate()
  const setPalette = useUi((s) => s.setPalette)

  const engine = useForge((s) => s.engine)
  const gpu = useForge((s) => s.gpu)
  const lora = useForge((s) => s.lora)
  const progress = useForge((s) => s.engineProgress)
  const refreshStatus = useForge((s) => s.refreshStatus)

  const [clock, setClock] = useState('')
  const [engineMenuOpen, setEngineMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const t = setInterval(
      () => setClock(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setEngineMenuOpen(false)
      }
    }
    if (engineMenuOpen) window.addEventListener('mousedown', handler)
    return () => window.removeEventListener('mousedown', handler)
  }, [engineMenuOpen])

  const kind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  const currentDef = PAGES.find((p) => p.label === title || title.startsWith(p.label))

  return (
    <header className="flex items-center gap-4 h-14 px-6 border-b border-border/80 glass-panel shrink-0 select-none z-10">
      {/* Title & breadcrumbs */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          {currentDef?.section && (
            <span className="text-[11px] font-semibold text-text-faint uppercase tracking-wider hidden sm:inline">
              {currentDef.section} <span className="text-border-hi font-normal mx-0.5">/</span>
            </span>
          )}
          <h1 className="text-[15px] font-bold tracking-tight leading-tight truncate text-text">
            {title}
          </h1>
          {progress && (
            <span className="text-[11.5px] text-warn font-medium flex items-center gap-1 bg-warn/10 px-2 py-0.5 rounded-full border border-warn/25 animate-pulse truncate max-w-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-warn" />
              {progress}
            </span>
          )}
        </div>
        {subtitle && !progress && (
          <div className="text-[11.5px] text-text-faint leading-tight truncate mt-0.5">
            {subtitle}
          </div>
        )}
      </div>

      {/* Right side controls */}
      <div className="ml-auto flex items-center gap-2.5">
        {actions}

        {/* LoRA active chip */}
        {lora.current && (
          <button
            onClick={() => navigate('/lora')}
            title={`Active LoRA: ${lora.current} (mode: ${lora.mode})`}
            className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11.5px] font-medium bg-panel-alt border border-border hover:border-accent/40 text-text-dim hover:text-text transition-colors cursor-pointer">
            <Layers size={13} className="text-accent" />
            <span className="max-w-[100px] truncate">{lora.current.split(/[\\/]/).pop()}</span>
            <span className="text-[9.5px] font-mono uppercase text-accent-hi bg-accent/15 px-1 py-0.2 rounded border border-accent/25">
              {lora.mode}
            </span>
          </button>
        )}

        {/* GPU Quick Pill */}
        {gpu.available && (
          <button
            onClick={() => navigate('/compute')}
            title={`RTX 5070 — ${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(0)} GB (${Math.round(gpu.vram_pct)}%), Temp: ${gpu.temp_c ?? '—'}°C`}
            className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-panel-alt/80 border border-border hover:border-border-hi text-[11.5px] font-mono text-text-dim hover:text-text transition-colors cursor-pointer">
            <Gauge size={13} className="text-accent2" />
            <span>{(gpu.vram_used_mb / 1024).toFixed(1)}GB</span>
            <div className="w-10 h-1.5 rounded-full bg-panel overflow-hidden">
              <div
                className={clsx('h-full rounded-full transition-all',
                  gpu.vram_pct > 85 ? 'bg-err' : 'bg-gradient-to-r from-grad-a to-grad-b')}
                style={{ width: `${gpu.vram_pct}%` }}
              />
            </div>
            {gpu.temp_c != null && (
              <span className="text-text-faint">{gpu.temp_c}°C</span>
            )}
          </button>
        )}

        {/* Engine status & quick menu */}
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setEngineMenuOpen(!engineMenuOpen)}
            className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-panel border border-border hover:border-border-hi transition-all cursor-pointer">
            <StatusDot kind={kind} pulse={engine.state === 'loading'} />
            <span className="text-[12px] font-medium text-text">
              {engine.state === 'ready'
                ? (engine.info.config_name ?? 'ready')
                : engine.state}
            </span>
            <ChevronDown size={13} className="text-text-faint" />
          </button>

          {engineMenuOpen && (
            <div className="absolute right-0 mt-1.5 w-52 glass-dropdown rounded-xl border border-border-hi p-1.5 shadow-2xl z-50 fade-in">
              <div className="px-3 py-1.5 border-b border-border/50 text-[10.5px] font-semibold uppercase tracking-wider text-text-faint">
                Engine Controls
              </div>
              {engine.state === 'ready' ? (
                <>
                  <button
                    onClick={async () => {
                      setEngineMenuOpen(false)
                      await api.post('/api/engine/unload').catch(() => undefined)
                      await refreshStatus()
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-[12.5px] text-text-dim hover:text-err hover:bg-err/10 rounded-lg transition-colors cursor-pointer">
                    <Square size={13} /> Unload Model
                  </button>
                  <button
                    onClick={async () => {
                      setEngineMenuOpen(false)
                      await api.post('/api/engine/reactivate').catch(() => undefined)
                      await refreshStatus()
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-[12.5px] text-text-dim hover:text-text hover:bg-panel-alt rounded-lg transition-colors cursor-pointer">
                    <RefreshCw size={13} /> Reactivate Engine
                  </button>
                  <button
                    onClick={async () => {
                      setEngineMenuOpen(false)
                      await api.post('/api/engine/sleep').catch(() => undefined)
                      await refreshStatus()
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-[12.5px] text-text-dim hover:text-accent hover:bg-accent/10 rounded-lg transition-colors cursor-pointer">
                    <Moon size={13} /> Put Engine to Sleep
                  </button>
                  <button
                    onClick={async () => {
                      setEngineMenuOpen(false)
                      await api.post('/api/engine/clear_recovery').catch(() => undefined)
                      await refreshStatus()
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-[12.5px] text-text-dim hover:text-text hover:bg-panel-alt rounded-lg transition-colors cursor-pointer">
                    <Zap size={13} /> Clear Recovery State
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => {
                      setEngineMenuOpen(false)
                      navigate('/models')
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-[12.5px] text-accent-hi hover:bg-accent/10 rounded-lg transition-colors cursor-pointer font-medium">
                    <Play size={13} /> Select & Load Model
                  </button>
                  {engine.state === 'idle' && (
                    <button
                      onClick={async () => {
                        setEngineMenuOpen(false)
                        await api.post('/api/engine/wake').catch(() => undefined)
                        await refreshStatus()
                      }}
                      className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-[12.5px] text-text-dim hover:text-ok hover:bg-ok/10 rounded-lg transition-colors cursor-pointer">
                      <Sun size={13} /> Wake Engine
                    </button>
                  )}
                </>
              )}
              <div className="border-t border-border/50 my-1" />
              <button
                onClick={() => {
                  setEngineMenuOpen(false)
                  navigate('/engine')
                }}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-[12px] text-text-faint hover:text-text hover:bg-panel-alt rounded-lg transition-colors cursor-pointer">
                <Cpu size={13} /> Open Engine Settings
              </button>
            </div>
          )}
        </div>

        {/* Command Palette Trigger */}
        <button
          onClick={() => setPalette(true)}
          title="Command Palette (Ctrl+K)"
          className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-panel-alt hover:bg-panel-hover border border-border text-text-faint hover:text-text transition-colors cursor-pointer text-[11px]">
          <Search size={13} />
          <kbd className="hidden sm:inline font-mono px-1 py-0.2 rounded bg-input border border-border/80 text-[10px]">
            Ctrl+K
          </kbd>
        </button>

        {/* Clock */}
        <span className="text-[11.5px] text-text-faint font-mono tabular-nums hidden lg:inline">
          {clock}
        </span>
      </div>
    </header>
  )
}

/** Floating approval requests (backup restore / library install / agent actions). */
export function ApprovalPopups() {
  const approvals = useForge((s) => s.approvals)
  const respond = useForge((s) => s.respondApproval)
  const [respondingId, setRespondingId] = useState<string | null>(null)

  if (!approvals.length) return null

  const handleRespond = async (id: string, granted: boolean) => {
    setRespondingId(id)
    try {
      await respond(id, granted)
    } finally {
      setRespondingId(null)
    }
  }

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-3 max-w-md w-full">
      {approvals.map((a) => {
        const detailObj = a.detail as Record<string, unknown>
        const warning = (detailObj.warning as string) || (detailObj.msg as string)
        const isBusy = respondingId === a.id

        return (
          <div
            key={a.id}
            className="glass-dropdown border-2 border-warn/70 rounded-2xl shadow-2xl p-5 slide-in-left">
            <div className="flex items-center justify-between gap-2 mb-2">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-warn/15 border border-warn/30 flex items-center justify-center text-warn">
                  <AlertTriangle size={15} />
                </div>
                <div>
                  <div className="text-[13px] font-bold text-text">Approval Requested</div>
                  <div className="text-[11px] text-text-dim font-mono">{a.kind}</div>
                </div>
              </div>
              <Tag kind="warn" pulse>ACTION NEEDED</Tag>
            </div>

            {warning && (
              <div className="text-[12.5px] text-warn-light font-medium bg-warn/10 border border-warn/25 rounded-lg px-3 py-2 mb-2.5">
                {warning}
              </div>
            )}

            <div className="bg-deep border border-border rounded-lg p-2.5 mb-3 text-[11.5px] font-mono text-text-dim max-h-36 overflow-y-auto">
              <pre className="whitespace-pre-wrap break-all">
                {JSON.stringify(a.detail, null, 2)}
              </pre>
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-1">
              <Btn
                variant="danger"
                size="sm"
                disabled={isBusy}
                onClick={() => handleRespond(a.id, false)}>
                <X size={13} /> Deny Request
              </Btn>
              <Btn
                variant="primary"
                size="sm"
                loading={isBusy}
                onClick={() => handleRespond(a.id, true)}>
                <Check size={13} /> Grant Approval
              </Btn>
            </div>
          </div>
        )
      })}
    </div>
  )
}

interface PaletteAction {
  id: string
  title: string
  subtitle?: string
  category: 'Pages' | 'Engine Actions' | 'Quick Tools'
  icon: React.ReactNode
  shortcut?: string
  run: () => void | Promise<void>
}

/** Ctrl+K palette — jump to any page or run engine & system commands. */
export function CommandPalette() {
  const open = useUi((s) => s.paletteOpen)
  const setOpen = useUi((s) => s.setPalette)
  const [q, setQ] = useState('')
  const [selectedIdx, setSelectedIdx] = useState(0)
  const navigate = useNavigate()

  const refreshStatus = useForge((s) => s.refreshStatus)

  useEffect(() => {
    if (open) {
      setQ('')
      setSelectedIdx(0)
    }
  }, [open])

  const allActions: PaletteAction[] = [
    // Pages
    ...PAGES.map((p, idx) => ({
      id: `page-${p.path}`,
      title: p.label,
      subtitle: `${p.section} — ${p.subtitle}`,
      category: 'Pages' as const,
      icon: p.icon,
      shortcut: idx < 9 ? `Ctrl+${idx + 1}` : undefined,
      run: () => navigate(p.path),
    })),

    // Engine Actions
    {
      id: 'eng-load',
      title: 'Load Model Checkpoint',
      subtitle: 'Pick and initialize a model in VRAM',
      category: 'Engine Actions',
      icon: <Play size={15} />,
      run: () => navigate('/models'),
    },
    {
      id: 'eng-unload',
      title: 'Unload Model',
      subtitle: 'Free GPU VRAM and reset engine to idle',
      category: 'Engine Actions',
      icon: <Square size={15} />,
      run: async () => {
        await api.post('/api/engine/unload').catch(() => undefined)
        await refreshStatus()
      },
    },
    {
      id: 'eng-reactivate',
      title: 'Reactivate Engine',
      subtitle: 'Reinitialize current model state and activation',
      category: 'Engine Actions',
      icon: <RefreshCw size={15} />,
      run: async () => {
        await api.post('/api/engine/reactivate').catch(() => undefined)
        await refreshStatus()
      },
    },
    {
      id: 'eng-sleep',
      title: 'Sleep Engine',
      subtitle: 'Offload memory and preserve live state',
      category: 'Engine Actions',
      icon: <Moon size={15} />,
      run: async () => {
        await api.post('/api/engine/sleep').catch(() => undefined)
        await refreshStatus()
      },
    },
    {
      id: 'eng-wake',
      title: 'Wake Engine',
      subtitle: 'Restore model back to VRAM and active state',
      category: 'Engine Actions',
      icon: <Sun size={15} />,
      run: async () => {
        await api.post('/api/engine/wake').catch(() => undefined)
        await refreshStatus()
      },
    },
    {
      id: 'eng-clear-rec',
      title: 'Clear Crash Recovery State',
      subtitle: 'Purge checkpoint recovery journal files',
      category: 'Engine Actions',
      icon: <Zap size={15} />,
      run: async () => {
        await api.post('/api/engine/clear_recovery').catch(() => undefined)
        await refreshStatus()
      },
    },

    // Quick Tools
    {
      id: 'tool-agent-new',
      title: 'New Agent Run',
      subtitle: 'Launch an autonomous task with tool harness',
      category: 'Quick Tools',
      icon: <Bot size={15} />,
      run: () => navigate('/agent'),
    },
    {
      id: 'tool-tasks-clear',
      title: 'Clear Finished Tasks',
      subtitle: 'Remove completed and killed processes from task manager',
      category: 'Quick Tools',
      icon: <Terminal size={15} />,
      run: async () => {
        await api.post('/api/tasks/clear_finished').catch(() => undefined)
        await refreshStatus()
      },
    },
    {
      id: 'tool-compute',
      title: 'Inspect GPU Telemetry',
      subtitle: 'View Blackwell VRAM, temperature, and power charts',
      category: 'Quick Tools',
      icon: <Gauge size={15} />,
      run: () => navigate('/compute'),
    },
    {
      id: 'tool-logs',
      title: 'View System Logs',
      subtitle: 'Unified log tail with filtering and search',
      category: 'Quick Tools',
      icon: <FileText size={15} />,
      run: () => navigate('/logs'),
    },
  ]

  const ql = q.trim().toLowerCase()
  const filtered = ql
    ? allActions.filter(
        (a) =>
          a.title.toLowerCase().includes(ql) ||
          (a.subtitle && a.subtitle.toLowerCase().includes(ql)) ||
          a.category.toLowerCase().includes(ql)
      )
    : allActions

  useEffect(() => {
    setSelectedIdx(0)
  }, [q])

  if (!open) return null

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false)
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIdx((prev) => (prev + 1) % Math.max(1, filtered.length))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIdx((prev) => (prev - 1 + filtered.length) % Math.max(1, filtered.length))
    } else if (e.key === 'Enter' && filtered.length > 0) {
      e.preventDefault()
      const action = filtered[selectedIdx]
      if (action) {
        action.run()
        setOpen(false)
      }
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-start justify-center pt-[15vh] p-4 fade-in"
      onClick={() => setOpen(false)}>
      <div
        className="w-full max-w-xl glass-dropdown rounded-2xl border border-border-hi shadow-2xl overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}>
        {/* search input */}
        <div className="flex items-center px-4 py-3 border-b border-border/80 gap-3">
          <Search size={17} className="text-accent shrink-0" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command, page name, or action…"
            className="w-full bg-transparent text-[14.5px] text-text placeholder:text-text-faint focus:outline-none"
          />
          {q && (
            <button
              onClick={() => setQ('')}
              className="text-text-faint hover:text-text cursor-pointer p-1">
              <X size={14} />
            </button>
          )}
        </div>

        {/* results list */}
        <div className="max-h-[380px] overflow-y-auto p-2 space-y-1">
          {filtered.map((action, idx) => {
            const isSelected = idx === selectedIdx
            return (
              <div
                key={action.id}
                onClick={() => {
                  action.run()
                  setOpen(false)
                }}
                onMouseEnter={() => setSelectedIdx(idx)}
                className={clsx(
                  'flex items-center gap-3 px-3.5 py-2.5 rounded-xl cursor-pointer transition-all duration-100',
                  isSelected
                    ? 'bg-accent/20 text-white border border-accent/40 shadow-[0_0_12px_rgba(122,151,255,0.2)]'
                    : 'text-text-dim border border-transparent hover:bg-panel-alt hover:text-text'
                )}>
                <span className={clsx('shrink-0', isSelected ? 'text-accent-hi' : 'text-text-faint')}>
                  {action.icon}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-medium leading-tight truncate">
                    {action.title}
                  </div>
                  {action.subtitle && (
                    <div className="text-[11px] text-text-faint leading-tight truncate mt-0.5">
                      {action.subtitle}
                    </div>
                  )}
                </div>
                {action.shortcut && (
                  <kbd className="font-mono text-[10.5px] text-text-faint bg-input px-1.5 py-0.5 rounded border border-border/70 shrink-0">
                    {action.shortcut}
                  </kbd>
                )}
              </div>
            )
          })}

          {!filtered.length && (
            <div className="py-10 text-center text-[13px] text-text-faint">
              No matching commands or pages found for "{q}"
            </div>
          )}
        </div>

        {/* footer shortcuts info */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border/50 bg-panel/40 text-[11px] text-text-faint">
          <div className="flex items-center gap-3">
            <span><kbd className="font-mono bg-input px-1 rounded border border-border">↑</kbd> <kbd className="font-mono bg-input px-1 rounded border border-border">↓</kbd> navigate</span>
            <span><kbd className="font-mono bg-input px-1 rounded border border-border">↵</kbd> select</span>
            <span><kbd className="font-mono bg-input px-1 rounded border border-border">esc</kbd> close</span>
          </div>
          <span>ForgeAI Command Center</span>
        </div>
      </div>
    </div>
  )
}

