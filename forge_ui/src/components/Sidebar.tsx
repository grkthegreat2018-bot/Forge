// Collapsible navigation sidebar — brand, grouped sections, live status badges & quick controls.

import { clsx } from 'clsx'
import {
  Activity, Bot, Box, ChevronsLeft, ChevronsRight, Cpu,
  FileText, Gauge, Hammer, Layers, LayoutDashboard, ListTodo,
  MessageSquare, Moon, Play, Search, Sparkles, Sun, TrendingUp, X, Zap,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import { useUi } from '../lib/ui'

export interface PageDef {
  path: string
  label: string
  icon: React.ReactNode
  section: string
  subtitle: string
}

export const PAGES: PageDef[] = [
  { path: '/agent', label: 'Agent', icon: <Bot size={17} />, section: 'Agentic', subtitle: 'Autonomous tool-using agent' },
  { path: '/chat', label: 'Chat', icon: <MessageSquare size={17} />, section: 'Agentic', subtitle: 'Chat studio' },
  { path: '/generations', label: 'Generations', icon: <Sparkles size={17} />, section: 'Agentic', subtitle: 'Generation playground' },
  { path: '/', label: 'Dashboard', icon: <LayoutDashboard size={17} />, section: 'Model', subtitle: 'System overview' },
  { path: '/engine', label: 'Engine', icon: <Cpu size={17} />, section: 'Model', subtitle: 'Engine control + tuning' },
  { path: '/models', label: 'Models', icon: <Box size={17} />, section: 'Model', subtitle: 'Checkpoints & configs' },
  { path: '/lora', label: 'LoRA', icon: <Layers size={17} />, section: 'Model', subtitle: 'Adapters & merging' },
  { path: '/finetune', label: 'Fine-Tune', icon: <Hammer size={17} />, section: 'Train', subtitle: 'SFT launcher' },
  { path: '/selfplay', label: 'Self-Play', icon: <Zap size={17} />, section: 'Train', subtitle: 'Autonomous training loop' },
  { path: '/training', label: 'Training Live', icon: <TrendingUp size={17} />, section: 'Train', subtitle: 'Run telemetry' },
  { path: '/launch', label: 'Launch', icon: <Play size={17} />, section: 'System', subtitle: 'Process presets' },
  { path: '/tasks', label: 'Tasks', icon: <ListTodo size={17} />, section: 'System', subtitle: 'Process manager' },
  { path: '/compute', label: 'Compute', icon: <Gauge size={17} />, section: 'System', subtitle: 'GPU monitor' },
  { path: '/logs', label: 'Logs', icon: <FileText size={17} />, section: 'System', subtitle: 'Log viewer' },
]

const SECTIONS = ['Agentic', 'Model', 'Train', 'System']

export function Sidebar() {
  const navigate = useNavigate()
  const collapsed = useUi((s) => s.collapsed)
  const toggle = useUi((s) => s.toggleCollapsed)
  const [query, setQuery] = useState('')

  const engine = useForge((s) => s.engine)
  const gpu = useForge((s) => s.gpu)
  const lora = useForge((s) => s.lora)
  const runs = useForge((s) => s.runs)
  const tasks = useForge((s) => s.tasks)
  const agentRuns = useForge((s) => s.agentRuns)
  const selfplayStatus = useForge((s) => s.selfplayStatus)
  const connected = useForge((s) => s.connected)
  const refreshStatus = useForge((s) => s.refreshStatus)

  // Live telemetry metrics for badges
  const activeAgents = useMemo(() => agentRuns.filter((r) => r.status === 'running').length, [agentRuns])
  const activeTasks = useMemo(() => tasks.filter((t) => t.is_live).length, [tasks])
  const activeRuns = useMemo(() => runs.filter((r) => r.is_live).length, [runs])
  const isSelfPlayRunning = useMemo(() => !!selfplayStatus?.status?.running, [selfplayStatus])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return PAGES
    return PAGES.filter((p) => p.label.toLowerCase().includes(q) || p.section.toLowerCase().includes(q))
  }, [query])

  const engKind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  const handleQuickPower = async () => {
    if (engine.state === 'ready') {
      await api.post('/api/engine/sleep').catch(() => undefined)
      await refreshStatus()
    } else if (engine.state === 'idle') {
      await api.post('/api/engine/wake').catch(() => undefined)
      await refreshStatus()
    }
  }

  return (
    <aside
      className={clsx(
        'flex flex-col h-full bg-bg-alt border-r border-border transition-all duration-200 shrink-0 select-none z-20',
        collapsed ? 'w-[64px]' : 'w-[252px]'
      )}>
      {/* brand header */}
      <div className={clsx('flex items-center gap-3 pt-4 pb-3',
        collapsed ? 'px-3 justify-center' : 'px-4')}>
        <div
          onClick={() => navigate('/')}
          className="w-8 h-8 rounded-xl bg-gradient-to-br from-grad-a to-grad-b flex items-center justify-center shrink-0 shadow-[0_0_16px_rgba(99,102,241,0.4)] cursor-pointer hover:scale-105 transition-transform">
          <Activity size={18} className="text-white" />
        </div>
        {!collapsed && (
          <div className="min-w-0 cursor-pointer" onClick={() => navigate('/')}>
            <div className="text-[14.5px] font-bold tracking-tight text-text leading-tight flex items-center gap-1.5">
              ForgeAI
              <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-accent/15 text-accent-hi border border-accent/30 font-normal">
                v12
              </span>
            </div>
            <div className="text-[9.5px] uppercase tracking-[0.16em] text-text-faint font-medium">
              Intelligence Studio
            </div>
          </div>
        )}
        {!collapsed && (
          <button
            onClick={toggle}
            className="ml-auto text-text-faint hover:text-text p-1 rounded-md hover:bg-panel-alt transition-colors cursor-pointer"
            title="Collapse (Ctrl+B)">
            <ChevronsLeft size={16} />
          </button>
        )}
      </div>

      {collapsed && (
        <button
          onClick={toggle}
          className="mx-auto mb-2 text-text-faint hover:text-text p-1 rounded-md hover:bg-panel-alt transition-colors cursor-pointer"
          title="Expand (Ctrl+B)">
          <ChevronsRight size={16} />
        </button>
      )}

      {/* search input */}
      {!collapsed && (
        <div className="px-3 pb-2">
          <div className="relative flex items-center">
            <Search size={13} className="absolute left-2.5 text-text-faint pointer-events-none" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search features… (/)"
              className="w-full bg-input border border-border rounded-lg pl-7 pr-7 py-1.5 text-[12px] text-text placeholder:text-text-faint focus:outline-none focus:border-accent/70 focus:ring-1 focus:ring-accent/30 transition-all" />
            {query && (
              <button
                onClick={() => setQuery('')}
                className="absolute right-2 text-text-faint hover:text-text cursor-pointer p-0.5">
                <X size={12} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* nav menu */}
      <nav className="flex-1 overflow-y-auto px-2.5 pb-2 space-y-3">
        {SECTIONS.map((sec) => {
          const items = filtered.filter((p) => p.section === sec)
          if (!items.length) return null
          return (
            <div key={sec}>
              {!collapsed && (
                <div className="px-2.5 pb-1 text-[10px] font-bold uppercase tracking-[0.16em] text-text-faint/80">
                  {sec}
                </div>
              )}
              <div className="space-y-0.5">
                {items.map((p) => {
                  let badge = null
                  if (p.path === '/agent' && activeAgents > 0) {
                    badge = <span className="ml-auto text-[10px] px-1.5 py-0.2 rounded-full font-bold bg-accent/20 text-accent-hi border border-accent/40 shadow-[0_0_8px_rgba(122,151,255,0.25)]">{activeAgents}</span>
                  } else if (p.path === '/tasks' && activeTasks > 0) {
                    badge = <span className="ml-auto text-[10px] px-1.5 py-0.2 rounded-full font-bold bg-accent2/20 text-accent2 border border-accent2/40 shadow-[0_0_8px_rgba(45,212,191,0.25)]">{activeTasks}</span>
                  } else if (p.path === '/training' && activeRuns > 0) {
                    badge = <span className="ml-auto w-2 h-2 rounded-full bg-ok shadow-[0_0_8px_rgba(52,211,153,0.6)] pulse-dot" />
                  } else if (p.path === '/selfplay' && isSelfPlayRunning) {
                    badge = <span className="ml-auto w-2 h-2 rounded-full bg-warn shadow-[0_0_8px_rgba(251,191,36,0.6)] pulse-dot" />
                  } else if (p.path === '/lora' && lora.current) {
                    badge = <span className="ml-auto text-[9.5px] px-1.5 py-0.2 rounded bg-panel border border-border text-text-dim truncate max-w-[65px]">{lora.current.split(/[\/\\]/).pop()}</span>
                  }

                  return (
                    <NavLink
                      key={p.path}
                      to={p.path}
                      end={p.path === '/'}
                      title={collapsed ? `${p.label} — ${p.subtitle}` : undefined}
                      className={({ isActive }) => clsx(
                        'group flex items-center gap-2.5 rounded-lg text-[13px] transition-all duration-150',
                        collapsed ? 'justify-center px-0 py-2' : 'px-3 py-2',
                        isActive
                          ? 'bg-accent/15 text-accent-hi border border-accent/30 font-medium shadow-[0_0_14px_-2px_rgba(122,151,255,0.2)]'
                          : 'text-text-dim border border-transparent hover:bg-panel-alt hover:text-text'
                      )}>
                      <span className={clsx('shrink-0 transition-transform group-hover:scale-110')}>
                        {p.icon}
                      </span>
                      {!collapsed && <span className="truncate flex-1">{p.label}</span>}
                      {!collapsed && badge}
                    </NavLink>
                  )
                })}
              </div>
            </div>
          )
        })}
      </nav>

      {/* status footer */}
      <div className={clsx('border-t border-border/80 px-3 py-3 bg-panel/30',
        collapsed && 'flex flex-col items-center gap-2')}>
        {collapsed ? (
          <>
            <span
              title={`Engine: ${engine.state} ${engine.info.config_name ? `(${engine.info.config_name})` : ''}`}
              className={clsx('w-2.5 h-2.5 rounded-full cursor-pointer',
                engKind === 'ok' ? 'bg-ok shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                : engKind === 'warn' ? 'bg-warn pulse-dot'
                : engKind === 'err' ? 'bg-err' : 'bg-text-faint')}
              onClick={() => navigate('/engine')} />
            <span
              title={`GPU: ${Math.round(gpu.vram_used_mb)} / ${Math.round(gpu.vram_total_mb)} MB (${Math.round(gpu.vram_pct)}%)`}
              className="w-2.5 h-2.5 rounded-full bg-accent cursor-pointer shadow-[0_0_8px_rgba(122,151,255,0.5)]"
              onClick={() => navigate('/compute')} />
          </>
        ) : (
          <div className="space-y-2">
            {/* engine status line */}
            <div className="flex items-center justify-between text-[11.5px]">
              <div
                className="flex items-center gap-2 min-w-0 cursor-pointer hover:text-text transition-colors"
                onClick={() => navigate('/engine')}>
                <span className={clsx('w-2 h-2 rounded-full shrink-0',
                  engKind === 'ok' ? 'bg-ok shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                  : engKind === 'warn' ? 'bg-warn pulse-dot'
                  : engKind === 'err' ? 'bg-err' : 'bg-text-faint')} />
                <span className="truncate font-medium text-text-dim">
                  {engine.state === 'ready'
                    ? (engine.info.config_name || 'engine ready')
                    : engine.state}
                </span>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {engine.state === 'ready' && (
                  <button
                    onClick={handleQuickPower}
                    title="Sleep engine"
                    className="text-text-faint hover:text-accent p-0.5 rounded hover:bg-panel-alt transition-colors cursor-pointer">
                    <Moon size={12} />
                  </button>
                )}
                {engine.state === 'idle' && (
                  <button
                    onClick={handleQuickPower}
                    title="Wake engine"
                    className="text-text-faint hover:text-ok p-0.5 rounded hover:bg-panel-alt transition-colors cursor-pointer">
                    <Sun size={12} />
                  </button>
                )}
                <span
                  title={connected ? 'WebSocket connected' : 'WebSocket disconnected'}
                  className="flex items-center gap-1 text-[10px] text-text-faint font-mono">
                  <span className={clsx('w-1.5 h-1.5 rounded-full', connected ? 'bg-ok' : 'bg-err')} />
                  {connected ? 'live' : 'off'}
                </span>
              </div>
            </div>

            {/* GPU VRAM progress */}
            {gpu.available && (
              <div
                className="cursor-pointer group"
                onClick={() => navigate('/compute')}
                title="Click to view detailed GPU telemetry">
                <div className="flex justify-between text-[10.5px] text-text-faint group-hover:text-text-dim transition-colors mb-1">
                  <span>RTX 5070</span>
                  <span className="font-mono tabular-nums">
                    {(gpu.vram_used_mb / 1024).toFixed(1)} / {Math.round(gpu.vram_total_mb / 1024)}GB ({Math.round(gpu.vram_pct)}%)
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-panel-alt overflow-hidden">
                  <div
                    className={clsx('h-full rounded-full transition-all duration-300',
                      gpu.vram_pct > 85 ? 'bg-err' : 'bg-gradient-to-r from-grad-a to-grad-b')}
                    style={{ width: `${Math.max(3, gpu.vram_pct)}%` }} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  )
}

