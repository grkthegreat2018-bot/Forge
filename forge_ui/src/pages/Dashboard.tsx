// Dashboard — engine state, GPU health, live runs, quick actions & test prompt.

import {
  ArrowRight, Bot, Cpu, Layers,
  MessageSquare, Play, RefreshCw, Square,
} from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ssePost } from '../lib/api'
import { useForge } from '../lib/store'
import { Gauge as GaugeChart } from '../components/charts'
import { Btn, Card, KvRow, MetricCard, StatusDot, Tag } from '../components/ui'

export default function Dashboard() {
  const engine = useForge((s) => s.engine)
  const gpu = useForge((s) => s.gpu)
  const lora = useForge((s) => s.lora)
  const runs = useForge((s) => s.runs)
  const tasks = useForge((s) => s.tasks)
  const agentRuns = useForge((s) => s.agentRuns)
  const progress = useForge((s) => s.engineProgress)
  const refreshStatus = useForge((s) => s.refreshStatus)
  const navigate = useNavigate()

  // Quick test inference prompt directly from dashboard
  const [quickPrompt, setQuickPrompt] = useState('def fibonacci(n):')
  const [quickOutput, setQuickOutput] = useState('')
  const [quickBusy, setQuickBusy] = useState(false)
  const [quickTokS, setQuickTokS] = useState<number | null>(null)

  const liveRuns = runs.filter((r) => r.is_live)
  const liveTasks = tasks.filter((t) => t.is_live)
  const kind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  const runQuickTest = async () => {
    if (!quickPrompt.trim() || quickBusy || engine.state !== 'ready') return
    setQuickOutput('')
    setQuickTokS(null)
    setQuickBusy(true)
    let acc = ''
    try {
      await ssePost('/api/gen/stream', {
        prompt: quickPrompt,
        max_new_tokens: 64,
        temperature: 0.2,
      }, (evt) => {
        if (evt.type === 'token') {
          acc += evt.data as string
          setQuickOutput(acc)
        } else if (evt.type === 'done') {
          const d = evt.data as { tok_s?: number }
          if (typeof d.tok_s === 'number') setQuickTokS(d.tok_s)
        }
      })
    } catch {
      // quiet fallback
    } finally {
      setQuickBusy(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto p-5 space-y-4">
      {/* Hero Status Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-border/80 bg-gradient-to-r from-panel via-panel-alt to-panel p-5 shadow-card">
        <div className="absolute right-0 top-0 h-full w-96 bg-gradient-to-l from-accent/10 to-transparent pointer-events-none" />
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 relative z-10">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-accent shadow-[0_0_8px_rgba(122,151,255,0.7)]" />
              <span className="text-[11px] font-bold uppercase tracking-wider text-accent-hi">
                System Command Center
              </span>
              <span className="text-text-faint font-mono text-[11px]">|</span>
              <span className="text-[11px] text-text-faint font-mono">
                NVIDIA RTX 5070 12GB • Intel Ultra 7 265F • 32GB RAM
              </span>
            </div>
            <h2 className="text-[20px] font-bold text-text tracking-tight flex items-center gap-2">
              ForgeAI Control Center
              {engine.state === 'ready' && (
                <Tag kind="ok">
                  {engine.info.config_name ?? 'Active'}
                </Tag>
              )}
            </h2>
            <p className="text-[12.5px] text-text-dim max-w-2xl leading-relaxed">
              Unified operating studio for high-throughput inference, Jamba/FLUX memory modeling, autonomous tool agents, and local fine-tuning.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {engine.state === 'ready' ? (
              <>
                <Btn
                  variant="subtle"
                  size="sm"
                  onClick={async () => {
                    await api.post('/api/engine/reactivate').catch(() => undefined)
                    await refreshStatus()
                  }}>
                  <RefreshCw size={13} /> Reactivate
                </Btn>
                <Btn
                  variant="danger"
                  size="sm"
                  onClick={async () => {
                    await api.post('/api/engine/unload').catch(() => undefined)
                    await refreshStatus()
                  }}>
                  <Square size={13} /> Unload
                </Btn>
              </>
            ) : (
              <Btn variant="primary" size="sm" onClick={() => navigate('/models')}>
                <Play size={13} /> Select & Load Model
              </Btn>
            )}
          </div>
        </div>
      </div>

      {/* Top telemetry metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          title="Engine Status"
          value={engine.state === 'ready' ? (engine.info.config_name || 'Ready') : engine.state}
          trend={engine.state === 'ready' ? `${engine.info.device || 'cuda'} • ${engine.info.dtype || 'bfloat16'}` : undefined}
          color="#34d399"
        />
        <MetricCard
          title="VRAM Usage"
          value={gpu.available ? `${(gpu.vram_used_mb / 1024).toFixed(1)}GB` : '—'}
          unit={gpu.available ? `/ ${Math.round(gpu.vram_total_mb / 1024)}GB` : undefined}
          trend={gpu.available ? `${Math.round(gpu.vram_pct)}% allocated` : undefined}
          color="#7a97ff"
        />
        <MetricCard
          title="GPU Compute"
          value={gpu.available ? `${Math.round(gpu.util_pct)}%` : '—'}
          trend={gpu.temp_c != null ? `${gpu.temp_c}°C • ${Math.round(gpu.power_w ?? 0)}W` : undefined}
          color="#2dd4bf"
        />
        <MetricCard
          title="Active LoRA"
          value={lora.current ? (lora.current.split(/[\\/]/).pop() ?? 'Loaded') : 'None'}
          trend={lora.mode ? `Mode: ${lora.mode}` : undefined}
          color="#c084fc"
        />
      </div>

      {/* Top strip: engine details + gpu visual gauges */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-2" title="Engine Architecture & Parameters">
          <div className="flex items-start gap-6">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <StatusDot kind={kind} pulse={engine.state === 'loading'} />
                <span className="text-[16px] font-semibold capitalize text-text">
                  {engine.state}
                </span>
                {engine.busy && <Tag kind="accent" pulse>busy</Tag>}
              </div>
              {engine.state === 'ready' && (
                <div className="grid grid-cols-2 gap-x-6 gap-y-1">
                  <KvRow k="checkpoint" mono
                    v={engine.info.checkpoint?.split(/[\\/]/).pop()} />
                  <KvRow k="config" v={engine.info.config_name} />
                  <KvRow k="load time" v={`${engine.info.load_s?.toFixed(2)}s`} />
                  <KvRow k="device" v={engine.info.device} />
                  <KvRow k="precision" v={engine.info.dtype} />
                  <KvRow k="torch compile" v={engine.info.use_compile ? 'enabled' : 'disabled'} />
                </div>
              )}
              {engine.state === 'loading' && (
                <div className="text-[12.5px] text-warn font-medium flex items-center gap-2 py-3">
                  <span className="w-2 h-2 rounded-full bg-warn animate-ping" />
                  {progress ?? 'Loading checkpoint weights into VRAM…'}
                </div>
              )}
              {engine.state === 'error' && (
                <div className="text-[12.5px] text-err bg-err/10 border border-err/30 rounded-lg p-2.5">
                  {engine.error}
                </div>
              )}
              {engine.state === 'idle' && (
                <div className="text-[12.5px] text-text-dim py-3">
                  No model loaded — select a checkpoint to begin inference and agent workflows.
                </div>
              )}
            </div>
          </div>
        </Card>

        <Card title="NVIDIA RTX 5070 Telemetry">
          {gpu.available ? (
            <div className="flex items-center justify-around py-1">
              <GaugeChart value={gpu.vram_pct} caption="VRAM" size={110} />
              <GaugeChart value={gpu.util_pct} caption="Util" size={110} />
            </div>
          ) : (
            <div className="text-[12.5px] text-text-dim py-8 text-center">
              No GPU telemetry available
            </div>
          )}
          {gpu.available && (
            <div className="grid grid-cols-2 gap-x-4 border-t border-border/40 pt-2 mt-1">
              <KvRow k="VRAM" v={`${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(0)} GB`} />
              <KvRow k="Temp / Power" v={`${gpu.temp_c != null ? `${gpu.temp_c}°C` : '—'} · ${gpu.power_w != null ? `${Math.round(gpu.power_w)}W` : '—'}`} />
            </div>
          )}
        </Card>
      </div>

      {/* Quick Test Bar */}
      <Card
        title="Quick Inference Probe"
        subtitle="Instant test generation through the resident ForgeEngine pipeline">
        <div className="space-y-2.5">
          <div className="flex gap-2">
            <input
              value={quickPrompt}
              onChange={(e) => setQuickPrompt(e.target.value)}
              disabled={engine.state !== 'ready' || quickBusy}
              placeholder="Test prompt (e.g. def fibonacci(n):)"
              className="flex-1 bg-input border border-border rounded-lg px-3 py-1.5 text-[13px] text-text font-mono focus:outline-none focus:border-accent/70"
            />
            <Btn
              variant="primary"
              size="sm"
              loading={quickBusy}
              disabled={engine.state !== 'ready' || !quickPrompt.trim()}
              onClick={runQuickTest}>
              <Play size={13} /> Probe
            </Btn>
          </div>
          {quickOutput && (
            <div className="bg-deep border border-border rounded-lg p-3 text-[12.5px] font-mono text-text max-h-36 overflow-y-auto whitespace-pre-wrap">
              {quickOutput}
              {quickTokS && (
                <div className="text-[10.5px] text-accent mt-1.5 pt-1 border-t border-border/40">
                  Speed: {quickTokS.toFixed(1)} tok/s
                </div>
              )}
            </div>
          )}
        </div>
      </Card>

      {/* Quick Navigation Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <QuickAction
          icon={<Bot size={20} className="text-accent" />}
          label="Autonomous Agent"
          desc="Multi-turn tool execution"
          onClick={() => navigate('/agent')}
        />
        <QuickAction
          icon={<MessageSquare size={20} className="text-accent2" />}
          label="Chat Studio"
          desc="Steerable conversation"
          onClick={() => navigate('/chat')}
        />
        <QuickAction
          icon={<Layers size={20} className="text-think" />}
          label="LoRA Adapters"
          desc={lora.current ? shortName(lora.current) : 'Manage & merge'}
          onClick={() => navigate('/lora')}
        />
        <QuickAction
          icon={<Cpu size={20} className="text-ok" />}
          label="Engine Tuning"
          desc="Activation & power state"
          onClick={() => navigate('/engine')}
        />
      </div>

      {/* Activity Streams Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card
          title="Live Training Telemetry"
          actions={
            <button
              onClick={() => navigate('/training')}
              className="text-[11px] text-accent hover:underline flex items-center gap-1 cursor-pointer">
              View <ArrowRight size={11} />
            </button>
          }>
          {liveRuns.length ? liveRuns.map((r) => (
            <div key={r.id} className="flex items-center gap-2 py-1.5 border-b border-border/30 last:border-none">
              <StatusDot kind="ok" pulse />
              <span className="text-[12.5px] font-medium truncate flex-1">{r.name}</span>
              <span className="text-[11px] text-text-faint font-mono">
                step {r.step}/{r.max_steps} · loss {r.loss.toFixed(3)}
              </span>
            </div>
          )) : (
            <div className="text-[12px] text-text-faint py-3 text-center">
              No active SFT or self-play training runs
            </div>
          )}
        </Card>

        <Card
          title="Active System Tasks"
          actions={
            <button
              onClick={() => navigate('/tasks')}
              className="text-[11px] text-accent hover:underline flex items-center gap-1 cursor-pointer">
              View <ArrowRight size={11} />
            </button>
          }>
          {liveTasks.length ? liveTasks.map((t) => (
            <div key={t.id} className="flex items-center gap-2 py-1.5 border-b border-border/30 last:border-none">
              <StatusDot kind="ok" pulse />
              <span className="text-[12.5px] font-medium truncate flex-1">{t.name}</span>
              <span className="text-[11px] text-text-faint font-mono">
                {Math.round(t.elapsed_s)}s
              </span>
            </div>
          )) : (
            <div className="text-[12px] text-text-faint py-3 text-center">
              No background processes active
            </div>
          )}
        </Card>

        <Card
          title="Recent Agent Runs"
          actions={
            <button
              onClick={() => navigate('/agent')}
              className="text-[11px] text-accent hover:underline flex items-center gap-1 cursor-pointer">
              View <ArrowRight size={11} />
            </button>
          }>
          {agentRuns.length ? agentRuns.slice(0, 4).map((r) => (
            <div
              key={r.run_id}
              onClick={() => navigate('/agent')}
              className="flex items-center gap-2 py-1.5 border-b border-border/30 last:border-none cursor-pointer hover:bg-panel-alt/40 px-1 rounded transition-colors">
              <Tag kind={r.status === 'running' ? 'accent'
                : r.status === 'done' ? 'ok' : 'err'}>
                {r.status}
              </Tag>
              <span className="text-[12px] text-text-dim truncate flex-1" title={r.task}>
                {r.task}
              </span>
            </div>
          )) : (
            <div className="text-[12px] text-text-faint py-3 text-center">
              No agent runs launched yet
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

function shortName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

function QuickAction({ icon, label, desc, onClick }: {
  icon: React.ReactNode
  label: string
  desc: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 bg-panel border border-border rounded-xl px-4 py-3.5 text-left hover:border-accent/50 hover:bg-panel-alt transition-all cursor-pointer group shadow-card hover:shadow-lg">
      <div className="w-10 h-10 rounded-xl bg-panel-alt border border-border flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
        {icon}
      </div>
      <div className="min-w-0">
        <div className="text-[13.5px] font-semibold text-text group-hover:text-accent-hi transition-colors">
          {label}
        </div>
        <div className="text-[11px] text-text-faint truncate">{desc}</div>
      </div>
    </button>
  )
}

