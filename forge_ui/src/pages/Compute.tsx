// Compute — GPU telemetry (live ~2s WS push) + torch/CUDA memory probe.
// Keeps a rolling 120-sample history of vram/util/temp/power for charts.

import { Cpu, RefreshCw, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import type { GpuSnapshot } from '../lib/types'
import { Gauge, LineChart } from '../components/charts'
import {
  Btn, Card, IconBtn, KvRow, MetricCard, Progress, Tag,
} from '../components/ui'

const HIST = 120

interface TorchInfo {
  torch?: string
  cuda_available?: boolean
  device?: string
  capability?: string
  allocated_gb?: number
  reserved_gb?: number
  max_allocated_gb?: number
  error?: string
}

interface ComputeInfo extends GpuSnapshot {
  torch?: TorchInfo
}

interface Hist {
  vram: number[]
  util: number[]
  temp: number[]
  power: number[]
}

export default function Compute() {
  const gpu = useForge((s) => s.gpu)
  const refreshStatus = useForge((s) => s.refreshStatus)
  const [info, setInfo] = useState<ComputeInfo | null>(null)
  const [err, setErr] = useState('')
  const [cleaning, setCleaning] = useState(false)
  const [hist, setHist] = useState<Hist>({
    vram: [], util: [], temp: [], power: [],
  })

  const load = async () => {
    try {
      setInfo(await api.get<ComputeInfo>('/api/compute'))
      setErr('')
    } catch (e) {
      setErr(String(e))
    }
  }
  useEffect(() => { void load() }, [])

  // rolling history — appended whenever a new gpu snapshot arrives
  useEffect(() => {
    if (!gpu.available) return
    setHist((h) => ({
      vram: [...h.vram, gpu.vram_pct].slice(-HIST),
      util: [...h.util, gpu.util_pct].slice(-HIST),
      temp: [...h.temp, gpu.temp_c ?? 0].slice(-HIST),
      power: [...h.power, gpu.power_w ?? 0].slice(-HIST),
    }))
  }, [gpu])

  const cleanMemory = async () => {
    setCleaning(true)
    try {
      await api.post('/api/engine/clear_recovery').catch(() => undefined)
      await refreshStatus()
      await load()
    } finally {
      setCleaning(false)
    }
  }

  const torch = info?.torch
  const gbUsed = gpu.vram_used_mb / 1024
  const gbTotal = gpu.vram_total_mb / 1024
  const hasGpu = gpu.available || !!info?.available

  return (
    <div className="h-full overflow-y-auto p-5 space-y-4">
      {/* Hardware spec header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-4 rounded-xl border border-border bg-panel-alt/40 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-accent/15 border border-accent/30 flex items-center justify-center text-accent shrink-0 shadow-[0_0_12px_rgba(122,151,255,0.2)]">
            <Cpu size={22} />
          </div>
          <div>
            <div className="text-[14px] font-bold text-text flex items-center gap-2">
              NVIDIA RTX 5070
              <Tag kind="accent">Blackwell SM120</Tag>
            </div>
            <div className="text-[11px] text-text-faint font-mono mt-0.5">
              12GB GDDR7 • Tensor Cores Gen 5 • NVFP4 / INT8 / BF16 Native
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Btn
            variant="subtle"
            size="sm"
            loading={cleaning}
            onClick={cleanMemory}
            title="Clear CUDA memory fragments and recovery checkpoints">
            <Zap size={13} /> Clear Cache
          </Btn>
          <Btn
            variant="outline"
            size="sm"
            onClick={() => void load()}>
            <RefreshCw size={13} /> Refresh
          </Btn>
        </div>
      </div>

      {/* gauges + metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card title="GPU Workload" className="col-span-2">
          {gpu.available ? (
            <div className="flex items-center justify-around py-1">
              <Gauge value={gpu.vram_pct} caption="VRAM Memory" size={120} />
              <Gauge value={gpu.util_pct} caption="Compute Util" size={120} />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-text-faint">
              <Cpu size={28} strokeWidth={1.2} className="mb-2" />
              <span className="text-[12px]">no GPU telemetry</span>
            </div>
          )}
        </Card>
        <MetricCard
          title="Core Temp"
          value={gpu.temp_c != null ? gpu.temp_c : '—'}
          unit="°C"
          trend={gpu.temp_c != null && gpu.temp_c > 75 ? 'Warning: High Temp' : 'Optimal'}
          spark={hist.temp}
          color="#fbbf24" />
        <MetricCard
          title="Board Power"
          value={gpu.power_w != null ? Math.round(gpu.power_w) : '—'}
          unit={gpu.power_limit_w != null
            ? `/ ${Math.round(gpu.power_limit_w)} W` : 'W'}
          trend={gpu.power_w != null && gpu.power_limit_w != null ? `${Math.round((gpu.power_w / gpu.power_limit_w) * 100)}% load` : undefined}
          spark={hist.power}
          color="#c084fc" />
      </div>

      {/* history charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="VRAM Memory History (%)" subtitle="Rolling 120-sample window">
          <LineChart
            series={[{ name: 'VRAM Allocation', color: '#7a97ff', data: hist.vram }]}
            minY={0} maxY={100} window={HIST} height={130} />
        </Card>
        <Card title="GPU Compute Utilization (%)" subtitle="Real-time SM occupancy">
          <LineChart
            series={[{ name: 'Compute Util', color: '#2dd4bf', data: hist.util }]}
            minY={0} maxY={100} window={HIST} height={130} />
        </Card>
      </div>

      {/* torch + gpu details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card
          title="PyTorch / CUDA Backend"
          actions={
            <IconBtn title="Refresh" onClick={() => void load()}>
              <RefreshCw size={13} />
            </IconBtn>
          }>
          {err && (
            <div className="text-[11.5px] text-err mb-1">{err}</div>
          )}
          {torch?.error ? (
            <div className="text-[12px] text-err">{torch.error}</div>
          ) : (
            <div className="space-y-1">
              <KvRow k="PyTorch Version" mono v={torch?.torch ?? '—'} />
              <KvRow k="CUDA Available" v={torch ? (
                <Tag kind={torch.cuda_available ? 'ok' : 'err'}>
                  {torch.cuda_available ? 'active' : 'unavailable'}
                </Tag>
              ) : '—'} />
              <KvRow k="Device Identifier" v={torch?.device ?? '—'} />
              <KvRow k="Compute Capability" mono v={torch?.capability ?? '—'} />
              <KvRow k="Torch Allocated"
                v={torch?.allocated_gb != null
                  ? `${torch.allocated_gb.toFixed(2)} GB` : '—'} />
              <KvRow k="Torch Reserved"
                v={torch?.reserved_gb != null
                  ? `${torch.reserved_gb.toFixed(2)} GB` : '—'} />
              <KvRow k="Peak Allocated"
                v={torch?.max_allocated_gb != null
                  ? `${torch.max_allocated_gb.toFixed(2)} GB` : '—'} />
            </div>
          )}
        </Card>

        <Card title="Device Telemetry Breakdown">
          {hasGpu ? (
            <div className="space-y-1">
              <KvRow k="Adapter Name" v={gpu.name || info?.name || '—'} />
              <KvRow k="Total VRAM"
                v={`${gbUsed.toFixed(2)} / ${gbTotal.toFixed(1)} GB (${Math.round(gpu.vram_pct)}%)`} />
              <div className="py-2">
                <Progress value={gpu.vram_pct} />
              </div>
              <KvRow k="Compute Load" v={`${Math.round(gpu.util_pct)}%`} />
              <KvRow k="Power Consumption"
                v={gpu.power_w != null
                  ? `${Math.round(gpu.power_w)} / ${gpu.power_limit_w != null ? Math.round(gpu.power_limit_w) : '—'} W`
                  : '—'} />
              <KvRow k="Thermal Sensor"
                v={gpu.temp_c != null ? `${gpu.temp_c}°C` : '—'} />
            </div>
          ) : (
            <div className="text-[12px] text-text-dim py-4 text-center">
              GPU not detected
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
