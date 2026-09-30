// Global store — one WebSocket feeds engine/gpu/task/agent state.
// Mirrors the Qt app's timer-driven refresh model: the WS stream
// replaces the 500ms/2s page refresh loops.

import { create } from 'zustand'
import { api } from './api'
import type {
  AgentEvent,
  AgentRunDetail,
  AgentRunInfo,
  EngineSnapshot,
  GpuSnapshot,
  LoraStatus,
  RunSnapshot,
  SelfPlayStatusPayload,
  StatusResponse,
  TaskInfo,
  WsEvent,
} from './types'

interface ApprovalReq {
  id: string
  kind: string
  detail: Record<string, unknown>
}

interface ForgeState {
  connected: boolean
  engine: EngineSnapshot
  gpu: GpuSnapshot
  lora: LoraStatus
  runs: RunSnapshot[]
  tasks: TaskInfo[]
  agentRuns: AgentRunInfo[]
  agentRunEvents: Record<string, AgentEvent[]>
  selectedAgentRun: string | null
  approvals: ApprovalReq[]
  engineProgress: string | null
  taskLines: Record<string, string[]>
  selfplayEvents: Record<string, unknown>[]
  selfplayStatus: SelfPlayStatusPayload | null
  timerFired: { timer_id: string; label: string; message: string }[]
  boot: { running: boolean; status: string; output: string }

  refreshStatus: () => Promise<void>
  respondApproval: (id: string, granted: boolean) => Promise<void>
  agentRespond: (runId: string, granted: boolean) => Promise<void>
  agentMessage: (runId: string, message: string) => Promise<void>
  deleteAgentRun: (runId: string) => Promise<void>
  selectAgentRun: (id: string | null) => void
  hydrateAgentRun: (id: string) => Promise<void>
  clearRunEvents: (runId: string) => void
}

const IDLE_ENGINE: EngineSnapshot = { state: 'idle', info: {}, error: '', busy: false }
const IDLE_GPU: GpuSnapshot = {
  available: false, name: '', vram_used_mb: 0, vram_total_mb: 0,
  vram_pct: 0, util_pct: 0, temp_c: null, power_w: null, power_limit_w: null,
}
const IDLE_LORA: LoraStatus = {
  mode: 'chat', pinned: null, current: null, busy: false,
  target_presets: [], modes: [],
}

export const useForge = create<ForgeState>((set) => ({
  connected: false,
  engine: IDLE_ENGINE,
  gpu: IDLE_GPU,
  lora: IDLE_LORA,
  runs: [],
  tasks: [],
  agentRuns: [],
  agentRunEvents: {},
  selectedAgentRun: null,
  approvals: [],
  engineProgress: null,
  taskLines: {},
  selfplayEvents: [],
  selfplayStatus: null,
  timerFired: [],
  boot: { running: false, status: '', output: '' },

  refreshStatus: async () => {
    const s = await api.get<StatusResponse>('/api/status')
    set({
      engine: s.engine, gpu: s.gpu, lora: s.lora, runs: s.runs,
      tasks: s.tasks, agentRuns: s.agent_runs,
    })
  },

  respondApproval: async (id, granted) => {
    await api.post(`/api/approvals/${id}`, { granted })
    set((st) => ({ approvals: st.approvals.filter((a) => a.id !== id) }))
  },

  agentRespond: async (runId, granted) => {
    await api.post(`/api/agent/runs/${runId}/respond`, { granted })
  },

  agentMessage: async (runId, message) => {
    await api.post(`/api/agent/runs/${runId}/message`, { message })
  },

  deleteAgentRun: async (runId) => {
    await api.del(`/api/agent/runs/${runId}`).catch(() => undefined)
    set((st) => {
      const events = { ...st.agentRunEvents }
      delete events[runId]
      return {
        agentRuns: st.agentRuns.filter((r) => r.run_id !== runId),
        agentRunEvents: events,
        selectedAgentRun:
          st.selectedAgentRun === runId ? null : st.selectedAgentRun,
      }
    })
  },

  selectAgentRun: (id) => {
    set({ selectedAgentRun: id })
    if (id) useForge.getState().hydrateAgentRun(id).catch(() => undefined)
  },

  hydrateAgentRun: async (id) => {
    const d = await api.get<AgentRunDetail>(`/api/agent/runs/${id}`)
    if (!d.ok) return
    set((st) => {
      const live = st.agentRunEvents[id] ?? []
      const lastSeq = d.events.length ? d.events[d.events.length - 1].seq : -1
      const extra = live.filter((e) => e.seq > lastSeq)
      return {
        agentRunEvents: { ...st.agentRunEvents, [id]: [...d.events, ...extra] },
      }
    })
  },

  clearRunEvents: (runId) => set((st) => {
    const next = { ...st.agentRunEvents }
    delete next[runId]
    return { agentRunEvents: next }
  }),
}))

function handleEvent(evt: WsEvent) {
  const st = useForge.getState()
  const p = evt.payload as Record<string, unknown>
  switch (evt.type) {
    case 'gpu':
      useForge.setState({ gpu: p as unknown as GpuSnapshot })
      break
    case 'engine':
      useForge.setState({ engine: p as unknown as EngineSnapshot, engineProgress: null })
      break
    case 'engine_progress':
      useForge.setState({ engineProgress: (p as { message: string }).message })
      break
    case 'engine_error':
      useForge.setState((s) => ({
        engine: { ...s.engine, error: (p as { error: string }).error },
      }))
      break
    case 'task_added':
    case 'task_finished':
    case 'task_status':
    case 'task_removed':
      st.refreshStatus().catch(() => undefined)
      break
    case 'task_line': {
      const { task_id, line } = p as { task_id: string; line: string }
      useForge.setState((s) => {
        const lines = [...(s.taskLines[task_id] ?? []), line].slice(-4000)
        return { taskLines: { ...s.taskLines, [task_id]: lines } }
      })
      break
    }
    case 'agent_event': {
      const ae = evt.payload as unknown as AgentEvent
      useForge.setState((s) => {
        const list = s.agentRunEvents[ae.run_id] ?? []
        const lastSeq = list.length ? list[list.length - 1].seq : -1
        if (ae.seq <= lastSeq) return s  // already hydrated
        return {
          agentRunEvents: {
            ...s.agentRunEvents,
            [ae.run_id]: [...list, ae].slice(-2000),
          },
        }
      })
      if (ae.kind === 'started' || ae.kind === 'finished' || ae.kind === 'failed') {
        st.refreshStatus().catch(() => undefined)
      }
      break
    }
    case 'approval':
      useForge.setState((s) => ({
        approvals: [...s.approvals, evt.payload as ApprovalReq],
      }))
      break
    case 'lora':
      st.refreshStatus().catch(() => undefined)
      break
    case 'timer':
      if ((p as { kind: string }).kind === 'fired') {
        useForge.setState((s) => ({
          timerFired: [...s.timerFired, evt.payload as never].slice(-20),
        }))
      }
      break
    case 'selfplay_event': {
      useForge.setState((s) => ({
        selfplayEvents: [...s.selfplayEvents, p].slice(-2000),
      }))
      break
    }
    case 'selfplay_status':
      useForge.setState({ selfplayStatus: p as unknown as SelfPlayStatusPayload })
      break
    case 'boot': {
      const b = p as { kind: string; message?: string; data?: { output?: string } }
      useForge.setState((s) => ({
        boot: {
          running: b.kind === 'status' || b.kind === 'result',
          status: b.message ?? b.kind,
          output: b.data?.output
            ? (s.boot.output + b.data.output)
            : s.boot.output,
        },
      }))
      break
    }
    default:
      break
  }
}

let ws: WebSocket | null = null
let retryMs = 1000

export function connectWs() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return
  }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  ws = new WebSocket(`${proto}://${location.host}/ws`)
  ws.onopen = () => {
    retryMs = 1000
    useForge.setState({ connected: true })
    useForge.getState().refreshStatus().catch(() => undefined)
  }
  ws.onmessage = (e) => {
    try {
      handleEvent(JSON.parse(e.data) as WsEvent)
    } catch { /* malformed frame */ }
  }
  ws.onclose = () => {
    useForge.setState({ connected: false })
    ws = null
    setTimeout(connectWs, retryMs)
    retryMs = Math.min(retryMs * 1.5, 10000)
  }
  ws.onerror = () => ws?.close()
}
