// API response types (mirror forge_gui_server routes)

export interface EngineSnapshot {
  state: 'idle' | 'loading' | 'ready' | 'error'
  info: {
    checkpoint?: string
    config_name?: string
    load_s?: number
    device?: string
    dtype?: string
    use_compile?: boolean
    activation?: Record<string, unknown>
  }
  error: string
  busy: boolean
}

export interface GpuSnapshot {
  available: boolean
  name: string
  vram_used_mb: number
  vram_total_mb: number
  vram_pct: number
  util_pct: number
  temp_c: number | null
  power_w: number | null
  power_limit_w: number | null
}

export interface RunSnapshot {
  id: string
  name: string
  status_file: string
  status: string
  step: number
  max_steps: number
  loss: number
  lr: number
  vram_gb: number
  method: string
  updated_at: number
  heartbeat_age_s: number
  extra: Record<string, unknown>
  progress_pct: number
  is_live: boolean
}

export interface TaskInfo {
  id: string
  name: string
  command: string[]
  pid: number
  status: 'starting' | 'running' | 'done' | 'crashed' | 'killed'
  started_at: number
  ended_at: number
  exit_code: number | null
  log_path: string | null
  lines: string[]
  elapsed_s: number
  is_live: boolean
}

export interface ModelEntry {
  name: string
  path: string
  size_bytes: number
  size_label: string
  ext: string
  config_name: string | null
  config: Record<string, unknown>
  meta: Record<string, unknown>
  modified: number
  is_safetensors: boolean
  is_flux: boolean
  is_lora: boolean
}

export interface ConfigEntry {
  name: string
  d_model: number
  n_layers: number
  n_heads: number
  n_kv_heads: number | null
  vocab_size: number
  attn_type: string
  ffn_type: string
  max_seq_len: number
  params_label: string
}

export interface LoraEntry {
  name: string
  path: string
  size_bytes: number
  size_label: string
  modified: number
  rank: number | null
  n_tensors: number
  n_params: number
  dtype: string
  base_hint: string
  header_error: string
  category: string
}

export interface LoraStatus {
  mode: string
  pinned: string | null
  current: string | null
  busy: boolean
  target_presets: string[]
  modes: string[]
}

export interface ChatSummary {
  id: string
  title: string
  model: string
  created_at: number
  updated_at: number
  n_messages: number
  good: number
  bad: number
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  rating?: 'good' | 'bad' | null
  ts?: number
  image?: string
  tool_calls?: ToolCall[] | null
  name?: string
  reasoning_content?: string
}

export interface ToolCall {
  name: string
  arguments?: Record<string, unknown>
}

export interface ToolResultRec {
  name: string
  args: Record<string, unknown>
  ok: boolean
  elapsed_s: number
  result: Record<string, unknown>
}

export interface Conversation {
  id: string
  title: string
  model: string
  created_at: number
  updated_at: number
  messages: ChatMessage[]
}

export interface AgentRunInfo {
  run_id: string
  task: string
  status: 'running' | 'done' | 'cancelled' | 'failed'
  started_at: number
  error: string
  project: string
  workspace: string
  rounds: number
  n_events: number
}

export interface AgentEvent {
  run_id: string
  seq: number
  kind:
    | 'created' | 'started' | 'round_started' | 'prompt_rendered'
    | 'raw_output' | 'text' | 'tool_call' | 'tool_result'
    | 'approval_requested' | 'user_message' | 'finished' | 'failed'
  round: number | null
  data: unknown
}

export interface AgentRunDetail {
  ok: boolean
  run: AgentRunInfo
  events: AgentEvent[]
}

export interface WsEvent {
  seq: number
  ts: number
  type: string
  payload: unknown
}

export interface StatusResponse {
  engine: EngineSnapshot
  gpu: GpuSnapshot
  lora: LoraStatus
  runs: RunSnapshot[]
  tasks: TaskInfo[]
  agent_runs: AgentRunInfo[]
  ws_clients: number
}

export interface ActivationField {
  name: string
  kind: 'combo' | 'int' | 'opt_int' | 'opt_float' | 'bool'
  label: string
  tooltip: string
  default: unknown
  options: { value: string; label: string; tip: string }[]
  lo: number
  hi: number
  step: number
  decimals: number
  suffix: string
}

export interface ActivationCatalog {
  categories: { name: string; fields: ActivationField[] }[]
  presets: { name: string; desc: string }[]
  default: Record<string, unknown>
}

export interface Dataset {
  path: string
  name: string
  examples: number
  size_bytes: number
  modified: number
}

export interface ProcessPreset {
  name: string
  script: string
  description: string
  arg_defaults: Record<string, string>
}

export interface SelfPlayEvent {
  ts?: number
  kind?: string
  [key: string]: unknown
}

export interface SelfPlayStatusPayload {
  status: Record<string, unknown>
  heartbeat_age_s: number | null
  heartbeat_stalled: boolean | null
}
