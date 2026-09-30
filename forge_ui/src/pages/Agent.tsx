// Agent page — chat-based agentic runs.
// Left: run list + new-run config (real tool defs from /api/agent/tools).
// Right: the selected run as a round-grouped chat thread — tool calls and
// their results render as one merged activity card; approvals pin an
// interactive card; the composer steers a live run or starts a new one.

import { clsx } from 'clsx'
import {
  Bot, Check, FolderOpen,
  Square, Trash2, Wrench, X,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import {
  CopyBtn, SegmentedBody, ToolActivityCard, UserBubble,
} from '../components/chat/blocks'
import { Composer } from '../components/chat/Composer'
import { Markdown } from '../components/chat/Markdown'
import { ScrollFeed } from '../components/chat/ScrollFeed'
import {
  Btn, Card, Check as CheckBox, Chip, EmptyState, Field, IconBtn,
  Input, Modal, SearchInput, Select, Spinner, Tag,
} from '../components/ui'
import { api } from '../lib/api'
import { groupTools, type ToolDef } from '../lib/agentTools'
import { fmtDur, relTime } from '../lib/format'
import { useForge } from '../lib/store'
import type { AgentEvent, AgentRunInfo, ToolResultRec } from '../lib/types'

const APPROVAL_MODES = [
  { v: 'destructive', label: 'destructive only' },
  { v: 'all', label: 'all side-effects' },
  { v: 'none', label: 'none (YOLO)' },
]
const ROUND_MODES = [
  { v: 'auto', label: 'auto — model decides' },
  { v: '8', label: '8 rounds' },
  { v: '16', label: '16 rounds' },
  { v: '32', label: '32 rounds' },
  { v: '64', label: '64 rounds' },
]

// last-resort list if /api/agent/tools is unreachable
const FALLBACK_TOOLS = [
  'read_file', 'write_file', 'edit_file', 'search_replace', 'project_search_replace',
  'run_python', 'run_cmd', 'run_tests', 'list_dir', 'find_files', 'git_status',
  'git_diff', 'web_search', 'checkpoint_compare', 'undo_edit', 'delete_file',
  'create_file', 'rename_file', 'git_revert', 'git_stash', 'git_branch',
  'search_repo', 'read_file_lines', 'spawn_subagent', 'lora_status',
  'lora_apply', 'lora_remove', 'lora_merge', 'lora_keep_for_task',
  'lora_edit_training_data', 'start_lora_training', 'check_lora_training_progress',
  'library_list', 'library_install', 'backup_list', 'backup_restore', 'backup_delete',
].map((name) => ({ name, description: '' }))

const CFG_KEY = 'forge.agent.cfg.v1'
const NO_EVENTS: AgentEvent[] = []

interface AgentCfg {
  rounds: string
  approval: string
  project: string
}

function loadCfg(): AgentCfg {
  try {
    const j = JSON.parse(localStorage.getItem(CFG_KEY) ?? '{}')
    return { rounds: 'auto', approval: 'destructive', project: '', ...j }
  } catch { return { rounds: 'auto', approval: 'destructive', project: '' } }
}

const STARTERS = [
  { title: 'Mamba-3 Research', desc: 'Web search & document generation', task: 'Research the Mamba-3 architecture and write a summary.md in the workspace.' },
  { title: 'Benchmark Script', desc: 'Python performance tests with output', task: 'Create a Python script that benchmarks JSON vs CSV parsing performance with charts.' },
  { title: 'Codebase Tour', desc: 'Scan workspace & summarize layout', task: 'Explore this workspace and explain its architecture and key modules.' },
  { title: 'Refactoring Audit', desc: 'Find dead code or missing types', task: 'Scan the codebase for unused imports or duplicate helper functions and propose a cleanup plan.' },
]

export default function Agent() {
  const engine = useForge((s) => s.engine)
  const agentRuns = useForge((s) => s.agentRuns)
  const runEvents = useForge((s) => s.agentRunEvents)
  const selectedId = useForge((s) => s.selectedAgentRun)
  const selectRun = useForge((s) => s.selectAgentRun)
  const respond = useForge((s) => s.agentRespond)
  const sendMessage = useForge((s) => s.agentMessage)
  const deleteRun = useForge((s) => s.deleteAgentRun)
  const clearRunEvents = useForge((s) => s.clearRunEvents)

  const [input, setInput] = useState('')
  const [cfg, setCfg] = useState<AgentCfg>(loadCfg)
  const [toolsOpen, setToolsOpen] = useState(false)
  const [toolQuery, setToolQuery] = useState('')
  const [tools, setTools] = useState<Set<string>>(new Set())
  const [toolDefs, setToolDefs] = useState<ToolDef[]>(FALLBACK_TOOLS)
  const [verbose, setVerbose] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    try { localStorage.setItem(CFG_KEY, JSON.stringify(cfg)) }
    catch { /* quota */ }
  }, [cfg])

  // real tool catalog from the harness
  useEffect(() => {
    api.get<{ tools: ToolDef[] }>('/api/agent/tools')
      .then((r) => { if (r.tools?.length) setToolDefs(r.tools) })
      .catch(() => undefined)
  }, [])

  // pick a run to display: explicit selection → active run → newest
  const run = useMemo(() => {
    const sel = agentRuns.find((r) => r.run_id === selectedId)
    return sel ?? agentRuns.find((r) => r.status === 'running') ?? agentRuns[0]
  }, [agentRuns, selectedId])

  const running = run?.status === 'running'
  const ready = engine.state === 'ready'

  // 1s ticker for the live elapsed readout
  useEffect(() => {
    if (!running) return
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [running])

  const events = run ? runEvents[run.run_id] ?? NO_EVENTS : NO_EVENTS
  const nodes = useMemo(() => buildNodes(events, verbose), [events, verbose])
  const stats = useMemo(() => runStats(events), [events])

  const pendingApproval = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i]
      if (e.kind === 'approval_requested') {
        const resolved = events.slice(i + 1).some(
          (x) => x.kind === 'tool_result' || x.kind === 'finished' || x.kind === 'failed')
        if (!resolved) return e
      }
    }
    return null
  }, [events])

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return
    setError('')
    setInput('')
    setBusy(true)
    try {
      if (running && run) {
        await sendMessage(run.run_id, text)
      } else {
        const r = await api.post<{ ok: boolean; run_id?: string; error?: string }>(
          '/api/agent/runs', {
            task: text,
            project: cfg.project || undefined,
            max_rounds: cfg.rounds === 'auto' ? null : Number(cfg.rounds),
            approval_mode: cfg.approval,
            enabled_tools: tools.size ? [...tools] : undefined,
          })
        if (!r.ok) throw new Error(r.error || 'failed to start')
        if (r.run_id) selectRun(r.run_id)
      }
    } catch (e) {
      setError(String(e))
      setInput(text)
    } finally { setBusy(false) }
  }

  const cancel = () => {
    if (running && run) {
      void api.post(`/api/agent/runs/${run.run_id}/cancel`).catch(() => undefined)
    }
  }

  const toolGroups = useMemo(() => {
    const q = toolQuery.toLowerCase()
    const defs = q
      ? toolDefs.filter((t) => t.name.includes(q) || t.description.toLowerCase().includes(q))
      : toolDefs
    return groupTools(defs)
  }, [toolDefs, toolQuery])

  const toggleTool = (t: string) => {
    setTools((s) => {
      const n = new Set(s)
      if (n.has(t)) n.delete(t); else n.add(t)
      return n
    })
  }

  const elapsed = run
    ? (running ? now / 1000 - run.started_at : stats.elapsedS ?? 0)
    : 0

  return (
    <div className="flex h-full min-h-0">
      {/* ---- left: new run + history ---- */}
      <div className="w-[292px] shrink-0 border-r border-border overflow-y-auto p-3 space-y-3 bg-bg-alt/40">
        <Card title="New run">
          <div className="space-y-2.5">
            <Field label="Project"
              hint="Subfolder under ForgeAI_Projects — blank slugs the task">
              <Input value={cfg.project}
                onChange={(e) => setCfg((c) => ({ ...c, project: e.target.value }))}
                placeholder="name → ForgeAI_Projects/<name>" className="w-full" />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Stop after">
                <Select value={cfg.rounds} className="w-full"
                  onChange={(e) => setCfg((c) => ({ ...c, rounds: e.target.value }))}>
                  {ROUND_MODES.map((m) =>
                    <option key={m.v} value={m.v}>{m.label}</option>)}
                </Select>
              </Field>
              <Field label="Approvals">
                <Select value={cfg.approval} className="w-full"
                  onChange={(e) => setCfg((c) => ({ ...c, approval: e.target.value }))}>
                  {APPROVAL_MODES.map((m) =>
                    <option key={m.v} value={m.v}>{m.label}</option>)}
                </Select>
              </Field>
            </div>

            {/* tool harness trigger */}
            <div className="flex items-center justify-between p-2.5 rounded-xl border border-border bg-panel-alt/50">
              <div className="min-w-0">
                <div className="text-[12px] font-semibold text-text flex items-center gap-1.5">
                  <Wrench size={13} className="text-accent" /> Tool Harness
                </div>
                <div className="text-[11px] text-text-faint mt-0.5">
                  {tools.size ? `${tools.size} of ${toolDefs.length} enabled` : `all ${toolDefs.length} enabled`}
                </div>
              </div>
              <Btn variant="outline" size="xs" onClick={() => setToolsOpen(true)}>
                Configure
              </Btn>
            </div>
          </div>
        </Card>

        <Card title={`Runs (${agentRuns.length})`}>
          <div className="space-y-1">
            {agentRuns.slice(0, 30).map((r) => (
              <RunRow key={r.run_id} run={r}
                active={r.run_id === run?.run_id}
                onClick={() => selectRun(r.run_id)}
                onDelete={() => void deleteRun(r.run_id)} />
            ))}
            {!agentRuns.length && (
              <div className="text-[11.5px] text-text-faint">no runs yet</div>
            )}
          </div>
        </Card>
      </div>

      {/* ---- right: run thread ---- */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* run header */}
        <div className="flex items-center gap-2 px-5 h-11 border-b border-border shrink-0">
          {run ? (
            <>
              <StatusDot run={run} />
              <span className="text-[13px] font-medium truncate max-w-[38%]"
                title={run.task}>{run.task}</span>
              {run.project && (
                <span title={run.workspace}>
                  <Tag kind="accent"><FolderOpen size={11} /> {run.project}</Tag>
                </span>
              )}
              <span className="text-[10.5px] text-text-faint tabular-nums shrink-0">
                {fmtDur(elapsed)}
                {run.rounds > 0 && ` · ${run.rounds} round${run.rounds === 1 ? '' : 's'}`}
                {stats.tools > 0 && ` · ${stats.tools} tools`}
                {stats.failed > 0 && (
                  <span className="text-err"> ({stats.failed} failed)</span>)}
              </span>
            </>
          ) : (
            <span className="text-[13px] font-medium text-text-dim">Agent</span>
          )}
          <div className="ml-auto flex items-center gap-1.5">
            {run && (
              <Chip on={verbose} onClick={() => setVerbose(!verbose)}
                title="Show raw model output + rendered prompts">
                verbose
              </Chip>
            )}
            {run && (
              <CopyBtn text={run.workspace} className="!w-7 !h-7" />
            )}
            {running && (
              <Btn variant="danger" onClick={cancel}>
                <Square size={12} /> Stop
              </Btn>
            )}
            {run && !running && (
              <IconBtn title="Delete run" className="hover:!text-err"
                onClick={() => void deleteRun(run.run_id)}>
                <Trash2 size={13} />
              </IconBtn>
            )}
            {run && events.length > 0 && (
              <Btn variant="ghost"
                onClick={() => clearRunEvents(run.run_id)}>
                Clear
              </Btn>
            )}
          </div>
        </div>

        <ScrollFeed watch={`${run?.run_id ?? ''}:${events.length}`}
          className="px-5 py-4">
          <div className="max-w-[820px] mx-auto space-y-3">
            {!run && (
              <EmptyState
                icon={<Bot size={38} strokeWidth={1.5} />}
                title={ready ? 'Agent Workspace Idle' : 'Engine not loaded'}
                desc={ready
                  ? 'Describe an objective below. The agent decomposes goals, invokes file/shell/web tools, and streams every step live.'
                  : 'Load a checkpoint on the Models page to initialize the agent harness.'}
                action={ready ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-xl text-left mt-3">
                    {STARTERS.map((s) => (
                      <button
                        key={s.title}
                        onClick={() => setInput(s.task)}
                        className="p-3 rounded-xl border border-border/80 bg-panel/60 hover:bg-panel-alt hover:border-accent/50 transition-all text-left group cursor-pointer shadow-sm">
                        <div className="text-[12.5px] font-semibold text-text group-hover:text-accent-hi transition-colors flex items-center justify-between">
                          {s.title}
                          <Bot size={12} className="text-text-faint group-hover:text-accent transition-colors" />
                        </div>
                        <div className="text-[11px] text-text-dim mt-0.5 leading-snug">
                          {s.desc}
                        </div>
                      </button>
                    ))}
                  </div>
                ) : undefined} />
            )}
            {nodes.map((n) => (
              <EventNode key={n.key} node={n}
                pendingApproval={pendingApproval}
                onRespond={(g) => run && void respond(run.run_id, g)} />
            ))}
            {running && !pendingApproval && (
              <div className="flex items-center gap-2 text-[12px] text-text-faint">
                <Spinner /> working{run.rounds > 0 ? ` — round ${run.rounds}` : ''}…
              </div>
            )}
            {error && (
              <div className="text-[12.5px] text-err bg-err/10 border border-err/30 rounded-lg px-3 py-2">
                {error}
              </div>
            )}
          </div>
        </ScrollFeed>

        {/* composer — steers a live run or starts a new one */}
        <div className="px-5 pb-4 pt-1.5">
          <div className="max-w-[820px] mx-auto">
            <Composer
              value={input}
              onChange={setInput}
              onSubmit={() => void send()}
              busy={false}
              disabled={!ready && !running}
              autoFocus
              sendLabel={running ? 'Steer' : 'Run'}
              submitTitle={running
                ? 'Inject this message before the next round'
                : 'Start a new agent run'}
              placeholder={running
                ? 'Steer the agent — your message joins the next round…'
                : ready
                  ? 'Describe the task — a project folder is created in ForgeAI_Projects…'
                  : 'Load a model first…'}
              meta={running ? (
                <Tag kind="accent"><Spinner /> live</Tag>
              ) : undefined}
            />
          </div>
        </div>
      </div>

      {/* ---- Tool Harness Configuration Modal ---- */}
      <Modal
        open={toolsOpen}
        onClose={() => setToolsOpen(false)}
        title="Agent Tool Harness Catalog"
        maxWidth="max-w-2xl">
        <div className="space-y-3.5">
          <div className="flex items-center gap-3">
            <SearchInput
              value={toolQuery}
              onChange={setToolQuery}
              placeholder="Search available tools…"
              className="flex-1"
            />
            <div className="flex items-center gap-2">
              <Btn variant="subtle" size="xs" onClick={() => setTools(new Set(toolDefs.map((t) => t.name)))}>
                Enable all ({toolDefs.length})
              </Btn>
              <Btn variant="ghost" size="xs" onClick={() => setTools(new Set())}>
                Reset to default
              </Btn>
            </div>
          </div>

          <div className="space-y-3 max-h-[55vh] overflow-y-auto pr-1">
            {toolGroups.map((g) => (
              <div key={g.label} className="p-3 rounded-xl border border-border/80 bg-panel/40">
                <div className="flex items-center justify-between pb-2 border-b border-border/40 mb-2">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-text-faint">
                    {g.label}
                  </div>
                  <span className="text-[10px] text-text-faint font-mono">
                    {g.tools.length} tools
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {g.tools.map((t) => (
                    <CheckBox
                      key={t.name}
                      label={
                        <div className="min-w-0">
                          <span className="font-mono text-[11.5px] font-medium text-text block truncate">
                            {t.name}
                          </span>
                          {t.description && (
                            <span className="text-[10.5px] text-text-faint block truncate">
                              {t.description}
                            </span>
                          )}
                        </div>
                      }
                      checked={tools.size === 0 || tools.has(t.name)}
                      onChange={() => toggleTool(t.name)}
                      className="p-1 rounded-lg hover:bg-panel-alt/60 transition-colors"
                      title={t.description}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-border/50">
            <span className="text-[11.5px] text-text-faint">
              {tools.size ? `${tools.size} custom tools enabled` : 'All tools available to the model'}
            </span>
            <Btn variant="primary" size="sm" onClick={() => setToolsOpen(false)}>
              Apply Tools
            </Btn>
          </div>
        </div>
      </Modal>
    </div>
  )
}

/* ---------- run list ---------- */

function StatusDot({ run }: { run: AgentRunInfo }) {
  const cls = run.status === 'running' ? 'bg-accent pulse-dot'
    : run.status === 'done' ? 'bg-ok'
    : run.status === 'cancelled' ? 'bg-warn' : 'bg-err'
  return <span className={clsx('w-2 h-2 rounded-full shrink-0', cls)}
    title={run.status} />
}

function RunRow({ run, active, onClick, onDelete }: {
  run: AgentRunInfo
  active: boolean
  onClick: () => void
  onDelete: () => void
}) {
  return (
    <div
      onClick={onClick}
      className={clsx(
        'w-full flex items-center gap-2 py-1.5 px-2 rounded-lg text-left transition-colors cursor-pointer group',
        active ? 'bg-accent/15 border border-accent/25'
          : 'hover:bg-panel-alt/60 border border-transparent')}>
      <StatusDot run={run} />
      <div className="min-w-0 flex-1">
        <div className="text-[12px] text-text-dim truncate" title={run.task}>
          {run.project || run.task}
        </div>
        <div className="text-[10px] text-text-faint flex gap-1.5">
          {run.project && <span className="truncate max-w-[110px]">{run.task}</span>}
          <span>{relTime(run.started_at)}</span>
          {run.rounds > 0 && <span>· {run.rounds}r</span>}
        </div>
      </div>
      <IconBtn title="Delete run"
        className="!w-5 !h-5 opacity-0 group-hover:opacity-100 hover:!text-err shrink-0"
        onClick={(e) => { e.stopPropagation(); onDelete() }}>
        <X size={11} />
      </IconBtn>
    </div>
  )
}

/* ---------- event → display nodes ---------- */

type Node =
  | { t: 'user'; text: string; key: string }
  | { t: 'text'; raw: string; key: string }
  | { t: 'tool'; call: { name: string; args: string }; result?: ToolResultRec; key: string }
  | { t: 'round'; n: number; key: string }
  | { t: 'approval'; evt: AgentEvent; key: string }
  | { t: 'end'; ok: boolean; data: Record<string, unknown>; key: string }
  | { t: 'raw'; label: string; text: string; key: string }

type ToolNode = Extract<Node, { t: 'tool' }>

function buildNodes(events: AgentEvent[], verbose: boolean): Node[] {
  const nodes: Node[] = []
  const pendingTools: ToolNode[] = [] // tool nodes awaiting a result
  for (const e of events) {
    const d = (e.data ?? {}) as Record<string, unknown>
    switch (e.kind) {
      case 'started':
        nodes.push({ t: 'user', text: String(d.task ?? ''), key: `e${e.seq}` })
        break
      case 'user_message':
        nodes.push({ t: 'user', text: String(e.data ?? ''), key: `e${e.seq}` })
        break
      case 'round_started':
        if (e.round != null && e.round > 0) {
          nodes.push({ t: 'round', n: e.round + 1, key: `e${e.seq}` })
        }
        break
      case 'text':
        if (String(d.text ?? '').trim()) {
          nodes.push({ t: 'text', raw: String(d.text), key: `e${e.seq}` })
        }
        break
      case 'tool_call': {
        const n: ToolNode = {
          t: 'tool',
          call: {
            name: String(d.name ?? 'tool'),
            args: JSON.stringify(d.arguments ?? d.args ?? {}, null, 2),
          },
          key: `e${e.seq}`,
        }
        pendingTools.push(n)
        nodes.push(n)
        break
      }
      case 'tool_result': {
        const rec = d as unknown as ToolResultRec
        const byName = pendingTools.findIndex((n) => n.call.name === rec.name)
        const idx = byName >= 0 ? byName : (pendingTools.length ? 0 : -1)
        if (idx >= 0) {
          pendingTools[idx].result = rec
          pendingTools.splice(idx, 1)
        } else {
          nodes.push({
            t: 'tool',
            call: { name: rec.name, args: '' },
            result: rec,
            key: `e${e.seq}`,
          })
        }
        break
      }
      case 'approval_requested':
        nodes.push({ t: 'approval', evt: e, key: `e${e.seq}` })
        break
      case 'finished':
        nodes.push({ t: 'end', ok: true, data: d, key: `e${e.seq}` })
        break
      case 'failed':
        nodes.push({ t: 'end', ok: false, data: d, key: `e${e.seq}` })
        break
      case 'raw_output':
        if (verbose) {
          nodes.push({ t: 'raw', label: 'raw output', text: String(e.data ?? ''), key: `e${e.seq}` })
        }
        break
      case 'prompt_rendered':
        if (verbose) {
          nodes.push({ t: 'raw', label: 'prompt', text: String(e.data ?? ''), key: `e${e.seq}` })
        }
        break
      default:
        break
    }
  }
  return nodes
}

function runStats(events: AgentEvent[]) {
  let tools = 0, failed = 0, elapsedS: number | undefined
  for (const e of events) {
    if (e.kind === 'tool_result') {
      tools++
      if ((e.data as ToolResultRec).ok === false) failed++
    } else if (e.kind === 'finished') {
      const d = e.data as Record<string, unknown>
      if (typeof d.elapsed_s === 'number') elapsedS = d.elapsed_s
    }
  }
  return { tools, failed, elapsedS }
}

/* ---------- node rendering ---------- */

function EventNode({ node, pendingApproval, onRespond }: {
  node: Node
  pendingApproval: AgentEvent | null
  onRespond: (granted: boolean) => void
}) {
  switch (node.t) {
    case 'user':
      return <UserBubble text={node.text} />
    case 'text':
      return (
        <div className="max-w-[92%] min-w-0">
          <SegmentedBody raw={node.raw} showThinking />
        </div>
      )
    case 'tool': {
      const rec = node.result
      return (
        <div className="max-w-[92%]">
          <ToolActivityCard
            name={node.call.name}
            args={node.call.args}
            running={!rec}
            result={rec && {
              ok: rec.ok,
              elapsed_s: rec.elapsed_s,
              text: String(rec.result?.text ?? JSON.stringify(rec.result)),
            }} />
        </div>
      )
    }
    case 'round':
      return (
        <div className="flex items-center gap-2 pt-2">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-text-faint">
            round {node.n}
          </span>
          <div className="flex-1 h-px bg-border" />
        </div>
      )
    case 'approval': {
      const isPending = pendingApproval?.seq === node.evt.seq
      const d = node.evt.data as Record<string, unknown>
      const name = String(d.tool ?? d.name ?? 'tool')
      if (!isPending) {
        return (
          <div className="flex items-center gap-2 text-[11px] text-warn/80">
            <Tag kind="warn">approval</Tag> {name}
          </div>
        )
      }
      return (
        <div className="bg-warn/10 border border-warn/40 rounded-card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Tag kind="warn">approval required</Tag>
            <span className="text-[12.5px] font-medium font-mono">{name}</span>
          </div>
          <pre className="text-[11.5px] text-text-dim font-mono whitespace-pre-wrap max-h-40 overflow-y-auto mb-3">
            {JSON.stringify(d.args, null, 2)}
          </pre>
          <div className="flex gap-2">
            <Btn variant="primary" onClick={() => onRespond(true)}>
              <Check size={13} /> Approve
            </Btn>
            <Btn variant="danger" onClick={() => onRespond(false)}>
              <X size={13} /> Deny
            </Btn>
          </div>
        </div>
      )
    }
    case 'end': {
      if (!node.ok) {
        return (
          <div className="bg-err/10 border border-err/30 rounded-lg px-3 py-2 text-[12.5px] text-err">
            {String(node.data.error || 'run failed')}
          </div>
        )
      }
      const d = node.data
      const sandbox = (d.sandbox ?? {}) as Record<string, unknown>
      const toolsUsed = (sandbox.tools_used ?? []) as string[]
      return (
        <div className="bg-ok/10 border border-ok/30 rounded-card px-4 py-3">
          <div className="flex items-center gap-2 text-[12.5px] text-ok font-medium">
            <Check size={13} /> run finished
            <span className="text-text-faint font-normal">
              {typeof d.rounds === 'number' && `${d.rounds} rounds`}
              {typeof d.elapsed_s === 'number' && ` · ${fmtDur(d.elapsed_s)}`}
              {typeof sandbox.n_calls === 'number' && ` · ${sandbox.n_calls} calls`}
            </span>
          </div>
          {Boolean(d.content) && (
            <div className="mt-2">
              <Markdown text={String(d.content)} className="!text-[12.5px]" />
            </div>
          )}
          {toolsUsed.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {toolsUsed.slice(0, 12).map((t) => (
                <Tag key={t} kind="idle" className="!text-[10px] font-mono">{t}</Tag>
              ))}
              {toolsUsed.length > 12 && (
                <span className="text-[10px] text-text-faint">+{toolsUsed.length - 12}</span>)}
            </div>
          )}
          {typeof d.safety === 'string' && d.safety !== 'no safety violations' && (
            <div className="text-[11px] text-warn mt-2">{d.safety}</div>
          )}
        </div>
      )
    }
    case 'raw':
      return (
        <details className="border border-border rounded-lg max-w-[92%]">
          <summary className="px-3 py-1.5 text-[11px] text-text-faint cursor-pointer hover:text-text-dim select-none">
            {node.label}
          </summary>
          <pre className="px-3 py-2 text-[11px] font-mono text-text-faint whitespace-pre-wrap max-h-72 overflow-y-auto border-t border-border">
            {node.text}
          </pre>
        </details>
      )
    default:
      return null
  }
}
