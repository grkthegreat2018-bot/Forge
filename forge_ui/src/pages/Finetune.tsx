// Fine-Tune — dataset picker + SFT launcher with command preview.

import { clsx } from 'clsx'
import {
  Check as CheckIcon, Copy, Database, Download, Play, RefreshCw, SquareTerminal,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import type { Dataset } from '../lib/types'
import {
  Btn, Card, Check, EmptyState, IconBtn, Input, NumInput, SearchInput, Select, Spinner,
} from '../components/ui'

interface FinetuneForm {
  config: string
  checkpoint: string
  save: string
  use_lora: boolean
  lora_r: number
  lora_alpha: number
  save_adapter: boolean
  bitnet: boolean
  max_steps: number
  lr: number
  min_lr: number
  warmup: number
  batch_size: number
  grad_accum: number
  seq_len: number
  weight_decay: number
  grad_clip: number
  qk_clip_tau: number
  optimizer: string
  loss_function: string
  entropy_alpha: number
  curriculum: string
  ema: boolean
  augment: boolean
  synpro: boolean
  val_every: number
}

interface DatasetIndex {
  datasets: Dataset[]
  exports: unknown[]
  choices: { optimizers: string[]; losses: string[]; curricula: string[] }
}

const DEFAULT_FORM: FinetuneForm = {
  config: 'forgelm_v2',
  checkpoint: 'research/checkpoints/ForgeLM_V2.safetensors',
  save: 'research/checkpoints/forge_finetuned.safetensors',
  use_lora: true, lora_r: 32, lora_alpha: 64, save_adapter: true,
  bitnet: true,
  max_steps: 500, lr: 5e-5, min_lr: 5e-6, warmup: 20,
  batch_size: 1, grad_accum: 5, seq_len: 1024,
  weight_decay: 0.01, grad_clip: 1.0, qk_clip_tau: 0.0,
  optimizer: 'muon_sf', loss_function: 'ce', entropy_alpha: 0.5,
  curriculum: 'none',
  ema: false, augment: false, synpro: false, val_every: 0,
}

const fmtSize = (b: number) =>
  b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB`
  : b >= 1e3 ? `${(b / 1e3).toFixed(1)} KB`
  : `${b} B`

const fmtDate = (t: number) =>
  new Date(t * 1000).toLocaleString(undefined,
    { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

export default function Finetune() {
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [nExports, setNExports] = useState(0)
  const [choices, setChoices] = useState<DatasetIndex['choices']>(
    { optimizers: [], losses: [], curricula: [] })
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [form, setForm] = useState<FinetuneForm>(DEFAULT_FORM)
  const [datasetSearch, setDatasetSearch] = useState('')
  const [copiedCmd, setCopiedCmd] = useState(false)
  const [preview, setPreview] = useState('')
  const [previewing, setPreviewing] = useState(false)
  const [launching, setLaunching] = useState(false)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')

  const set = <K extends keyof FinetuneForm>(k: K, v: FinetuneForm[K]) =>
    setForm((f) => ({ ...f, [k]: v }))

  const load = useCallback(() =>
    api.get<DatasetIndex>('/api/finetune/datasets')
      .then((r) => {
        setDatasets(r.datasets ?? [])
        setNExports(r.exports?.length ?? 0)
        if (r.choices) setChoices(r.choices)
      })
      .catch((e) => setError(String(e))), [])

  useEffect(() => { load() }, [load])

  const toggle = (path: string) =>
    setSelected((s) => {
      const n = new Set(s)
      if (n.has(path)) n.delete(path); else n.add(path)
      return n
    })

  const filteredDatasets = useMemo(() => {
    const q = datasetSearch.trim().toLowerCase()
    if (!q) return datasets
    return datasets.filter((d) =>
      d.name.toLowerCase().includes(q) ||
      d.path.toLowerCase().includes(q)
    )
  }, [datasets, datasetSearch])

  const totalExamples = useMemo(
    () => datasets.filter((d) => selected.has(d.path))
      .reduce((a, d) => a + d.examples, 0),
    [datasets, selected])

  const body = () => ({ ...form, datasets: [...selected] })

  const doPreview = async () => {
    setPreviewing(true); setError('')
    try {
      const r = await api.post<{ cmd: string[] | null; preview: string }>(
        '/api/finetune/preview', body())
      setPreview(r.preview || (r.cmd ? r.cmd.join(' ') : ''))
      if (!r.cmd) setError('select at least one dataset')
    } catch (e) { setError(String(e)) }
    finally { setPreviewing(false) }
  }

  const doLaunch = async () => {
    if (!selected.size || launching) return
    setLaunching(true); setError(''); setNote('')
    try {
      const r = await api.post<{ ok: boolean; task_id?: string; error?: string }>(
        '/api/finetune/launch', body())
      if (r.ok) {
        setNote(`Launched${r.task_id ? ` as task ${r.task_id}` : ''} — monitor on the Tasks page.`)
      } else {
        setError(r.error ?? 'launch failed')
      }
    } catch (e) { setError(String(e)) }
    finally { setLaunching(false) }
  }

  const doExport = async () => {
    setError(''); setNote('')
    try {
      const r = await api.post<{ path: string; examples: number }>(
        '/api/finetune/export')
      setNote(`Exported ${r.examples} rated examples → ${r.path}`)
      load()
    } catch (e) { setError(String(e)) }
  }

  const opts = (list: string[], cur: string) =>
    list.includes(cur) ? list : [cur, ...list]

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="flex-1 min-h-0 flex gap-4 p-4 pb-3">
        {/* left ~55%: dataset picker */}
        <div className="w-[55%] shrink-0 bg-panel border border-border rounded-card shadow-card flex flex-col min-h-0">
          <div className="flex items-center justify-between px-4 pt-3 pb-2">
            <div className="flex items-center gap-3">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-text-faint">
                Datasets ({selected.size} / {datasets.length})
              </span>
              <button onClick={() => setSelected(new Set(datasets.map((d) => d.path)))}
                className="text-[10.5px] text-accent-hi hover:underline cursor-pointer">all</button>
              <button onClick={() => setSelected(new Set())}
                className="text-[10.5px] text-text-faint hover:underline cursor-pointer">none</button>
            </div>
            <IconBtn title="Refresh datasets" onClick={load}>
              <RefreshCw size={12} />
            </IconBtn>
          </div>
          <div className="px-3 pb-2">
            <SearchInput
              value={datasetSearch}
              onChange={setDatasetSearch}
              placeholder="Search datasets…"
              className="w-full"
            />
          </div>
          <div className="flex-1 overflow-y-auto px-2 pb-2">
            {filteredDatasets.map((d) => (
              <label
                key={d.path}
                className={clsx(
                  'flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg cursor-pointer transition-colors mb-0.5',
                  selected.has(d.path)
                    ? 'bg-accent/10 border border-accent/25'
                    : 'hover:bg-panel-alt border border-transparent')}>
                <input
                  type="checkbox" checked={selected.has(d.path)}
                  onChange={() => toggle(d.path)}
                  className="w-3.5 h-3.5 rounded accent-[#8aa3ff] cursor-pointer shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="block text-[12.5px] text-text truncate">{d.name}</span>
                  <span className="block text-[10.5px] text-text-faint font-mono truncate">
                    {d.path}
                  </span>
                </span>
                <span className="text-[11px] text-text-dim shrink-0">
                  {d.examples} ex
                </span>
                <span className="text-[11px] text-text-faint shrink-0 w-[62px] text-right">
                  {fmtSize(d.size_bytes)}
                </span>
                <span className="text-[10.5px] text-text-faint shrink-0 w-[92px] text-right hidden xl:block">
                  {fmtDate(d.modified)}
                </span>
              </label>
            ))}
            {!datasets.length && (
              <EmptyState
                icon={<Database size={34} strokeWidth={1.2} />}
                title="No datasets"
                desc="*.jsonl files under the data/ directory appear here. Export rated chat data to create one." />
            )}
          </div>
          <div className="flex items-center gap-2 px-3 py-2 border-t border-border">
            <span className="text-[11.5px] text-text-dim">
              {selected.size} selected · {totalExamples.toLocaleString()} examples
            </span>
            {nExports > 0 && (
              <span className="text-[10.5px] text-text-faint">
                ({nExports} export{nExports > 1 ? 's' : ''} on disk)
              </span>
            )}
            <div className="ml-auto">
              <Btn variant="subtle" onClick={doExport}>
                <Download size={12} /> Export rated data
              </Btn>
            </div>
          </div>
        </div>

        {/* right: parameter groups */}
        <div className="flex-1 min-w-0 overflow-y-auto space-y-3 pr-0.5">
          <Card title="Data & model">
            <div className="space-y-2">
              <Field label="Config preset">
                <Input value={form.config}
                  onChange={(e) => set('config', e.target.value)}
                  className="w-full font-mono !text-[12px]" />
              </Field>
              <Field label="Base checkpoint">
                <Input value={form.checkpoint}
                  onChange={(e) => set('checkpoint', e.target.value)}
                  className="w-full font-mono !text-[12px]" />
              </Field>
              <Field label="Save checkpoint to">
                <Input value={form.save}
                  onChange={(e) => set('save', e.target.value)}
                  className="w-full font-mono !text-[12px]" />
              </Field>
            </div>
          </Card>

          <Card title="LoRA">
            <div className="space-y-2.5">
              <Check label="Use LoRA adapters"
                checked={form.use_lora}
                onChange={(v) => set('use_lora', v)} />
              <div className={clsx('grid grid-cols-2 gap-x-4 gap-y-2',
                !form.use_lora && 'opacity-40 pointer-events-none')}>
                <NumInput label="Rank (r)" value={form.lora_r}
                  onChange={(v) => set('lora_r', v)} min={1} max={256} step={1}
                  className="justify-between w-full" />
                <NumInput label="Alpha" value={form.lora_alpha}
                  onChange={(v) => set('lora_alpha', v)} min={1} max={512} step={1}
                  className="justify-between w-full" />
              </div>
              <div className={!form.use_lora ? 'opacity-40 pointer-events-none' : ''}>
                <Check label="Save adapter separately"
                  checked={form.save_adapter}
                  onChange={(v) => set('save_adapter', v)} />
              </div>
            </div>
          </Card>

          <Card title="Schedule">
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              <NumInput label="Max steps" value={form.max_steps}
                onChange={(v) => set('max_steps', v)} min={1} max={200000} step={50}
                className="justify-between w-full" />
              <NumInput label="Warmup" value={form.warmup}
                onChange={(v) => set('warmup', v)} min={0} max={10000} step={5}
                className="justify-between w-full" />
              <NumInput label="LR" value={form.lr}
                onChange={(v) => set('lr', v)} min={0} step={0.000001}
                className="justify-between w-full" />
              <NumInput label="Min LR" value={form.min_lr}
                onChange={(v) => set('min_lr', v)} min={0} step={0.000001}
                className="justify-between w-full" />
              <NumInput label="Batch size" value={form.batch_size}
                onChange={(v) => set('batch_size', v)} min={1} max={64} step={1}
                className="justify-between w-full" />
              <NumInput label="Grad accum" value={form.grad_accum}
                onChange={(v) => set('grad_accum', v)} min={1} max={128} step={1}
                className="justify-between w-full" />
              <NumInput label="Seq len" value={form.seq_len}
                onChange={(v) => set('seq_len', v)} min={64} max={32768} step={128}
                className="justify-between w-full" />
              <NumInput label="Val every" value={form.val_every}
                onChange={(v) => set('val_every', v)} min={0} max={10000} step={10}
                className="justify-between w-full" />
            </div>
          </Card>

          <Card title="Optimizer">
            <div className="space-y-2.5">
              <SelectRow label="Optimizer" value={form.optimizer}
                onChange={(v) => set('optimizer', v)}
                options={opts(choices.optimizers, form.optimizer)} />
              <SelectRow label="Loss" value={form.loss_function}
                onChange={(v) => set('loss_function', v)}
                options={opts(choices.losses, form.loss_function)} />
              <SelectRow label="Curriculum" value={form.curriculum}
                onChange={(v) => set('curriculum', v)}
                options={opts(choices.curricula, form.curriculum)} />
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 pt-1">
                <NumInput label="Weight decay" value={form.weight_decay}
                  onChange={(v) => set('weight_decay', v)} min={0} max={1} step={0.005}
                  className="justify-between w-full" />
                <NumInput label="Grad clip" value={form.grad_clip}
                  onChange={(v) => set('grad_clip', v)} min={0} max={10} step={0.1}
                  className="justify-between w-full" />
                <NumInput label="QK clip τ" value={form.qk_clip_tau}
                  onChange={(v) => set('qk_clip_tau', v)} min={0} max={10} step={0.05}
                  className="justify-between w-full" />
                <NumInput label="Entropy α" value={form.entropy_alpha}
                  onChange={(v) => set('entropy_alpha', v)} min={0} max={5} step={0.05}
                  className="justify-between w-full" />
              </div>
            </div>
          </Card>

          <Card title="Flags">
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              <Check label="BitNet everywhere" checked={form.bitnet}
                onChange={(v) => set('bitnet', v)} />
              <Check label="EMA weights" checked={form.ema}
                onChange={(v) => set('ema', v)} />
              <Check label="Augment data" checked={form.augment}
                onChange={(v) => set('augment', v)} />
              <Check label="SynPro mixing" checked={form.synpro}
                onChange={(v) => set('synpro', v)} />
            </div>
          </Card>
        </div>
      </div>

      {/* bottom bar: preview + launch */}
      <div className="border-t border-border px-4 py-2.5 space-y-2">
        {preview && (
          <div className="relative">
            <pre className="font-mono text-[11.5px] text-text-dim bg-deep border border-border rounded-lg p-3 pr-20 whitespace-pre-wrap break-all max-h-28 overflow-y-auto">
              {preview}
            </pre>
            <button
              onClick={() => {
                void navigator.clipboard.writeText(preview).then(() => {
                  setCopiedCmd(true)
                  setTimeout(() => setCopiedCmd(false), 1200)
                })
              }}
              title="Copy command"
              className="absolute right-2.5 top-2.5 flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-panel-alt hover:bg-panel-hover border border-border text-text-dim hover:text-text cursor-pointer transition-colors">
              {copiedCmd ? <CheckIcon size={11} className="text-ok" /> : <Copy size={11} />}
              {copiedCmd ? 'Copied' : 'Copy'}
            </button>
          </div>
        )}
        <div className="flex items-center gap-2">
          <Btn variant="subtle" onClick={doPreview} disabled={previewing}>
            {previewing ? <Spinner /> : <SquareTerminal size={13} />} Preview cmd
          </Btn>
          <Btn variant="primary" onClick={doLaunch}
            disabled={!selected.size || launching}>
            {launching ? <Spinner /> : <Play size={13} />} Launch fine-tune
          </Btn>
          {note && <span className="text-[12px] text-ok truncate">{note}</span>}
          {note && (
            <Btn variant="ghost" className="!py-0.5 !px-2 text-[11.5px]"
              onClick={() => navigate('/tasks')}>
              Open Tasks →
            </Btn>
          )}
          {error && <span className="text-[12px] text-err truncate">{error}</span>}
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: {
  label: string
  children: React.ReactNode
}) {
  return (
    <label className="block text-[12px] text-text-dim space-y-1">
      <span>{label}</span>
      {children}
    </label>
  )
}

function SelectRow({ label, value, onChange, options }: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
}) {
  return (
    <div className="flex items-center justify-between text-[12px] text-text-dim">
      <span>{label}</span>
      <Select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </Select>
    </div>
  )
}
