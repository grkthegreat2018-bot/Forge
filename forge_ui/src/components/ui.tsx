// Shared UI primitives — the ForgeAI design system.
// Every page builds from these so restyling means editing one file.

import { clsx } from 'clsx'
import { Loader2, Search, X } from 'lucide-react'
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'

export function Card({ children, className, title, subtitle, actions, glow = 'none', variant = 'default' }: {
  children: ReactNode
  className?: string
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  glow?: 'accent' | 'warn' | 'err' | 'none'
  variant?: 'default' | 'glass' | 'subtle'
}) {
  const glowStyle = {
    accent: 'shadow-[0_0_24px_-4px_rgba(122,151,255,0.18)] border-accent/40',
    warn: 'shadow-[0_0_24px_-4px_rgba(251,191,36,0.18)] border-warn/40',
    err: 'shadow-[0_0_24px_-4px_rgba(248,113,113,0.18)] border-err/40',
    none: '',
  }[glow]

  const variantStyle = {
    default: 'bg-panel border-border',
    glass: 'glass-panel',
    subtle: 'bg-panel-alt/70 border-border/80',
  }[variant]

  return (
    <div className={clsx(
      'border rounded-card shadow-card transition-all duration-200',
      variantStyle,
      glowStyle,
      className
    )}>
      {(title || subtitle || actions) && (
        <div className="flex items-center justify-between px-4 pt-3.5 pb-1.5 border-b border-border/40">
          <div className="min-w-0">
            {title && (
              <div className="text-[11px] font-semibold uppercase tracking-wider text-text-faint flex items-center gap-2">
                {title}
              </div>
            )}
            {subtitle && (
              <div className="text-[11.5px] text-text-dim mt-0.5 truncate">{subtitle}</div>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0">{actions}</div>
        </div>
      )}
      <div className="px-4 py-3">{children}</div>
    </div>
  )
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-widest text-text-faint px-1 pt-4 pb-1.5 flex items-center gap-2">
      {children}
    </div>
  )
}

type BtnVariant = 'primary' | 'ghost' | 'danger' | 'subtle' | 'outline' | 'glass'
type BtnSize = 'xs' | 'sm' | 'md' | 'lg'

export function Btn({
  variant = 'ghost',
  size = 'md',
  loading = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: BtnVariant
  size?: BtnSize
  loading?: boolean
}) {
  const styles: Record<BtnVariant, string> = {
    primary:
      'bg-gradient-to-r from-grad-a to-grad-b text-white font-medium shadow-[0_2px_12px_-2px_rgba(99,102,241,0.5)] hover:shadow-[0_2px_16px_0_rgba(139,92,246,0.6)] hover:brightness-110 active:scale-[0.98]',
    ghost:
      'text-text-dim hover:text-text hover:bg-panel-alt/90 border border-transparent',
    danger:
      'text-err hover:bg-err/12 border border-err/30 hover:border-err/60 active:scale-[0.98]',
    subtle:
      'bg-panel-alt text-text-dim hover:text-text border border-border hover:border-border-hi active:scale-[0.98]',
    outline:
      'border border-border-hi text-text hover:border-accent hover:text-accent-hi hover:bg-accent/5 active:scale-[0.98]',
    glass:
      'glass-panel text-text hover:text-white hover:border-accent/40 active:scale-[0.98]',
  }

  const sizes: Record<BtnSize, string> = {
    xs: 'px-2 py-1 text-[11px] rounded-md gap-1',
    sm: 'px-2.5 py-1 text-[12px] rounded-lg gap-1.5',
    md: 'px-3 py-1.5 text-[13px] rounded-lg gap-1.5',
    lg: 'px-4 py-2 text-[14px] rounded-xl gap-2 font-medium',
  }

  return (
    <button
      disabled={disabled || loading}
      className={clsx(
        'inline-flex items-center justify-center transition-all duration-150 cursor-pointer select-none disabled:opacity-40 disabled:cursor-not-allowed',
        styles[variant],
        sizes[size],
        className
      )}
      {...rest}>
      {loading && <Loader2 size={13} className="animate-spin text-current shrink-0" />}
      {children}
    </button>
  )
}

export function IconBtn({ className, children, title, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      title={title}
      className={clsx(
        'inline-flex items-center justify-center w-7 h-7 rounded-md text-text-dim hover:text-text hover:bg-panel-alt border border-transparent hover:border-border transition-all duration-150 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed',
        className)}
      {...rest}>
      {children}
    </button>
  )
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={clsx(
        'bg-input border border-border rounded-lg px-3 py-1.5 text-[13px] text-text placeholder:text-text-faint',
        'focus:outline-none focus:border-accent/70 focus:ring-1 focus:ring-accent/30 transition-all',
        className)}
      {...rest} />
  )
}

export function SearchInput({
  value,
  onChange,
  onClear,
  placeholder = 'Search…',
  className,
}: {
  value: string
  onChange: (v: string) => void
  onClear?: () => void
  placeholder?: string
  className?: string
}) {
  return (
    <div className={clsx('relative flex items-center', className)}>
      <Search size={13} className="absolute left-2.5 text-text-faint pointer-events-none" />
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-input border border-border rounded-lg pl-7 pr-7 py-1.5 text-[12.5px] text-text placeholder:text-text-faint focus:outline-none focus:border-accent/70 focus:ring-1 focus:ring-accent/30 transition-all" />
      {value && (
        <button
          onClick={() => {
            onChange('')
            onClear?.()
          }}
          className="absolute right-2 text-text-faint hover:text-text cursor-pointer p-0.5">
          <X size={12} />
        </button>
      )}
    </div>
  )
}

export function Textarea({ className, ref, ...rest }:
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
  & { ref?: React.Ref<HTMLTextAreaElement> }) {
  return (
    <textarea
      ref={ref}
      className={clsx(
        'bg-input border border-border rounded-lg px-3 py-2 text-[13px] text-text placeholder:text-text-faint',
        'focus:outline-none focus:border-accent/70 focus:ring-1 focus:ring-accent/30 transition-all resize-y',
        className)}
      {...rest} />
  )
}

export function Select({ className, children, ...rest }:
  SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={clsx(
        'bg-input border border-border rounded-lg px-2.5 py-1.5 text-[13px] text-text',
        'focus:outline-none focus:border-accent/70 focus:ring-1 focus:ring-accent/30 cursor-pointer transition-all',
        className)}
      {...rest}>
      {children}
    </select>
  )
}

/** Labeled slider + numeric readout — for sampling params. */
export function Slider({ label, value, onChange, min, max, step, fmt, hint }: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  fmt?: (v: number) => string
  hint?: string
}) {
  return (
    <div className="select-none" title={hint}>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-[11.5px] text-text-dim">{label}</span>
        <span className="text-[11.5px] font-mono text-text tabular-nums bg-panel-alt/80 px-1.5 py-0.2 rounded border border-border/50">
          {fmt ? fmt(value) : value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full bg-panel-alt appearance-none cursor-pointer accent-[#7a97ff]" />
    </div>
  )
}

/** Vertical label + control wrapper for settings panels. */
export function Field({ label, hint, children }: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div>
      <div className="text-[11.5px] font-medium text-text-dim mb-1" title={hint}>{label}</div>
      {children}
    </div>
  )
}

/** Toggle chip — composer quick toggles (Tools / Thinking / …). */
export function Chip({ on, onClick, title, children }: {
  on: boolean
  onClick: () => void
  title?: string
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={clsx(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11.5px] font-medium border transition-all duration-150 cursor-pointer select-none',
        on
          ? 'bg-accent/20 text-accent-hi border-accent/40 shadow-[0_0_12px_rgba(122,151,255,0.2)]'
          : 'text-text-faint border-transparent hover:text-text-dim hover:bg-panel-alt/80 hover:border-border/60')}>
      {children}
    </button>
  )
}

export function NumInput({ label, value, onChange, min, max, step, className }: {
  label?: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  className?: string
}) {
  return (
    <label className={clsx('flex items-center gap-2 text-[12px] text-text-dim', className)}>
      {label && <span className="whitespace-nowrap">{label}</span>}
      <input
        type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="bg-input border border-border rounded-lg px-2.5 py-1 w-20 text-[12.5px] text-text focus:outline-none focus:border-accent/70 focus:ring-1 focus:ring-accent/30 font-mono" />
    </label>
  )
}

export function Check({ label, checked, onChange, className, title }: {
  label: ReactNode
  checked: boolean
  onChange: (v: boolean) => void
  className?: string
  title?: string
}) {
  return (
    <label title={title}
      className={clsx('flex items-center gap-2 text-[12.5px] text-text-dim cursor-pointer select-none hover:text-text transition-colors', className)}>
      <input
        type="checkbox" checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-4 h-4 rounded accent-[#7a97ff] cursor-pointer" />
      {label}
    </label>
  )
}

type TagKind = 'ok' | 'warn' | 'err' | 'idle' | 'accent' | 'think' | 'cyan'
export function Tag({ kind = 'idle', pulse, children, className }: {
  kind?: TagKind
  pulse?: boolean
  children: ReactNode
  className?: string
}) {
  const styles: Record<TagKind, string> = {
    ok: 'bg-ok/15 text-ok border-ok/30 shadow-[0_0_10px_rgba(52,211,153,0.15)]',
    warn: 'bg-warn/15 text-warn border-warn/30 shadow-[0_0_10px_rgba(251,191,36,0.15)]',
    err: 'bg-err/15 text-err border-err/30 shadow-[0_0_10px_rgba(248,113,113,0.15)]',
    idle: 'bg-panel-alt text-text-dim border-border',
    accent: 'bg-accent/15 text-accent-hi border-accent/30 shadow-[0_0_10px_rgba(122,151,255,0.15)]',
    think: 'bg-think/15 text-think border-think/30 shadow-[0_0_10px_rgba(192,132,252,0.15)]',
    cyan: 'bg-accent2/15 text-accent2 border-accent2/30 shadow-[0_0_10px_rgba(45,212,191,0.15)]',
  }
  return (
    <span className={clsx(
      'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[11px] font-medium transition-colors',
      styles[kind],
      pulse && 'pulse-dot',
      className)}>
      {children}
    </span>
  )
}

export function StatusDot({ kind = 'idle', pulse }: {
  kind?: TagKind
  pulse?: boolean
}) {
  const colors: Record<TagKind, string> = {
    ok: 'bg-ok', warn: 'bg-warn', err: 'bg-err',
    idle: 'bg-text-faint', accent: 'bg-accent', think: 'bg-think', cyan: 'bg-accent2',
  }
  return (
    <span className="relative inline-flex items-center justify-center w-2.5 h-2.5">
      {pulse && (
        <span className={clsx('absolute inline-flex w-full h-full rounded-full opacity-75 pulse-ring', colors[kind])} />
      )}
      <span className={clsx('inline-block w-2 h-2 rounded-full', colors[kind])} />
    </span>
  )
}

export function Progress({ value, className, danger }: {
  value: number
  className?: string
  danger?: boolean
}) {
  const pct = Math.max(0, Math.min(100, value))
  return (
    <div className={clsx('h-1.5 rounded-full bg-panel-alt overflow-hidden', className)}>
      <div
        className={clsx('h-full rounded-full transition-all duration-300',
          danger || pct > 85 ? 'bg-err' : 'bg-gradient-to-r from-grad-a to-grad-b')}
        style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 size={14} className={clsx('animate-spin text-accent', className)} />
}

export function EmptyState({ icon, title, desc, action }: {
  icon?: ReactNode
  title: string
  desc?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center py-14 text-center fade-in">
      {icon && (
        <div className="w-14 h-14 rounded-2xl bg-panel-alt border border-border/80 flex items-center justify-center text-accent/80 mb-3.5 shadow-card">
          {icon}
        </div>
      )}
      <div className="text-[15px] font-semibold text-text">{title}</div>
      {desc && <div className="text-[12.5px] text-text-dim mt-1 max-w-sm leading-relaxed">{desc}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function KvRow({ k, v, mono }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1 border-b border-border/20 last:border-none">
      <span className="text-[12px] text-text-faint shrink-0">{k}</span>
      <span className={clsx('text-[12.5px] text-text text-right truncate',
        mono && 'font-mono text-[11.5px] text-accent-hi')}>{v}</span>
    </div>
  )
}

export function MetricCard({ title, value, unit, spark, color = '#7a97ff', trend }: {
  title: string
  value: ReactNode
  unit?: string
  spark?: number[]
  color?: string
  trend?: string
}) {
  return (
    <div className="bg-panel border border-border rounded-card shadow-card px-4 py-3 min-w-0 transition-all hover:border-border-hi">
      <div className="flex items-center justify-between">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-text-faint">{title}</div>
        {trend && (
          <span className="text-[10.5px] font-medium text-accent bg-accent/10 px-1.5 py-0.2 rounded border border-accent/20">
            {trend}
          </span>
        )}
      </div>
      <div className="flex items-end justify-between gap-2 mt-1.5">
        <div className="text-[23px] font-bold leading-none truncate tracking-tight text-text">
          {value}
          {unit && <span className="text-[12px] font-normal text-text-dim ml-1">{unit}</span>}
        </div>
        {spark && spark.length > 1 && (
          <Sparkline data={spark} color={color} className="shrink-0 mb-0.5" />
        )}
      </div>
    </div>
  )
}

export function Sparkline({ data, width = 72, height = 20, color = '#7a97ff', className }: {
  data: number[]
  width?: number
  height?: number
  color?: string
  className?: string
}) {
  if (data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const pts = data.map((v, i) =>
    `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * (height - 2) - 1}`)
    .join(' ')
  return (
    <svg width={width} height={height} className={className}>
      <polyline points={pts} fill="none" stroke={color}
        strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

export function Modal({
  open,
  onClose,
  title,
  children,
  maxWidth = 'max-w-xl',
}: {
  open: boolean
  onClose: () => void
  title?: ReactNode
  children: ReactNode
  maxWidth?: string
}) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm fade-in">
      <div
        className={clsx(
          'w-full glass-dropdown rounded-2xl border border-border-hi shadow-2xl overflow-hidden flex flex-col max-h-[88vh]',
          maxWidth
        )}>
        {title && (
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-border/60">
            <div className="text-[14px] font-semibold text-text">{title}</div>
            <button
              onClick={onClose}
              className="text-text-faint hover:text-text cursor-pointer transition-colors p-1 rounded-md hover:bg-panel-alt">
              <X size={16} />
            </button>
          </div>
        )}
        <div className="p-5 overflow-y-auto flex-1 min-h-0">{children}</div>
      </div>
    </div>
  )
}

