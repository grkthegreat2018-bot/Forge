// Lightweight SVG charts — LiveLineChart + CircularGauge equivalents.
// Custom (not recharts) keeps the bundle small and style fully ours.

import { useId, useMemo, useState } from 'react'

export interface Series {
  name: string
  color: string
  data: number[]
}

export function LineChart({
  series,
  height = 140,
  window = 240,
  minY,
  maxY,
  yLabel,
  showArea = true,
}: {
  series: Series[]
  height?: number
  window?: number
  minY?: number
  maxY?: number
  yLabel?: string
  showArea?: boolean
}) {
  const chartId = useId().replace(/:/g, '')
  const W = 560
  const H = height
  const pad = { l: 38, r: 12, t: 10, b: 20 }
  const iw = W - pad.l - pad.r
  const ih = H - pad.t - pad.b

  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const { lo, hi, paths } = useMemo(() => {
    const all = series.flatMap((s) => s.data.slice(-window))
    let lo = minY ?? Math.min(0, ...all)
    let hi = maxY ?? Math.max(1e-9, ...all)
    if (hi - lo < 1e-9) hi = lo + 1

    const paths = series.map((s, sIdx) => {
      const d = s.data.slice(-window)
      if (d.length < 2) return { s, d: '', area: '', raw: d, sIdx }
      const pts = d.map((v, i) => {
        const x = pad.l + (i / Math.max(d.length - 1, 1)) * iw
        const y = pad.t + ih - ((v - lo) / (hi - lo)) * ih
        return { x, y, v }
      })
      const pathD = `M${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L')}`
      const firstX = pts[0].x.toFixed(1)
      const lastX = pts[pts.length - 1].x.toFixed(1)
      const baselineY = (pad.t + ih).toFixed(1)
      const areaD = `${pathD} L${lastX},${baselineY} L${firstX},${baselineY} Z`

      return { s, d: pathD, area: areaD, pts, raw: d, sIdx }
    })
    return { lo, hi, paths }
  }, [series, window, minY, maxY, iw, ih])

  const maxPoints = Math.max(1, ...series.map((s) => s.data.slice(-window).length))

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * W
    if (x < pad.l || x > pad.l + iw || maxPoints <= 1) {
      setHoverIdx(null)
      return
    }
    const ratio = (x - pad.l) / iw
    const idx = Math.round(ratio * (maxPoints - 1))
    setHoverIdx(Math.max(0, Math.min(maxPoints - 1, idx)))
  }

  return (
    <div className="w-full select-none">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full overflow-visible"
        style={{ height }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}>
        <defs>
          {paths.map(({ s, sIdx }) => (
            <linearGradient key={`grad-${s.name}`} id={`areaGrad-${chartId}-${sIdx}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity="0.3" />
              <stop offset="100%" stopColor={s.color} stopOpacity="0.0" />
            </linearGradient>
          ))}
          <filter id={`glow-${chartId}`} x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* grid */}
        {[0, 0.5, 1].map((f) => {
          const y = pad.t + ih * f
          const v = hi - f * (hi - lo)
          return (
            <g key={f}>
              <line
                x1={pad.l}
                x2={W - pad.r}
                y1={y}
                y2={y}
                stroke="#1c2436"
                strokeWidth="1"
                strokeDasharray={f === 0.5 ? '3 3' : undefined}
              />
              <text
                x={pad.l - 6}
                y={y + 3}
                textAnchor="end"
                fontSize="9"
                fill="#5b677d"
                fontFamily="var(--font-mono)">
                {v >= 100 ? v.toFixed(0) : v.toPrecision(2)}
              </text>
            </g>
          )
        })}

        {/* area fills */}
        {showArea &&
          paths.map(({ s, area, sIdx }) =>
            area ? (
              <path
                key={`area-${s.name}`}
                d={area}
                fill={`url(#areaGrad-${chartId}-${sIdx})`}
              />
            ) : null
          )}

        {/* line paths */}
        {paths.map(({ s, d }) =>
          d ? (
            <path
              key={`line-${s.name}`}
              d={d}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              filter={`url(#glow-${chartId})`}
            />
          ) : null
        )}

        {/* latest dots */}
        {paths.map(({ s, d, pts }) => {
          if (!d || !pts || !pts.length) return null
          const last = pts[pts.length - 1]
          return (
            <g key={`last-${s.name}`}>
              <circle cx={last.x} cy={last.y} r="4" fill={s.color} opacity="0.4" className="pulse-ring" />
              <circle cx={last.x} cy={last.y} r="2.5" fill="#ffffff" stroke={s.color} strokeWidth="1.5" />
            </g>
          )
        })}

        {/* hover crosshair & values */}
        {hoverIdx !== null && (
          <g>
            <line
              x1={pad.l + (hoverIdx / Math.max(maxPoints - 1, 1)) * iw}
              x2={pad.l + (hoverIdx / Math.max(maxPoints - 1, 1)) * iw}
              y1={pad.t}
              y2={pad.t + ih}
              stroke="#42557d"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
            {paths.map(({ s, pts }) => {
              if (!pts || !pts[hoverIdx]) return null
              const p = pts[hoverIdx]
              return (
                <g key={`hover-pt-${s.name}`}>
                  <circle cx={p.x} cy={p.y} r="4" fill={s.color} stroke="#ffffff" strokeWidth="1.5" />
                  <rect
                    x={p.x + 6}
                    y={Math.max(pad.t, p.y - 12)}
                    width="44"
                    height="18"
                    rx="4"
                    fill="#111622"
                    stroke="#2c3852"
                    strokeWidth="1"
                  />
                  <text
                    x={p.x + 28}
                    y={Math.max(pad.t, p.y - 12) + 12}
                    textAnchor="middle"
                    fontSize="9.5"
                    fill="#eef3fb"
                    fontFamily="var(--font-mono)">
                    {p.v >= 100 ? p.v.toFixed(0) : p.v.toPrecision(3)}
                  </text>
                </g>
              )
            })}
          </g>
        )}
      </svg>
      <div className="flex items-center justify-between gap-3 px-2 pt-1 border-t border-border/30">
        <div className="flex items-center gap-3">
          {yLabel && <span className="text-[10px] font-semibold text-text-faint uppercase tracking-wider">{yLabel}</span>}
          {series.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5 text-[11px] text-text-dim">
              <span className="w-2 h-2 rounded-full shadow-[0_0_6px]" style={{ background: s.color, boxShadow: `0 0 6px ${s.color}` }} />
              <span className="font-medium">{s.name}</span>
            </span>
          ))}
        </div>
        {hoverIdx !== null && (
          <span className="text-[10px] text-text-faint font-mono">
            step #{hoverIdx + 1}
          </span>
        )}
      </div>
    </div>
  )
}

export function Gauge({ value, caption, size = 120, dangerAt = 85 }: {
  value: number          // 0..100
  caption: string
  size?: number
  dangerAt?: number
}) {
  const chartId = useId().replace(/:/g, '')
  const r = 42
  const cx = 60
  const cy = 60
  const start = 135
  const span = 270
  const frac = Math.max(0, Math.min(100, value)) / 100
  const end = start + span * frac

  const pt = (deg: number, rad: number) => {
    const a = ((deg - 90) * Math.PI) / 180
    return [cx + rad * Math.cos(a), cy + rad * Math.sin(a)]
  }
  const arc = (from: number, to: number, rad: number) => {
    const [x0, y0] = pt(from, rad)
    const [x1, y1] = pt(to, rad)
    const large = to - from > 180 ? 1 : 0
    return `M${x0.toFixed(1)},${y0.toFixed(1)} A${rad},${rad} 0 ${large} 1 ${x1.toFixed(1)},${y1.toFixed(1)}`
  }

  const isDanger = value >= dangerAt
  const color = isDanger ? '#f87171' : `url(#gaugeGrad-${chartId})`

  return (
    <div className="flex flex-col items-center group" style={{ width: size }}>
      <svg viewBox="0 0 120 120" width={size} height={size} className="overflow-visible">
        <defs>
          <linearGradient id={`gaugeGrad-${chartId}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#6366f1" />
            <stop offset="50%" stopColor="#7a97ff" />
            <stop offset="100%" stopColor="#2dd4bf" />
          </linearGradient>
          <filter id={`gaugeGlow-${chartId}`} x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        {/* outer track */}
        <path
          d={arc(start, start + span, r)}
          fill="none"
          stroke="#171d2b"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* inner faint accent line */}
        <path
          d={arc(start, start + span, r - 7)}
          fill="none"
          stroke="#121722"
          strokeWidth="1.5"
          strokeDasharray="2 3"
        />
        {/* live arc */}
        {frac > 0.005 && (
          <path
            d={arc(start, Math.min(end, start + span), r)}
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            filter={`url(#gaugeGlow-${chartId})`}
            className="transition-all duration-300"
          />
        )}
        <text
          x={cx}
          y={cy + 3}
          textAnchor="middle"
          fontSize="20"
          fontWeight="700"
          fill="#eef3fb"
          fontFamily="var(--font-mono)">
          {Math.round(value)}
        </text>
        <text
          x={cx}
          y={cy + 17}
          textAnchor="middle"
          fontSize="9"
          fontWeight="600"
          fill={isDanger ? '#f87171' : '#7a97ff'}
          fontFamily="var(--font-mono)">
          %
        </text>
      </svg>
      <div className="text-[10.5px] font-semibold uppercase tracking-widest text-text-faint -mt-1 group-hover:text-text-dim transition-colors">
        {caption}
      </div>
    </div>
  )
}

