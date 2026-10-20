import { useRef, useState, type ReactNode } from 'react'
import { EST_COLOR, EST_LABEL } from '../lib'

export function SectionHead({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="section-head">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2 className="title">{title}</h2>
      </div>
      {children && <div style={{ maxWidth: 460 }} className="note">{children}</div>}
    </div>
  )
}

export function Card({ k, title, right, children, className = '' }: {
  k?: string; title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <div className={`card ${className}`}>
      {(k || title || right) && (
        <div className="card-head">
          <div>{k && <div className="k">{k}</div>}{title && <h3 className="sub" style={{ marginTop: k ? 6 : 0 }}>{title}</h3>}</div>
          {right}
        </div>
      )}
      {children}
    </div>
  )
}

export function EstName({ name, short = false }: { name: string; short?: boolean }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontWeight: 600 }}>
      <span className="dot" style={{ background: EST_COLOR[name] }} aria-hidden />
      {short ? EST_LABEL[name].split(' ')[0] : EST_LABEL[name]}
    </span>
  )
}

export function Legend({ names }: { names: string[] }) {
  return <div className="legend">{names.map(n => <span key={n}><span className="dot" style={{ background: EST_COLOR[n] }} />{EST_LABEL[n]}</span>)}</div>
}

/** Segmented control = a radio group: one tab stop, arrow keys / Home / End move the choice. */
export function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: [T, string][]; onChange: (v: T) => void; label?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const key = (e: React.KeyboardEvent) => {
    const i = options.findIndex(o => o[0] === value)
    const n = options.length
    const j = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? (i + 1) % n
      : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? (i - 1 + n) % n
      : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : -1
    if (j < 0) return
    e.preventDefault()
    onChange(options[j][0])
    ref.current?.querySelectorAll<HTMLButtonElement>('button')[j]?.focus()
  }
  return (
    <div className="seg" role="radiogroup" aria-label={label} ref={ref} onKeyDown={key}>
      {options.map(([v, l]) => (
        <button key={v} type="button" role="radio" aria-checked={v === value} tabIndex={v === value ? 0 : -1}
          className={v === value ? 'on' : ''} onClick={() => onChange(v)}>{l}</button>
      ))}
    </div>
  )
}

export function Skeleton({ h = 160 }: { h?: number }) {
  return <div className="skeleton" style={{ height: h }} aria-label="Loading" />
}

export function ErrorBox({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="error" role="alert">
      {error}
      {onRetry && <button type="button" className="btn ghost" style={{ marginLeft: 12, height: 32 }} onClick={onRetry}>Retry</button>}
    </div>
  )
}

/** tooltip state helper: position follows the pointer */
export function useTip() {
  const [tip, setTip] = useState<{ x: number; y: number; html: ReactNode } | null>(null)
  const show = (e: React.MouseEvent, html: ReactNode) => setTip({ x: e.clientX + 14, y: e.clientY + 14, html })
  const hide = () => setTip(null)
  const node = tip && <div className="tip" style={{ left: Math.min(tip.x, window.innerWidth - 330), top: tip.y }}>{tip.html}</div>
  return { show, hide, node }
}
