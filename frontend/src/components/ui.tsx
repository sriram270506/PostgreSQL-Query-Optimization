import { useState, type ReactNode } from 'react'
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

export function Seg<T extends string>({ value, options, onChange }: { value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <div className="seg" role="tablist">
      {options.map(([v, l]) => (
        <button key={v} role="tab" aria-selected={v === value} className={v === value ? 'on' : ''} onClick={() => onChange(v)}>{l}</button>
      ))}
    </div>
  )
}

export function Skeleton({ h = 160 }: { h?: number }) {
  return <div className="skeleton" style={{ height: h }} aria-label="Loading" />
}

export function ErrorBox({ error }: { error: string }) {
  return <div className="error" role="alert">{error}</div>
}

/** tooltip state helper: position follows the pointer */
export function useTip() {
  const [tip, setTip] = useState<{ x: number; y: number; html: ReactNode } | null>(null)
  const show = (e: React.MouseEvent, html: ReactNode) => setTip({ x: e.clientX + 14, y: e.clientY + 14, html })
  const hide = () => setTip(null)
  const node = tip && <div className="tip" style={{ left: Math.min(tip.x, window.innerWidth - 330), top: tip.y }}>{tip.html}</div>
  return { show, hide, node }
}
