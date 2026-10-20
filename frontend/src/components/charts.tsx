import { useMemo } from 'react'
import { EST_COLOR, EST_LABEL, fmtCompact, fmtQ } from '../lib'
import { useTip } from './ui'

const log10 = (x: number) => Math.log10(Math.max(x, 1))

/** Estimate vs truth on log-log axes, with the perfect-estimate diagonal and a 2x band. */
export function LogScatter({ truth, est, name, size = 260, maxPoints = 1200 }: {
  truth: number[]; est: number[]; name: string; size?: number; maxPoints?: number
}) {
  const { show, hide, node } = useTip()
  const pad = 34
  const pts = useMemo(() => {
    const step = Math.max(1, Math.floor(truth.length / maxPoints))
    return truth.map((t, i) => [t, est[i]] as const).filter((_, i) => i % step === 0)
  }, [truth, est, maxPoints])
  const hi = Math.ceil(Math.max(...truth.map(log10), ...est.map(log10), 1))
  const s = (v: number) => pad + (log10(v) / hi) * (size - pad - 8)
  const y = (v: number) => size - s(v)
  const band = Math.log10(2) / hi * (size - pad - 8)
  const ticks = Array.from({ length: hi + 1 }, (_, i) => i).filter(i => hi <= 6 || i % 2 === 0)
  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${size} ${size}`} className="chart" role="img" aria-label={`${EST_LABEL[name]}: estimate versus true cardinality`} style={{ width: '100%', height: 'auto' }}>
        <g className="grid">{ticks.map(i => <line key={i} x1={s(10 ** i)} x2={s(10 ** i)} y1={8} y2={size - pad} />)}
          {ticks.map(i => <line key={'h' + i} y1={y(10 ** i)} y2={y(10 ** i)} x1={pad} x2={size - 8} />)}</g>
        <polygon points={`${pad},${size - pad - band} ${pad + band},${size - pad} ${size - 8},${8 + band} ${size - 8 - band},8`} fill="#8a153810" />
        <line x1={pad} y1={size - pad} x2={size - 8} y2={8} stroke="#8a153866" strokeDasharray="4 4" />
        {pts.map(([t, e], i) => (
          <circle key={i} cx={s(t)} cy={y(e)} r={2.4} fill={EST_COLOR[name]} fillOpacity={0.45}
            onMouseMove={ev => show(ev, <>true {fmtCompact(t)} · est {fmtCompact(e)}<br />q-error {fmtQ(Math.max(e, 1) / Math.max(t, 1) > 1 ? Math.max(e, 1) / Math.max(t, 1) : Math.max(t, 1) / Math.max(e, 1))}</>)}
            onMouseLeave={hide} />
        ))}
        {ticks.map(i => <text key={'x' + i} x={s(10 ** i)} y={size - pad + 15} textAnchor="middle">10{sup(i)}</text>)}
        {ticks.map(i => <text key={'y' + i} x={pad - 6} y={y(10 ** i) + 4} textAnchor="end">10{sup(i)}</text>)}
        <text x={size - 8} y={size - 6} textAnchor="end">true rows →</text>
        <text x={4} y={14}>estimate</text>
      </svg>
      {node}
    </div>
  )
}
const sup = (i: number) => String(i).split('').map(c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('')

/** Grouped bars on a log axis: groups (e.g. percentiles) x series (estimators). */
export function GroupedLogBars({ groups, series, value, height = 260, fmt = fmtQ }: {
  groups: string[]; series: string[]; value: (g: string, s: string) => number; height?: number; fmt?: (v: number) => string
}) {
  const { show, hide, node } = useTip()
  const vals = groups.flatMap(g => series.map(s => value(g, s)))
  const W = 640, gw0 = (W - 52) / groups.length
  // every bar carries its value: horizontal when there is room, rotated otherwise
  const upright = gw0 > 90 && series.length <= 4
  const pad = { l: 44, r: 8, t: upright ? 22 : 46, b: 30 }
  const top = Math.max(1, Math.ceil(Math.log10(Math.max(...vals, 2))))
  const gw = (W - pad.l - pad.r) / groups.length
  const bw = Math.min(26, (gw - 18) / series.length - 2)
  const yy = (v: number) => pad.t + (1 - Math.log10(Math.max(v, 1)) / top) * (height - pad.t - pad.b)
  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${height}`} className="chart" style={{ width: '100%', height: 'auto' }} role="img" aria-label="Q-error by percentile, log scale">
        <g className="grid">{Array.from({ length: top + 1 }, (_, i) => <line key={i} x1={pad.l} x2={W - pad.r} y1={yy(10 ** i)} y2={yy(10 ** i)} />)}</g>
        {Array.from({ length: top + 1 }, (_, i) => <text key={i} x={pad.l - 6} y={yy(10 ** i) + 4} textAnchor="end">{fmtCompact(10 ** i)}×</text>)}
        {groups.map((g, gi) => {
          const x0 = pad.l + gi * gw + (gw - series.length * (bw + 2)) / 2
          return (
            <g key={g}>
              {series.map((s, si) => {
                const v = value(g, s)
                const x = x0 + si * (bw + 2)
                const yv = yy(v)
                const h = Math.max(height - pad.b - yv, 1.5)
                return (
                  <g key={s} onMouseMove={e => show(e, <><b>{EST_LABEL[s]}</b><br />{g}: {fmt(v)}</>)} onMouseLeave={hide}>
                    <rect x={x - 1} y={pad.t} width={bw + 2} height={height - pad.t - pad.b} fill="transparent" />
                    <path d={roundTop(x, height - pad.b - h, bw, h)} fill={EST_COLOR[s]} />
                    {upright
                      ? <text x={x + bw / 2} y={height - pad.b - h - 4} textAnchor="middle" style={{ fontSize: 9.5, fill: 'var(--ink-2)' }}>{fmt(v).replace('×', '')}</text>
                      : <text transform={`translate(${x + bw / 2 + 3.5},${height - pad.b - h - 4}) rotate(-90)`} style={{ fontSize: 10, fill: 'var(--ink-2)' }}>{fmt(v).replace('×', '')}</text>}
                  </g>
                )
              })}
              <text x={pad.l + gi * gw + gw / 2} y={height - 9} textAnchor="middle" style={{ fontWeight: 600, fill: 'var(--ink-2)' }}>{g}</text>
            </g>
          )
        })}
        <line className="axis" x1={pad.l} x2={W - pad.r} y1={height - pad.b} y2={height - pad.b} />
      </svg>
      {node}
      <details className="chart-table">
        <summary>Show as table</summary>
        <div className="scroll-x">
          <table className="tbl">
            <thead><tr><th scope="col">Estimator</th>{groups.map(g => <th key={g} scope="col" className="r num">{g}</th>)}</tr></thead>
            <tbody>{series.map(s => <tr key={s}><th scope="row">{EST_LABEL[s]}</th>{groups.map(g => <td key={g} className="r num">{fmt(value(g, s))}</td>)}</tr>)}</tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
function roundTop(x: number, y: number, w: number, h: number) {
  const r = Math.min(4, w / 2, h)
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`
}

/** Labelled horizontal bars; value text always shown (keeps the amber series readable). */
export function HBars({ rows, max, fmt, labelWidth = 150 }: {
  rows: { label: React.ReactNode; value: number; color: string; key: string }[]; max?: number; fmt: (v: number) => string; labelWidth?: number
}) {
  const m = max ?? Math.max(...rows.map(r => r.value), 1e-9)
  return (
    <div>
      {rows.map(r => (
        <div key={r.key} className="bar-row" style={{ gridTemplateColumns: `${labelWidth}px 1fr 76px` }}>
          <div style={{ minWidth: 0 }}>{r.label}</div>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(1.5, (r.value / m) * 100)}%`, background: r.color }} /></div>
          <div className="num" style={{ textAlign: 'right', fontWeight: 600 }}>{fmt(r.value)}</div>
        </div>
      ))}
    </div>
  )
}

/** Matrix heatmap with row/col labels. */
export function Heatmap({ data, rows, cols, color, cell = 22, fmt, title }: {
  data: number[][]; rows: string[]; cols: string[]; color: (v: number) => string; cell?: number; fmt?: (v: number) => string; title: string
}) {
  const { show, hide, node } = useTip()
  const lw = Math.min(150, Math.max(...rows.map(r => r.length)) * 6.4 + 10)
  const ch = 74
  const W = lw + cols.length * cell, H = ch + rows.length * cell
  return (
    <div className="scroll-x" style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" style={{ width: '100%', minWidth: Math.min(W, 520), height: 'auto' }} role="img" aria-label={title}>
        {cols.map((c, j) => <text key={c + j} transform={`translate(${lw + j * cell + cell / 2 + 3},${ch - 6}) rotate(-50)`} style={{ fontSize: 10 }}>{c}</text>)}
        {rows.map((r, i) => (
          <g key={r + i}>
            <text x={lw - 6} y={ch + i * cell + cell / 2 + 4} textAnchor="end" style={{ fontSize: 10.5 }}>{r}</text>
            {cols.map((c, j) => (
              <rect key={j} x={lw + j * cell + 1} y={ch + i * cell + 1} width={cell - 2} height={cell - 2} rx={3} fill={color(data[i][j])}
                onMouseMove={e => show(e, <>{r} · {c}<br /><b>{fmt ? fmt(data[i][j]) : data[i][j].toFixed(3)}</b></>)} onMouseLeave={hide} />
            ))}
          </g>
        ))}
      </svg>
      {node}
    </div>
  )
}

/** 2-D point map (PCA) with categorical or sequential coloring. */
export function PointMap({ xy, color, label, size = 300, r = 2.6 }: {
  xy: number[][]; color: (i: number) => string; label?: (i: number) => React.ReactNode; size?: number; r?: number
}) {
  const { show, hide, node } = useTip()
  const xs = xy.map(p => p[0]), ys = xy.map(p => p[1])
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const pad = 10
  const sx = (v: number) => pad + ((v - x0) / (x1 - x0 || 1)) * (size - 2 * pad)
  const sy = (v: number) => size - pad - ((v - y0) / (y1 - y0 || 1)) * (size - 2 * pad)
  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', height: 'auto', background: '#fcfaf9', borderRadius: 10 }} role="img" aria-label="Two-dimensional projection of hidden states">
        {xy.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={r} fill={color(i)} fillOpacity={0.7} stroke="#fff" strokeWidth={0.5}
          onMouseMove={label ? e => show(e, label(i)) : undefined} onMouseLeave={hide} />)}
      </svg>
      {node}
    </div>
  )
}

export type PlanNode = { rels: string[]; est?: number; true?: number; left?: PlanNode; right?: PlanNode }

/** Join tree drawn bottom-up: leaves are relations, inner nodes show est vs true rows. */
export function PlanTree({ plan, color }: { plan: PlanNode; color: string }) {
  type P = { n: PlanNode; x: number; y: number; kids: P[] }
  let leaf = 0
  const depth = (n: PlanNode): number => (n.left ? 1 + Math.max(depth(n.left), depth(n.right!)) : 0)
  const D = depth(plan)
  const lay = (n: PlanNode, d: number): P => {
    if (!n.left) return { n, x: leaf++, y: d, kids: [] }
    const l = lay(n.left, d + 1), r = lay(n.right!, d + 1)
    return { n, x: (l.x + r.x) / 2, y: d, kids: [l, r] }
  }
  const root = lay(plan, 0)
  const W = Math.max(leaf, 2) * 92, H = (D + 1) * 62 + 10
  const px = (p: P) => 46 + p.x * 92, py = (p: P) => 24 + p.y * 62
  const all: P[] = []
  const walk = (p: P) => { all.push(p); p.kids.forEach(walk) }
  walk(root)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W * 1.1, height: 'auto' }} className="chart" role="img" aria-label="Join plan tree">
      {all.flatMap(p => p.kids.map((k, i) => <line key={`${p.n.rels.join()}-${i}`} x1={px(p)} y1={py(p)} x2={px(k)} y2={py(k)} stroke="#d9cfcf" />))}
      {all.map(p => p.kids.length ? (
        <g key={p.n.rels.join()}>
          <rect x={px(p) - 42} y={py(p) - 15} width={84} height={30} rx={8} fill="#fff" stroke={color} />
          <text x={px(p)} y={py(p) - 2} textAnchor="middle" style={{ fontSize: 10, fill: 'var(--ink)', fontWeight: 600 }}>⋈ {fmtCompact(p.n.true ?? 0)}</text>
          <text x={px(p)} y={py(p) + 10} textAnchor="middle" style={{ fontSize: 9 }}>est {fmtCompact(p.n.est ?? 0)}</text>
        </g>
      ) : (
        <g key={p.n.rels.join()}>
          <rect x={px(p) - 20} y={py(p) - 12} width={40} height={24} rx={12} fill="var(--maroon-tint)" />
          <text x={px(p)} y={py(p) + 4} textAnchor="middle" style={{ fontSize: 11, fill: 'var(--maroon)', fontWeight: 700 }}>{p.n.rels[0]}</text>
        </g>
      ))}
    </svg>
  )
}
