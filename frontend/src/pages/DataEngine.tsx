import { useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { api, fmtInt, getDataset, seq } from '../lib'
import { Card, ErrorBox, SectionHead, Skeleton } from '../components/ui'
import { Heatmap } from '../components/charts'

type Schema = { tables: { name: string; rows: number; columns: { name: string; type: string; distinct: number; min?: number; max?: number; values?: string[] }[] }[]; joins: { left: string; right: string }[] }
type BN = { table: string; rows: number; nodes: { id: string; bins: string[]; marginal: number[] }[]; edges: { parent: string; child: string; mi: number; cpt: number[][] }[]; pruned: { a: string; b: string; mi: number; mi_corrected: number }[] }
type Rows = { columns: string[]; rows: unknown[][]; total: number; page: number; page_size: number }

export default function DataEngine() {
  const [s, setS] = useState<Schema | null>(null)
  const [err, setErr] = useState('')
  const [table, setTable] = useState('')
  const stats = getDataset() === 'stats'
  useEffect(() => { api<Schema>('/schema').then(r => { setS(r); setTable(r.tables[stats ? 1 : 1]?.name ?? r.tables[0].name) }).catch(e => setErr(e.message)) }, [stats])
  return (
    <>
      <section style={{ paddingTop: 48 }}>
        <div className="eyebrow">Data & engine</div>
        {stats ? <>
          <h1 className="display" style={{ fontSize: 'clamp(36px,4.4vw,54px)' }}>Real data: <em>Stack Exchange</em></h1>
          <p className="lede">The STATS benchmark (Han et al., VLDB 2022): 8 tables and about 1 million rows from stats.stackexchange.com. It has skewed counts, NULLs, timestamps and many-to-many joins on user and post ids. Our engine's exact counts match all 146 official STATS-CEB ground-truth cardinalities.</p>
        </> : <>
          <h1 className="display" style={{ fontSize: 'clamp(36px,4.4vw,54px)' }}>Six tables, <em>planted correlations</em></h1>
          <p className="lede">A synthetic HR schema with dependencies built in. Engineering sits mostly in Bangalore, Bangalore projects get bigger budgets, and age, experience and salary rise together. These are exactly the cases where the independence assumption fails.</p>
        </>}
      </section>
      {err && <section className="section"><ErrorBox error={err} /></section>}
      {!s && !err && <section className="section"><Skeleton h={200} /></section>}
      {s && <>
        <section className="section">
          <SectionHead eyebrow="Schema" title="Tables & join keys">{s.joins.length} join edges: {s.joins.map(j => `${j.left} = ${j.right}`).join(' · ')}</SectionHead>
          <div className="grid g3">
            {s.tables.map(t => (
              <button key={t.name} onClick={() => setTable(t.name)} className="card" style={{ textAlign: 'left', borderColor: t.name === table ? 'var(--maroon)' : undefined, boxShadow: t.name === table ? '0 0 0 3px #8a153812' : undefined }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><h3 className="sub">{t.name}</h3><span className="num note">{fmtInt(t.rows)} rows</span></div>
                <div style={{ marginTop: 8, display: 'grid', gap: 3 }}>
                  {t.columns.slice(0, 9).map(c => (
                    <div key={c.name} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 12.5 }}>
                      <span className="mono">{c.name}</span>
                      <span className="note" style={{ textAlign: 'right' }}>{c.type === 'number' ? `${fmtInt(c.min!)}–${fmtInt(c.max!)}` : c.values ? `${c.values.length} values` : `${fmtInt(c.distinct)} distinct`}</span>
                    </div>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </section>
        {table && <BNView table={table} />}
        {table && <Browser table={table} />}
      </>}
    </>
  )
}

function BNView({ table }: { table: string }) {
  const [bn, setBn] = useState<BN | null>(null)
  const [edge, setEdge] = useState(0)
  useEffect(() => {
    setBn(null)
    api<BN>(`/bn/${table}`).then(b => { setBn(b); setEdge(b.edges.reduce((best, e, i, a) => (e.mi > a[best].mi ? i : best), 0)) }).catch(() => setBn(null))
  }, [table])
  const layout = useMemo(() => {
    if (!bn) return null
    // undirected forest: root each component at its first column, lay components side by side
    const nbr: Record<string, string[]> = {}
    bn.edges.forEach(e => { (nbr[e.parent] ??= []).push(e.child); (nbr[e.child] ??= []).push(e.parent) })
    const pos: Record<string, { x: number; y: number }> = {}
    let leaf = 0
    const place = (n: string, from: string | null, d: number): number => {
      const ks = (nbr[n] ?? []).filter(k => k !== from)
      const x = ks.length ? ks.map(k => place(k, n, d + 1)).reduce((a, b) => a + b) / ks.length : leaf++
      pos[n] = { x, y: d }
      return x
    }
    bn.nodes.forEach(nd => { if (!pos[nd.id]) place(nd.id, null, 0) })
    return { pos, w: Math.max(leaf, 1), h: Math.max(...Object.values(pos).map(p => p.y)) + 1 }
  }, [bn])
  if (!bn || !layout) return <section className="section"><Skeleton h={260} /></section>
  const W = Math.max(layout.w * 140, 300), H = layout.h * 90 + 20
  const P = (id: string) => ({ x: 70 + layout.pos[id].x * 140, y: 34 + layout.pos[id].y * 90 })
  const maxMi = Math.max(...bn.edges.map(e => e.mi), 1e-6)
  const e = bn.edges[edge]
  const pn = bn.nodes.find(n => n.id === e?.parent), cn = bn.nodes.find(n => n.id === e?.child)
  return (
    <section className="section">
      <SectionHead eyebrow="Bayesian network" title={<>Chow-Liu forest for <em>{table}</em></>}>
        Structure: the maximum-likelihood forest linking the most strongly related columns by mutual information (thicker = stronger). Parameters: MLE counts. Click an edge to see its conditional probability table.
      </SectionHead>
      <div className="grid g2">
        <Card k={`${bn.nodes.length} variables · ${bn.edges.length} edges`}>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} className="chart" role="img" aria-label={`Bayesian network for ${table}`}>
            {bn.edges.map((x, i) => {
              const a = P(x.parent), b = P(x.child)
              return <g key={i} onClick={() => setEdge(i)} style={{ cursor: 'pointer' }}>
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth={14} />
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={i === edge ? 'var(--maroon)' : '#cdbfc0'} strokeWidth={1 + 6 * x.mi / maxMi} strokeLinecap="round" />
                <text x={(a.x + b.x) / 2 + 6} y={(a.y + b.y) / 2} style={{ fontSize: 10 }}>{x.mi.toFixed(2)}</text>
              </g>
            })}
            {bn.nodes.map(n => { const p = P(n.id); return (
              <g key={n.id}>
                <rect x={p.x - 58} y={p.y - 14} width={116} height={28} rx={14} fill="#fff" stroke="var(--maroon-line)" />
                <text x={p.x} y={p.y + 4} textAnchor="middle" style={{ fontSize: 11, fill: 'var(--ink)', fontWeight: 600 }}>{n.id}</text>
              </g>) })}
          </svg>
          <p className="note">Edge label = bias-corrected mutual information (nats). Edges are kept only if a G-test rejects independence (p &lt; 0.001), so unrelated columns stay unconnected (a forest).</p>
          {bn.pruned.length > 0 && <p className="note" style={{ marginTop: 6 }}>Rejected as noise: {bn.pruned.slice(0, 6).map(p => `${p.a}–${p.b} (raw MI ${p.mi.toFixed(3)})`).join(', ')}{bn.pruned.length > 6 ? ` and ${bn.pruned.length - 6} more` : ''}.</p>}
        </Card>
        <Card k="Conditional probability table" title={e ? `P(${e.child} | ${e.parent})` : 'No edges'}>
          {e && pn && cn && (pn.bins.length * cn.bins.length <= 1600 ? (
            <Heatmap title={`P(${e.child} given ${e.parent})`} data={e.cpt} rows={pn.bins} cols={cn.bins} cell={Math.max(10, Math.min(26, 520 / cn.bins.length))}
              color={v => seq(Math.sqrt(v))} fmt={v => v.toFixed(3)} />
          ) : <p className="note">Table too large to draw ({pn.bins.length}×{cn.bins.length} bins). Join keys use 64 bins.</p>)}
          <p className="note" style={{ marginTop: 8 }}>Each row is one parent value and sums to 1. A sharp, non-uniform row means the parent predicts the child, which the independence assumption ignores.</p>
        </Card>
      </div>
    </section>
  )
}

function Browser({ table }: { table: string }) {
  const [page, setPage] = useState(1)
  const [d, setD] = useState<Rows | null>(null)
  useEffect(() => setPage(1), [table])
  useEffect(() => { api<Rows>(`/tables/${table}?page=${page}&page_size=15`).then(setD).catch(() => setD(null)) }, [table, page])
  const pages = d ? Math.ceil(d.total / d.page_size) : 1
  return (
    <section className="section">
      <Card k="Table browser" title={table} right={
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button className="chip" disabled={page <= 1} onClick={() => setPage(p => p - 1)} aria-label="Previous page"><ChevronLeft size={14} /></button>
          <span className="note num">{page} / {pages}</span>
          <button className="chip" disabled={page >= pages} onClick={() => setPage(p => p + 1)} aria-label="Next page"><ChevronRight size={14} /></button>
        </div>}>
        {!d ? <Skeleton h={200} /> : (
          <div className="scroll-x"><table className="tbl mono">
            <thead><tr>{d.columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>{d.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody>
          </table></div>
        )}
      </Card>
    </section>
  )
}
