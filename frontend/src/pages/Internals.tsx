import { useEffect, useMemo, useState } from 'react'
import { Play } from 'lucide-react'
import { api, div, fmtCompact, fmtInt, pct, seq } from '../lib'
import type { Neuron, Trace, TraceNode, VizGlobal } from '../types'
import { Card, ErrorBox, SectionHead, Seg, Skeleton, useTip } from '../components/ui'
import { Heatmap, HBars, PointMap } from '../components/charts'

const KIND_COLOR: Record<string, string> = { PRED: '#3d72b0', SCAN: '#d08c1e', JOIN: '#2e9c80', AGG: '#b01e4f' }
const KIND_TEXT: Record<string, [string, string, string]> = {
  PRED: ['Conditions', 'one WHERE condition', 'like edge detectors in a CNN’s first layer'],
  SCAN: ['Tables', 'a table after all its filters', 'like texture/shape detectors'],
  JOIN: ['Joins', 'two or more tables combined', 'like part detectors (eye, wheel)'],
  AGG: ['Whole query', 'the complete query', 'like the final object detector'],
}
const DEFAULT_SQL = "SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id JOIN assignments a ON e.emp_id = a.emp_id WHERE e.city = 'Bangalore' AND p.budget_tier = 5 AND a.hours_per_week BETWEEN 10 AND 30"

export default function Internals() {
  const [g, setG] = useState<VizGlobal | null>(null)
  const [err, setErr] = useState('')
  useEffect(() => { api<VizGlobal>('/viz/global?model=treelstm').then(setG).catch(e => setErr(e.message)) }, [])
  return (
    <>
      <section style={{ paddingTop: 48 }}>
        <div className="eyebrow">Inside the models</div>
        <h1 className="display" style={{ fontSize: 'clamp(36px,4.4vw,54px)' }}>What each layer <em>learned to see</em></h1>
        <p className="lede">A CNN’s first filters find edges, the next ones find shapes, the last ones find objects. Our TreeLSTM builds up a query the same way.
          It reads single conditions first, then whole tables, then joins, then the full query. The MLP on top turns that into a row count.</p>
      </section>
      {err && <section className="section"><ErrorBox error={err} /></section>}
      {!g && !err && <section className="section"><Skeleton h={300} /><p className="note" style={{ marginTop: 8 }}>The first load analyses 800 test queries (about 20 s). Later loads use the cached result.</p></section>}
      {g && g.tree_levels && <>
        <Ladder g={g} />
        <Atlas neurons={g.neurons!} kinds={g.kinds!} />
        <NodeMap g={g} />
      </>}
      {g && <LiveTrace />}
      {g && <MlpSection />}
    </>
  )
}

function Ladder({ g }: { g: VizGlobal }) {
  const levels = g.tree_levels!
  return (
    <section className="section">
      <SectionHead eyebrow="Layer by layer" title={<>From conditions to <em>whole queries</em></>}>
        Linear probe: fit a straight line from a node’s hidden state to that node’s true row count. A high R² means the count can be read straight off the hidden state.
      </SectionHead>
      <div className="grid g4">
        {levels.map((l, i) => (
          <div key={l.kind} className="card" style={{ borderTop: `3px solid ${KIND_COLOR[l.kind]}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span className="k" style={{ fontSize: 11, fontWeight: 700, letterSpacing: '.14em', color: 'var(--faint)' }}>LEVEL {i + 1} · {l.kind}</span>
              <span className="note num">{fmtInt(l.n_nodes)} nodes</span>
            </div>
            <h3 className="sub" style={{ marginTop: 10 }}>{KIND_TEXT[l.kind][0]}</h3>
            <p className="note">Sees {KIND_TEXT[l.kind][1]}, {KIND_TEXT[l.kind][2]}.</p>
            <div style={{ font: '500 34px var(--serif)', color: 'var(--maroon)', marginTop: 14 }} className="num">{l.probe_r2.toFixed(2)}</div>
            <div className="note">R² recovering true rows (log scale)</div>
            <hr className="rule" />
            <GateBars i={l.gate_input} f={l.gate_forget} o={l.gate_output} />
            <p className="note" style={{ marginTop: 8 }}>{l.active_units} active units</p>
          </div>
        ))}
      </div>
      <p className="note" style={{ marginTop: 12 }}>Gates: <b>input</b> = how much new information is written, <b>forget</b> = how much of each child’s memory is kept, <b>output</b> = how much memory is passed upward. Each value is the mean over all nodes at that level.</p>
    </section>
  )
}

function GateBars({ i, f, o }: { i: number; f: number | null; o: number }) {
  const rows: [string, number | null][] = [['input', i], ['forget', f], ['output', o]]
  return (
    <div>{rows.map(([k, v]) => (
      <div key={k} style={{ display: 'grid', gridTemplateColumns: '52px 1fr 36px', gap: 8, alignItems: 'center', fontSize: 12, padding: '3px 0' }}>
        <span style={{ color: 'var(--muted)' }}>{k}</span>
        <div className="bar-track" style={{ height: 7 }}>{v != null && <div className="bar-fill" style={{ width: `${v * 100}%`, background: 'var(--maroon)' }} />}</div>
        <span className="num" style={{ textAlign: 'right' }}>{v == null ? '–' : v.toFixed(2)}</span>
      </div>))}</div>
  )
}

function Atlas({ neurons, kinds }: { neurons: Neuron[]; kinds: string[] }) {
  const [k, setK] = useState('ALL')
  const shown = useMemo(() => neurons.filter(n => k === 'ALL' || n.detects === k).sort((a, b) => b.purity - a.purity), [neurons, k])
  const counts = Object.fromEntries(kinds.map(x => [x, neurons.filter(n => n.detects === x).length]))
  return (
    <section className="section">
      <SectionHead eyebrow="Neuron atlas" title={<>Every hidden unit, <em>and what it fires on</em></>}>
        Like plotting CNN filters: for each unit we take the 200 tree nodes that activate it most, across 800 queries, and see what they share.
      </SectionHead>
      <div className="chips" style={{ marginBottom: 14 }}>
        <button className={`chip ${k === 'ALL' ? 'on' : ''}`} onClick={() => setK('ALL')}>All {neurons.length}</button>
        {kinds.map(x => <button key={x} className={`chip ${k === x ? 'on' : ''}`} onClick={() => setK(x)}>{KIND_TEXT[x][0]} · {counts[x]}</button>)}
      </div>
      <div className="grid g4">
        {shown.slice(0, 24).map(n => <NeuronCard key={n.unit} n={n} kinds={kinds} />)}
      </div>
      {shown.length > 24 && <p className="note" style={{ marginTop: 10 }}>Showing the 24 most specialised of {shown.length} units.</p>}
    </section>
  )
}

function NeuronCard({ n, kinds }: { n: Neuron; kinds: string[] }) {
  const mx = Math.max(...n.profile.map(Math.abs), 1e-6)
  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="mono" style={{ color: 'var(--faint)' }}>unit {String(n.unit).padStart(3, '0')}</span>
        <span className="tag" style={{ borderColor: KIND_COLOR[n.detects], color: KIND_COLOR[n.detects] }}>{n.detects}</span>
      </div>
      <div style={{ fontWeight: 600, fontSize: 13.5, margin: '10px 0 2px', minHeight: 20 }}>{n.detail || KIND_TEXT[n.detects][0]}</div>
      <div className="note">{pct(n.purity)} of top hits · r = {n.card_corr.toFixed(2)} with rows</div>
      <svg viewBox="0 0 120 34" style={{ width: '100%', height: 34, marginTop: 8 }} role="img" aria-label="Mean activation per node kind">
        <line x1={0} x2={120} y1={17} y2={17} stroke="#e7dfdd" />
        {n.profile.map((v, i) => {
          const h = Math.abs(v) / mx * 15
          return <rect key={i} x={6 + i * 29} width={20} y={v >= 0 ? 17 - h : 17} height={Math.max(h, 1)} rx={2} fill={KIND_COLOR[kinds[i]]} opacity={0.85}><title>{kinds[i]}: {v.toFixed(3)}</title></rect>
        })}
      </svg>
      <div style={{ marginTop: 6 }}>
        {n.top.slice(0, 3).map(t => <div key={t.label} className="mono" style={{ fontSize: 10.5, color: 'var(--ink-2)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={t.label}>▸ {t.label}</div>)}
      </div>
    </div>
  )
}

function NodeMap({ g }: { g: VizGlobal }) {
  const m = g.node_map!
  const [mode, setMode] = useState<'kind' | 'rows'>('kind')
  const tmax = Math.max(...m.target, 1)
  return (
    <section className="section">
      <SectionHead eyebrow="Representation map" title={<>Similar sub-queries <em>cluster together</em></>}>
        Each dot is one tree node’s hidden state, projected to 2-D with PCA ({pct(m.var[0] + m.var[1])} of the variance). Color by node kind, or by how many rows the node really has.
      </SectionHead>
      <div className="grid g-side">
        <Card right={<Seg value={mode} onChange={setMode} options={[['kind', 'Node kind'], ['rows', 'True rows']]} />} k={`${m.xy.length} nodes`}>
          <PointMap xy={m.xy} size={420} color={i => mode === 'kind' ? KIND_COLOR[m.kind[i]] : seq(m.target[i] / tmax)}
            label={i => <><b>{m.kind[i]}</b> · {m.label[i]}<br />true rows ≈ {fmtCompact(10 ** m.target[i])}</>} />
        </Card>
        <Card k="Reading the map">
          {mode === 'kind' ? <div className="legend" style={{ flexDirection: 'column', gap: 10 }}>
            {Object.keys(KIND_COLOR).map(k => <span key={k}><span className="dot" style={{ background: KIND_COLOR[k] }} /><b>{k}</b>&nbsp;{KIND_TEXT[k][1]}</span>)}
          </div> : <div>
            <div style={{ height: 10, borderRadius: 999, background: `linear-gradient(90deg, ${seq(0)}, ${seq(1)})` }} />
            <div style={{ display: 'flex', justifyContent: 'space-between' }} className="note"><span>1 row</span><span>{fmtCompact(10 ** tmax)} rows</span></div>
          </div>}
          <hr className="rule" />
          <p className="note">Nodes of the same kind form their own regions. Within a region, color follows row count, so the network has learned a size direction without being told where it is.</p>
        </Card>
      </div>
    </section>
  )
}

function LiveTrace() {
  const [sql, setSql] = useState(DEFAULT_SQL)
  const [t, setT] = useState<Trace | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true); setErr('')
    try { setT(await api<Trace>('/viz/trace', { sql, model: 'treelstm' })) } catch (e) { setErr((e as Error).message) }
    setBusy(false)
  }
  useEffect(() => { run() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <section className="section">
      <SectionHead eyebrow="Live trace" title={<>Watch one query <em>flow upward</em></>}>
        Every node shows its hidden state as a strip of colored cells (red = positive, blue = negative), its gates, and the row count the probe reads from it, next to the true count.
      </SectionHead>
      <Card>
        <textarea className="editor" style={{ minHeight: 84 }} value={sql} onChange={e => setSql(e.target.value)} aria-label="Query to trace" spellCheck={false} />
        <div style={{ marginTop: 12 }}><button className="btn" onClick={run} disabled={busy}><Play size={15} aria-hidden />{busy ? 'Tracing…' : 'Trace query'}</button></div>
        {err && <div style={{ marginTop: 12 }}><ErrorBox error={err} /></div>}
      </Card>
      {t && t.nodes && (
        <div className="grid g-side" style={{ marginTop: 18 }}>
          <Card k="Bottom-up pass" title="TreeLSTM, leaves first">
            <TraceTree nodes={t.nodes} />
          </Card>
          <div className="grid" style={{ alignContent: 'start' }}>
            <Card k="Final answer">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                <div><div className="note">FactorJoin</div><div className="num" style={{ font: '500 24px var(--serif)' }}>{fmtCompact(t.fj_estimate)}</div></div>
                <div><div className="note">TreeLSTM + MLP</div><div className="num" style={{ font: '500 24px var(--serif)', color: 'var(--maroon)' }}>{fmtCompact(t.estimate)}</div></div>
                <div><div className="note">True</div><div className="num" style={{ font: '500 24px var(--serif)' }}>{fmtCompact(t.true_count)}</div></div>
              </div>
              <p className="note" style={{ marginTop: 10 }}>The MLP multiplied the FactorJoin estimate by <b>{Math.exp(t.residual).toFixed(3)}×</b> (residual {t.residual >= 0 ? '+' : ''}{t.residual.toFixed(3)} in log space).</p>
            </Card>
            <Card k="Why: integrated gradients" title="Which inputs pushed the correction">
              <Attribution items={t.input.map(x => ({ group: x.group, v: x.attribution }))} />
            </Card>
            <Card k="MLP activations for this query">
              {t.mlp.map(l => (
                <div key={l.name} style={{ marginBottom: 10 }}>
                  <div className="note">{l.name} · {l.acts.filter(a => a > 0).length}/{l.acts.length} units firing</div>
                  <Strip v={l.acts} />
                </div>
              ))}
            </Card>
          </div>
        </div>
      )}
    </section>
  )
}

function Strip({ v, h = 12 }: { v: number[]; h?: number }) {
  const mx = Math.max(...v.map(Math.abs), 1e-6)
  return (
    <svg viewBox={`0 0 ${v.length} 1`} preserveAspectRatio="none" style={{ width: '100%', height: h, display: 'block', borderRadius: 3 }} aria-hidden>
      {v.map((x, i) => <rect key={i} x={i} y={0} width={1.02} height={1} fill={div(x / mx)} />)}
    </svg>
  )
}

function TraceTree({ nodes }: { nodes: TraceNode[] }) {
  const { show, hide, node } = useTip()
  // order: deepest first so the reader follows the computation
  const depth = (n: TraceNode): number => (n.parent < 0 ? 0 : 1 + depth(nodes[n.parent]))
  const rows = [...nodes].sort((a, b) => a.height - b.height || a.id - b.id)
  return (
    <div style={{ position: 'relative' }}>
      {rows.map(n => {
        const err = n.probe_log10 == null ? null : Math.abs(n.probe_log10 - n.true_log10)
        return (
          <div key={n.id} style={{ display: 'grid', gridTemplateColumns: '1fr 120px', gap: 12, padding: '10px 0', borderTop: '1px solid var(--line)', marginLeft: depth(n) * 0 }}
            onMouseMove={e => show(e, <><b>{n.kind}</b> · height {n.height}<br />gates i {n.gate_i.toFixed(2)} · f {n.gate_f?.toFixed(2) ?? '–'} · o {n.gate_o.toFixed(2)}</>)} onMouseLeave={hide}>
            <div style={{ minWidth: 0 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span className="tag" style={{ borderColor: KIND_COLOR[n.kind], color: KIND_COLOR[n.kind], minWidth: 48, justifyContent: 'center' }}>{n.kind}</span>
                <span className="mono" style={{ fontSize: 11.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{n.label}</span>
              </div>
              <div style={{ marginTop: 7 }}><Strip v={n.h} /></div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className="num" style={{ fontSize: 13 }}><span className="note">probe </span><b>{n.probe_log10 == null ? '–' : fmtCompact(10 ** n.probe_log10)}</b></div>
              <div className="num note">true {fmtCompact(10 ** n.true_log10)}</div>
              {err != null && <span className={`tag ${err < 0.3 ? 'good' : err > 1 ? 'bad' : ''}`} style={{ marginTop: 3 }}>{(10 ** err).toFixed(1)}× off</span>}
            </div>
          </div>
        )
      })}
      {node}
      <p className="note" style={{ marginTop: 8 }}>Rows are in computation order: leaves (height 0) first, root last. Each node’s state depends only on the rows above it.</p>
    </div>
  )
}

function Attribution({ items }: { items: { group: string; v: number }[] }) {
  const mx = Math.max(...items.map(i => Math.abs(i.v)), 1e-6)
  return (
    <div>{items.map(i => (
      <div key={i.group} style={{ display: 'grid', gridTemplateColumns: '170px 1fr 56px', gap: 10, alignItems: 'center', fontSize: 12.5, padding: '5px 0' }}>
        <span>{i.group}</span>
        <div style={{ position: 'relative', height: 10 }}>
          <div style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 1, background: '#d9cfcf' }} />
          <div style={{ position: 'absolute', top: 0, height: 10, borderRadius: 999, background: i.v >= 0 ? 'var(--maroon)' : 'var(--c-postgres)',
            left: i.v >= 0 ? '50%' : `${50 - Math.abs(i.v) / mx * 50}%`, width: `${Math.abs(i.v) / mx * 50}%` }} />
        </div>
        <span className="num" style={{ textAlign: 'right' }}>{i.v >= 0 ? '+' : ''}{i.v.toFixed(3)}</span>
      </div>))}
      <p className="note" style={{ marginTop: 6 }}>Right (maroon) raises the estimate, left (blue) lowers it. The values sum to the MLP’s log correction.</p>
    </div>
  )
}

function MlpSection() {
  const [model, setModel] = useState<'treelstm' | 'mlp'>('treelstm')
  const [g, setG] = useState<VizGlobal | null>(null)
  const [err, setErr] = useState('')
  useEffect(() => { setG(null); api<VizGlobal>(`/viz/global?model=${model}`).then(setG).catch(e => setErr(e.message)) }, [model])
  const cmax = g ? Math.max(...g.pca_color, 1) : 1
  return (
    <section className="section">
      <SectionHead eyebrow="The MLP head" title={<>How the MLP <em>sorts queries by size</em></>}>
        Each panel places every test query by one MLP layer’s activations (PCA), colored by its true row count. The input already contains FactorJoin’s estimate, so size can be read from the very first layer. The hidden layers keep that ordering and shrink it into the few units that drive the correction.
      </SectionHead>
      <div style={{ marginBottom: 14 }}><Seg value={model} onChange={setModel} options={[['treelstm', 'TreeLSTM + MLP'], ['mlp', 'MLP only']]} /></div>
      {err && <ErrorBox error={err} />}
      {!g && !err && <Skeleton h={260} />}
      {g && <>
        <div className="grid g4">
          {g.mlp_layers.slice(0, 4).map(l => (
            <Card key={l.name} k={`${l.name} · ${l.width} units`} title={<span className="num">R² {l.probe_r2.toFixed(3)}</span>}>
              <PointMap xy={l.pca} size={240} r={2.2} color={i => seq(g.pca_color[i] / cmax)} label={i => <>true rows ≈ {fmtCompact(10 ** g.pca_color[i])}</>} />
              {l.name !== 'input' && <p className="note" style={{ marginTop: 6 }}>{pct(l.dead_frac)} units never fire</p>}
            </Card>
          ))}
        </div>
        <div className="grid g2" style={{ marginTop: 18 }}>
          <Card k="First-layer receptive fields" title="Which inputs each unit listens to">
            <Heatmap title="Mean absolute weight from each input group to the 32 strongest first-layer units"
              data={g.receptive_fields.units[0].map((_, j) => g.receptive_fields.units.map(u => u[j]))}
              rows={g.receptive_fields.groups} cols={g.receptive_fields.units.map((_, i) => `u${i}`)} cell={16}
              color={v => seq(v / Math.max(...g.receptive_fields.units.flat()))} />
            <p className="note">Darker = stronger weight. Like viewing a CNN’s first-layer kernels, this shows the raw input pattern each unit responds to.</p>
          </Card>
          <Card k="Global attribution" title="Average influence on the correction">
            <HBars labelWidth={190} fmt={v => v.toFixed(3)} rows={g.attribution.map(a => ({ key: a.group, label: a.group, value: a.mean_abs, color: 'var(--maroon)' }))} />
            <p className="note" style={{ marginTop: 8 }}>Mean |integrated gradient| over {g.n_queries} queries. The model raises FactorJoin’s estimate on {pct(g.residual.frac_up)} of queries. The typical correction is {Math.exp(g.residual.mean_abs).toFixed(2)}×.</p>
          </Card>
        </div>
      </>}
    </section>
  )
}
