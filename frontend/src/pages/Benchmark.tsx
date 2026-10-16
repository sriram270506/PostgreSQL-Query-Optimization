import { useEffect, useState } from 'react'
import { Copy } from 'lucide-react'
import { api, EST_COLOR, EST_LABEL, fmtCompact, fmtInt, fmtMs, fmtQ, pct } from '../lib'
import type { Bench, Tuning } from '../types'
import { Card, ErrorBox, EstName, Legend, SectionHead, Seg, Skeleton, useTip } from '../components/ui'
import { GroupedLogBars, HBars, LogScatter } from '../components/charts'

const PCTS: [string, keyof Bench['accuracy'][string]['overall']][] = [['median', 'median'], ['p90', 'p90'], ['p95', 'p95'], ['p99', 'p99'], ['max', 'max']]

export default function Benchmark() {
  const [b, setB] = useState<Bench | null>(null)
  const [err, setErr] = useState('')
  const [slice, setSlice] = useState<'overall' | 'correlated'>('overall')
  useEffect(() => { api<Bench>('/benchmark').then(setB).catch(e => setErr(e.message)) }, [])
  if (err) return <section className="section"><ErrorBox error={`${err}. Run: python -m scripts.benchmark`} /></section>
  if (!b) return <section className="section"><Skeleton h={420} /></section>
  const A = b.accuracy, names = b.estimators
  const cats = Object.keys(A[names[0]].by_category)
  const top = names.includes('treelstm') ? 'treelstm' : names[names.length - 1]
  const resume = [
    `Built a learned cardinality estimator (Chow-Liu Bayesian networks + FactorJoin + TreeLSTM/MLP fusion) on a custom in-memory SQL engine; cut p99 Q-error from ${fmtQ(A.postgres.overall.p99)} (PostgreSQL planner) to ${fmtQ(A[top].overall.p99)} and worst-case error from ${fmtQ(A.postgres.overall.max)} to ${fmtQ(A[top].overall.max)} on ${fmtInt(b.n_test)} held-out queries.`,
    `Estimates within 2× of the truth rose from ${pct(A.postgres.overall.within_2x, 1)} to ${pct(A[top].overall.within_2x, 1)}; on correlated-predicate queries p99 error fell from ${fmtQ(A.postgres.correlated.p99)} to ${fmtQ(A[top].correlated.p99)}.`,
    `Optimal join orders on 3–5 table queries rose from ${pct(b.plans.postgres.optimal_rate, 1)} to ${pct(b.plans[top].optimal_rate, 1)} (worst plan-cost blow-up ${b.plans.postgres.max_ratio.toFixed(0)}× → ${b.plans[top].max_ratio.toFixed(0)}×; FactorJoin alone reaches ${pct(b.plans.factorjoin.optimal_rate, 1)}), at ${fmtMs(b.latency[top].p50_ms)} median CPU latency per estimate.`,
    b.tuning.treelstm && b.tuning.mlp ? `Tuned both networks with Optuna (TPE sampler + median pruning, ${b.tuning.mlp.n_trials + b.tuning.treelstm.n_trials} trials) over ${fmtInt(b.n_train)} training queries.` : '',
  ].filter(Boolean)

  return (
    <>
      <section style={{ paddingTop: 48 }}>
        <div className="eyebrow">Benchmark</div>
        <h1 className="display" style={{ fontSize: 'clamp(36px,4.4vw,54px)' }}>Head to head on <em>{fmtInt(b.n_test)} unseen queries</em></h1>
        <p className="lede">The test split is never seen during training or tuning. The true counts come from exact execution. Train / validation / test: {fmtInt(b.n_train)} / {fmtInt(b.n_val)} / {fmtInt(b.n_test)}.</p>
      </section>

      <section className="section" style={{ paddingTop: 30 }}>
        <Card k="Resume-ready" title="Measured results" right={<button className="chip" onClick={() => navigator.clipboard?.writeText(resume.map(r => '• ' + r).join('\n'))}><Copy size={13} aria-hidden /> Copy</button>}>
          <ul style={{ margin: 0, paddingLeft: 18, display: 'grid', gap: 10 }}>
            {resume.map(r => <li key={r} style={{ fontSize: 14.5, lineHeight: 1.6, color: 'var(--ink-2)' }}>{r}</li>)}
          </ul>
        </Card>
      </section>

      <section className="section">
        <SectionHead eyebrow="Accuracy" title={<>Errors where it <em>hurts</em>: the tail</>}>
          Median errors are small for every estimator because most predicates touch independent columns. The bad cases are in the 95th and 99th percentiles, and those are what break query plans.
        </SectionHead>
        <Card right={<Seg value={slice} onChange={setSlice} options={[['overall', 'All queries'], ['correlated', 'Correlated predicates']]} />} k="Q-error percentiles · log scale">
          <GroupedLogBars groups={PCTS.map(p => p[0])} series={names} value={(g, s) => A[s][slice][PCTS.find(p => p[0] === g)![1]] as number} />
          <div style={{ marginTop: 10 }}><Legend names={names} /></div>
        </Card>
        <div className="grid g2" style={{ marginTop: 18 }}>
          <Card k="Table view" title="Q-error summary">
            <div className="scroll-x"><table className="tbl num">
              <thead><tr><th>Estimator</th><th className="r">Median</th><th className="r">p95</th><th className="r">p99</th><th className="r">Max</th><th className="r">≤2×</th></tr></thead>
              <tbody>{names.map(n => {
                const s = A[n][slice]
                return <tr key={n}><td><EstName name={n} /></td><td className="r">{fmtQ(s.median)}</td><td className="r">{fmtQ(s.p95)}</td><td className="r"><b>{fmtQ(s.p99)}</b></td><td className="r">{fmtQ(s.max)}</td><td className="r">{pct(s.within_2x, 1)}</td></tr>
              })}</tbody>
            </table></div>
            <p className="note" style={{ marginTop: 10 }}>{slice === 'correlated' ? `${A[names[0]].correlated.n} queries with 2 or more conditions on correlated columns.` : `${A[names[0]].overall.n} queries.`}</p>
          </Card>
          <Card k="By join size" title="p95 Q-error per category">
            <div className="scroll-x"><table className="tbl num">
              <thead><tr><th>Category</th>{names.map(n => <th key={n} className="r"><span className="dot" style={{ background: EST_COLOR[n], marginRight: 6 }} />{EST_LABEL[n].split(' ')[0]}</th>)}</tr></thead>
              <tbody>{cats.map(c => {
                const best = Math.min(...names.map(n => A[n].by_category[c].p95))
                return <tr key={c}><td>{c} <span className="note">({A[names[0]].by_category[c].n})</span></td>
                  {names.map(n => <td key={n} className="r" style={{ fontWeight: A[n].by_category[c].p95 === best ? 700 : 400, color: A[n].by_category[c].p95 === best ? 'var(--maroon)' : undefined }}>{fmtQ(A[n].by_category[c].p95)}</td>)}</tr>
              })}</tbody>
            </table></div>
          </Card>
        </div>
      </section>

      <section className="section">
        <SectionHead eyebrow="Every query" title={<>Estimate vs truth</>}>Points on the dashed diagonal are exact. The shaded band is within 2×. Hover a point to see its numbers.</SectionHead>
        <div className="grid g4">
          {names.map(n => (
            <Card key={n} title={<EstName name={n} />}>
              <LogScatter truth={b.scatter.true} est={b.scatter[n]} name={n} />
              <p className="note num">{pct(A[n].overall.within_2x, 1)} within 2× · {pct(A[n].under, 0)} under-estimated</p>
            </Card>
          ))}
        </div>
      </section>

      <section className="section">
        <SectionHead eyebrow="Consequence" title={<>Better counts, <em>faster plans</em></>}>
          For each 3+ table query, a dynamic-programming join-order optimizer runs once per estimator. The chosen plan is then scored by the true number of intermediate rows it produces.
        </SectionHead>
        <div className="grid g3">
          <Card k="Optimal plan chosen">
            <HBars fmt={v => pct(v, 1)} max={1} rows={names.map(n => ({ key: n, label: <EstName name={n} />, value: b.plans[n].optimal_rate, color: EST_COLOR[n] }))} />
          </Card>
          <Card k="Mean cost vs optimal (1.00 = perfect)">
            <HBars fmt={v => v.toFixed(3) + '×'} rows={names.map(n => ({ key: n, label: <EstName name={n} />, value: b.plans[n].mean_ratio, color: EST_COLOR[n] }))} />
          </Card>
          <Card k="Worst plan cost vs optimal">
            <HBars fmt={v => v.toFixed(2) + '×'} rows={names.map(n => ({ key: n, label: <EstName name={n} />, value: b.plans[n].max_ratio, color: EST_COLOR[n] }))} />
          </Card>
        </div>
      </section>

      <section className="section">
        <SectionHead eyebrow="Speed" title={<>Estimating vs <em>executing</em></>}>Time per query for a single call, including feature extraction. Our tables are small (5k–10k rows), so exact execution is still cheap here. Execution time grows with the data, while estimation time depends only on the query’s size.</SectionHead>
        <Card k="Median latency per query">
          <HBars labelWidth={170} fmt={fmtMs} rows={[...names.map(n => ({ key: n, label: <EstName name={n} />, value: b.latency[n].p50_ms, color: EST_COLOR[n] })),
            { key: 'exec', label: <b>Exact execution</b>, value: b.latency.exact_execution.p50_ms, color: '#a39699' }]} />
          <p className="note" style={{ marginTop: 8 }}>p95: {names.map(n => `${EST_LABEL[n].split(' ')[0]} ${fmtMs(b.latency[n].p95_ms)}`).join(' · ')} · execution {fmtMs(b.latency.exact_execution.p95_ms)}</p>
        </Card>
      </section>

      <TuningSection />
    </>
  )
}

function TuningSection() {
  const [name, setName] = useState<'treelstm' | 'mlp'>('treelstm')
  const [t, setT] = useState<Tuning | null>(null)
  const [err, setErr] = useState('')
  const { show, hide, node } = useTip()
  useEffect(() => { setT(null); setErr(''); api<Tuning>(`/tuning/${name}`).then(setT).catch(e => setErr(e.message)) }, [name])
  const W = 620, H = 220, pad = 38
  const done = t?.trials.filter(x => x.value != null) ?? []
  const vmax = Math.max(...done.map(x => x.value!)), vmin = Math.min(...done.map(x => x.value!))
  const sx = (i: number) => pad + (i / Math.max((t?.trials.length ?? 1) - 1, 1)) * (W - pad - 10)
  const sy = (v: number) => 12 + (1 - (v - vmin) / (vmax - vmin || 1)) * (H - pad - 12)
  let best = Infinity
  const bestLine = t?.trials.map(x => { if (x.value != null && x.state === 'COMPLETE') best = Math.min(best, x.value); return best }) ?? []
  return (
    <section className="section">
      <SectionHead eyebrow="Hyperparameter tuning" title={<>Optuna search, <em>{name === 'treelstm' ? 'TreeLSTM + MLP' : 'fusion MLP'}</em></>}>
        TPE sampler with median pruning. Every trial trains a full model, and weak trials are stopped early. Objective: mean log Q-error on the validation split.
      </SectionHead>
      <div style={{ marginBottom: 14 }}><Seg value={name} onChange={setName} options={[['treelstm', 'TreeLSTM + MLP'], ['mlp', 'Fusion MLP']]} /></div>
      {err && <ErrorBox error={err} />}
      {!t && !err && <Skeleton h={260} />}
      {t && (
        <div className="grid g-side">
          <Card k={`${t.trials.length} trials · ${(t.tune_seconds / 60).toFixed(1)} min`} title="Validation error per trial">
            <div style={{ position: 'relative' }}>
              <svg viewBox={`0 0 ${W} ${H}`} className="chart" style={{ width: '100%', height: 'auto' }} role="img" aria-label="Trial objective values">
                <g className="grid">{[0, .5, 1].map(f => <line key={f} x1={pad} x2={W - 10} y1={sy(vmin + f * (vmax - vmin))} y2={sy(vmin + f * (vmax - vmin))} />)}</g>
                {[0, .5, 1].map(f => <text key={f} x={pad - 6} y={sy(vmin + f * (vmax - vmin)) + 4} textAnchor="end">{(vmin + f * (vmax - vmin)).toFixed(3)}</text>)}
                <polyline fill="none" stroke="var(--maroon)" strokeWidth={2} points={bestLine.map((v, i) => isFinite(v) ? `${sx(i)},${sy(v)}` : '').filter(Boolean).join(' ')} />
                {t.trials.map((x, i) => x.value != null && (
                  <circle key={i} cx={sx(i)} cy={sy(x.value)} r={4.5} fill={x.state === 'PRUNED' ? '#fff' : 'var(--c-treelstm)'} stroke="var(--c-treelstm)" strokeWidth={1.5}
                    onMouseMove={e => show(e, <><b>Trial {x.number}</b> · {x.state.toLowerCase()}<br />val {x.value!.toFixed(4)}<br />{Object.entries(x.params).map(([k, v]) => `${k}=${typeof v === 'number' ? +v.toPrecision(3) : v}`).join(', ')}</>)} onMouseLeave={hide} />
                ))}
                <text x={W - 10} y={H - 8} textAnchor="end">trial →</text>
              </svg>
              {node}
            </div>
            <div className="legend" style={{ marginTop: 6 }}>
              <span><span className="dot" style={{ background: 'var(--c-treelstm)' }} />completed</span>
              <span><span className="dot" style={{ border: '1.5px solid var(--c-treelstm)' }} />pruned (stopped early)</span>
              <span><span style={{ width: 16, height: 2, background: 'var(--maroon)', display: 'inline-block' }} />best so far</span>
            </div>
          </Card>
          <Card k="Winning configuration" title={`${fmtInt(t.n_params)} parameters`}>
            <table className="tbl"><tbody>
              {Object.entries(t.best_params).map(([k, v]) => <tr key={k}><td style={{ color: 'var(--muted)' }}>{k}</td><td className="r mono">{typeof v === 'number' ? (+v.toPrecision(3)).toString() : v}</td></tr>)}
            </tbody></table>
            {Object.keys(t.importance).length > 0 && <>
              <hr className="rule" />
              <div className="k" style={{ fontSize: 11, fontWeight: 700, letterSpacing: '.14em', color: 'var(--faint)', marginBottom: 6 }}>WHAT MATTERED MOST (fANOVA)</div>
              <HBars labelWidth={110} fmt={v => pct(v)} max={1} rows={Object.entries(t.importance).slice(0, 6).map(([k, v]) => ({ key: k, label: <span className="mono">{k}</span>, value: v, color: 'var(--maroon)' }))} />
            </>}
            <p className="note" style={{ marginTop: 10 }}>Test after retraining: median {fmtQ(t.test.median_q)}, p95 {fmtQ(t.test.p95_q)}, p99 {fmtQ(t.test.p99_q)}, max {fmtCompact(t.test.max_q)}×</p>
          </Card>
        </div>
      )}
    </section>
  )
}
