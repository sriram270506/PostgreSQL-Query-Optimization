import { useEffect, useState } from 'react'
import { Copy } from 'lucide-react'
import { api, chartEstimators, EST_COLOR, EST_LABEL, EST_SHORT, fmtCompact, fmtInt, fmtMs, fmtQ, getDataset, pct } from '../lib'
import type { Acc, Bench, PlanStats, QSummary, Sig, Tuning } from '../types'
import { Card, ErrorBox, EstName, Legend, SectionHead, Seg, Skeleton, useTip } from '../components/ui'
import { GroupedLogBars, HBars, LogScatter } from '../components/charts'

const PCTS: [string, keyof QSummary][] = [['median', 'median'], ['p90', 'p90'], ['p95', 'p95'], ['p99', 'p99'], ['max', 'max']]
const LEARNED = ['treelstm', 'mlp', 'mscn']
const ORDER = ['postgres_live', 'postgres_xstats', 'factorjoin', 'mscn', 'mlp', 'treelstm']

function bestLearned(acc: Record<string, Acc>): string {
  return LEARNED.filter(n => acc[n]).sort((a, b) => acc[a].overall.p99 - acc[b].overall.p99)[0] ?? 'factorjoin'
}
const pgOf = (_acc: Record<string, unknown>) => 'postgres_live'
const estOrder = (o: Record<string, unknown>) => ORDER.filter(n => n in o)
const ciTxt = (c?: [number, number]) => (c ? `[${fmtQ(c[0]).replace('×', '')}–${fmtQ(c[1])}]` : '')

export default function Benchmark() {
  const [b, setB] = useState<Bench | null>(null)
  const [err, setErr] = useState('')
  const [set, setSet] = useState<'test' | 'ceb'>('test')
  useEffect(() => { api<Bench>('/benchmark').then(setB).catch(e => setErr(e.message)) }, [])
  if (err) return <section className="section"><ErrorBox error={`${err}. Run: python -m scripts.benchmark --dataset ${getDataset()}`} /></section>
  if (!b) return <section className="section"><Skeleton h={420} /></section>
  const useCeb = set === 'ceb' && !!b.ceb
  const A = useCeb ? b.ceb!.accuracy : b.accuracy
  const sig = useCeb ? b.ceb!.significance : b.significance
  const scatter = useCeb ? b.ceb!.scatter : b.scatter
  const names = chartEstimators(Object.keys(A))
  const top = bestLearned(b.accuracy)
  const pg = pgOf(b.accuracy)
  const isStats = b.dataset.startsWith('stats')
  const cats = Object.keys(b.accuracy[names[0]]).filter(k => k.startsWith('cat:')).sort()
  const n = useCeb ? A[names[0]].overall.n : b.n_test
  const pgPlans = b.plans?.postgres_live ?? null

  const resume = [
    `Built a learned cardinality estimator (Chow-Liu Bayesian-network forests + FactorJoin + TreeLSTM/MLP fusion) on a custom in-memory SQL engine whose exact counts match all 146 STATS-CEB ground-truth cardinalities.`,
    `On ${fmtInt(b.n_test)} held-out ${isStats ? 'STATS (Stack Exchange)' : 'HR'} queries, cut p99 Q-error from ${fmtQ(b.accuracy[pg].overall.p99)} (PostgreSQL 16) to ${fmtQ(b.accuracy[top].overall.p99)} (95% CI ${fmtQ(b.accuracy[top].ci95.p99[0])}–${fmtQ(b.accuracy[top].ci95.p99[1])})${b.accuracy.postgres_xstats ? `, vs ${fmtQ(b.accuracy.postgres_xstats.overall.p99)} for PostgreSQL with CREATE STATISTICS` : ''}.`,
    b.ceb ? `On the official STATS-CEB workload: median / p95 Q-error ${fmtQ(b.ceb.accuracy[top].overall.median)} / ${fmtQ(b.ceb.accuracy[top].overall.p95)} vs PostgreSQL's ${fmtQ(b.ceb.accuracy[pgOf(b.ceb.accuracy)].overall.median)} / ${fmtQ(b.ceb.accuracy[pgOf(b.ceb.accuracy)].overall.p95)}.` : '',
    b.plans && pgPlans && b.plans[top] ? `Optimal join orders ${pct(b.plans[top].optimal_rate, 1)} vs ${pct(pgPlans.optimal_rate, 1)} for PostgreSQL; worst plan-cost blow-up ${fmtCompact(pgPlans.max_ratio)}× → ${b.plans[top].max_ratio.toFixed(1)}×.` : '',
    `Production guards: ${Object.values(b.ensemble)[0]?.seeds ?? 5}-seed ensembles with uncertainty fallback, provable upper-bound clamping and order-invariant featurisation; ${Object.values(b.tuning).reduce((s, t) => s + t.n_trials, 0)} Optuna trials.`,
  ].filter(Boolean)

  return (
    <>
      <section style={{ paddingTop: 48 }}>
        <div className="eyebrow">Benchmark · {isStats ? 'STATS (real Stack Exchange data)' : 'HR (synthetic)'}</div>
        <h1 className="display" style={{ fontSize: 'clamp(36px,4.4vw,54px)' }}>Head to head on <em>{fmtInt(n)} unseen queries</em></h1>
        <p className="lede">The test set is never used for training, tuning or early stopping. True counts come from exact execution. PostgreSQL numbers come from a real PostgreSQL 16 server, with and without extended statistics.
          Train / validation / test: {fmtInt(b.n_train)} / {fmtInt(b.n_val)} / {fmtInt(b.n_test)}{b.ceb ? ', plus the 146 official STATS-CEB queries' : ''}.</p>
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
          Q-error = how many times off an estimate is. Medians are close to 1× for most methods. The worst 1–5% of queries are what break query plans. Brackets show bootstrap 95% confidence intervals. All PostgreSQL numbers come from a real PostgreSQL 16 server.
        </SectionHead>
        {b.ceb && <div style={{ marginBottom: 14 }}><Seg value={set} onChange={setSet} options={[['test', `Generated test set (${fmtInt(b.n_test)})`], ['ceb', 'STATS-CEB (146 official)']]} /></div>}
        <Card k="Q-error percentiles · log scale">
          <GroupedLogBars groups={PCTS.map(p => p[0])} series={names} value={(g, s) => A[s].overall[PCTS.find(p => p[0] === g)![1]] as number} />
          <div style={{ marginTop: 10 }}><Legend names={names} /></div>
        </Card>
        <div className="grid g2" style={{ marginTop: 18 }}>
          <Card k="Table view" title="Q-error with 95% confidence intervals">
            <div className="scroll-x"><table className="tbl num">
              <thead><tr><th>Estimator</th><th className="r">Median</th><th className="r">p95</th><th className="r">p99</th><th className="r">Max</th><th className="r">≤2×</th></tr></thead>
              <tbody>{estOrder(A).map(nm => {
                const s = A[nm].overall, c = A[nm].ci95
                return <tr key={nm}><td><EstName name={nm} /></td>
                  <td className="r">{fmtQ(s.median)}<span className="ci">{ciTxt(c.median)}</span></td>
                  <td className="r">{fmtQ(s.p95)}<span className="ci">{ciTxt(c.p95)}</span></td>
                  <td className="r"><b>{fmtQ(s.p99)}</b><span className="ci">{ciTxt(c.p99)}</span></td>
                  <td className="r">{fmtQ(s.max)}</td><td className="r">{pct(s.within_2x, 1)}</td></tr>
              })}</tbody>
            </table></div>
          </Card>
          <Card k="Statistical significance" title="Paired Wilcoxon signed-rank test on per-query log Q-error">
            <SigTable sig={sig} />
          </Card>
        </div>
        {!useCeb && cats.length > 0 && (
          <div style={{ marginTop: 18 }}><Card k="By join size" title="p95 Q-error per category">
            <div className="scroll-x"><table className="tbl num">
              <thead><tr><th>Category</th>{names.map(nm => <th key={nm} className="r"><span className="dot" style={{ background: EST_COLOR[nm], marginRight: 6 }} />{EST_SHORT[nm]}</th>)}</tr></thead>
              <tbody>{cats.map(c => {
                const vals = names.map(nm => (A[nm][c] as QSummary).p95)
                const best = Math.min(...vals)
                return <tr key={c}><td>{c.slice(4)} <span className="note">({(A[names[0]][c] as QSummary).n})</span></td>
                  {names.map((nm, i) => <td key={nm} className="r" style={{ fontWeight: vals[i] === best ? 700 : 400, color: vals[i] === best ? 'var(--maroon)' : undefined }}>{fmtQ(vals[i])}</td>)}</tr>
              })}</tbody>
            </table></div>
          </Card></div>
        )}
      </section>

      <section className="section">
        <SectionHead eyebrow="Every query" title={<>Estimate vs truth</>}>Points on the dashed diagonal are exact. The shaded band is within 2×. Hover a point to see its numbers.</SectionHead>
        <div className="grid g3">
          {names.map(nm => (
            <Card key={nm} title={<EstName name={nm} />}>
              <LogScatter truth={scatter.true} est={scatter[nm]} name={nm} />
              <p className="note num">{pct(A[nm].overall.within_2x, 1)} within 2× · {pct(A[nm].under, 0)} under-estimated</p>
            </Card>
          ))}
        </div>
      </section>

      <Generalisation b={b} />

      {b.plans && (
        <section className="section">
          <SectionHead eyebrow="Consequence" title={<>Better counts, <em>cheaper plans</em></>}>
            For every test query with 3 or more tables, a dynamic-programming optimizer picks a join order using each estimator's sub-plan estimates. The chosen plan is then costed with the true number of intermediate rows.
          </SectionHead>
          <PlanCards plans={b.plans} />
          {b.ceb?.plans && <><h3 className="sub" style={{ margin: '18px 0 8px' }}>STATS-CEB (146 official queries)</h3><PlanCards plans={b.ceb.plans} /></>}
        </section>
      )}

      <Robustness b={b} />

      <section className="section">
        <SectionHead eyebrow="Speed" title={<>Estimating vs <em>executing</em></>}>Single-query latency, guards and feature extraction included. Execution grows with data size. Estimation depends only on the query.</SectionHead>
        <Card k="Latency per query (p50 · p95 · p99)">
          <div className="scroll-x"><table className="tbl num">
            <thead><tr><th>Estimator</th><th className="r">p50</th><th className="r">p95</th><th className="r">p99</th></tr></thead>
            <tbody>{[...estOrder(b.latency), 'exact_execution'].filter(nm => b.latency[nm]).map(nm => (
              <tr key={nm}><td>{nm === 'exact_execution' ? <b>Exact execution</b> : <EstName name={nm} />}</td>
                <td className="r">{fmtMs(b.latency[nm].p50_ms)}</td><td className="r">{fmtMs(b.latency[nm].p95_ms)}</td><td className="r">{fmtMs(b.latency[nm].p99_ms)}</td></tr>
            ))}</tbody>
          </table></div>
        </Card>
      </section>

      <TuningSection b={b} />
    </>
  )
}

function SigTable({ sig }: { sig: Sig[] }) {
  return (
    <div className="scroll-x"><table className="tbl num">
      <thead><tr><th>Comparison</th><th className="r">mean log-Q</th><th className="r">closer on</th><th className="r">p-value</th><th /></tr></thead>
      <tbody>{sig.map(s => {
        const aWins = s.mean_log_q_a < s.mean_log_q_b
        return <tr key={s.a + s.b}>
          <td>{EST_SHORT[s.a]} <span className="note">vs</span> {EST_SHORT[s.b]}</td>
          <td className="r">{s.mean_log_q_a.toFixed(3)} / {s.mean_log_q_b.toFixed(3)}</td>
          <td className="r">{pct(s.a_better_frac)} / {pct(s.b_better_frac)}</td>
          <td className="r">{s.p_value < 1e-4 ? '<0.0001' : s.p_value.toFixed(4)}</td>
          <td>{s.p_value < 0.05 ? <span className={`tag ${aWins ? 'good' : 'bad'}`}>{EST_SHORT[aWins ? s.a : s.b]} better</span> : <span className="tag">no significant difference</span>}</td>
        </tr>
      })}</tbody>
    </table>
    <p className="note" style={{ marginTop: 8 }}>"Closer on" = share of queries where each estimate is closer to the truth (ties excluded). p &lt; 0.05 means the difference is unlikely to be chance.</p></div>
  )
}

function PlanCards({ plans }: { plans: Record<string, PlanStats> }) {
  const names = chartEstimators(Object.keys(plans))
  return (
    <div className="grid g3">
      <Card k="Optimal plan chosen"><HBars fmt={v => pct(v, 1)} max={1} rows={names.map(nm => ({ key: nm, label: <EstName name={nm} />, value: plans[nm].optimal_rate, color: EST_COLOR[nm] }))} /></Card>
      <Card k="Mean cost vs optimal (1.00 = perfect)"><HBars fmt={v => v.toFixed(3) + '×'} rows={names.map(nm => ({ key: nm, label: <EstName name={nm} />, value: plans[nm].mean_ratio, color: EST_COLOR[nm] }))} /></Card>
      <Card k="Worst plan cost vs optimal"><HBars fmt={v => fmtCompact(v) + '×'} rows={names.map(nm => ({ key: nm, label: <EstName name={nm} />, value: plans[nm].max_ratio, color: EST_COLOR[nm] }))} /></Card>
    </div>
  )
}

function Generalisation({ b }: { b: Bench }) {
  const splits = Object.entries(b.ood)
  const label: Record<string, string> = { split_tmpl: 'Unseen query templates', split_shape: 'Unseen join sizes (trained on 1–3 tables, tested on 4–5)' }
  if (!splits.length && !b.drift) return null
  return (
    <section className="section">
      <SectionHead eyebrow="Generalisation" title={<>Queries and data the models <em>never saw</em></>}>
        The i.i.d. test set only tests "more of the same". Here the models are retrained with whole query templates, or all larger joins, held out{b.drift ? '. They are also built on a 2013 snapshot of the database and then served on today’s data, which is about twice as large' : ''}.
      </SectionHead>
      <div className="grid g2">
        {splits.map(([k, v]) => (
          <Card key={k} k={`${fmtInt(v.n)} queries`} title={label[k] ?? k}>
            <table className="tbl num"><thead><tr><th>Estimator</th><th className="r">Median</th><th className="r">p95</th><th className="r">p99</th><th className="r">Max</th></tr></thead>
              <tbody>{estOrder(v.accuracy).map(nm => <tr key={nm}><td><EstName name={nm} /></td>
                <td className="r">{fmtQ(v.accuracy[nm].overall.median)}</td><td className="r">{fmtQ(v.accuracy[nm].overall.p95)}</td>
                <td className="r"><b>{fmtQ(v.accuracy[nm].overall.p99)}</b></td><td className="r">{fmtQ(v.accuracy[nm].overall.max)}</td></tr>)}</tbody></table>
          </Card>
        ))}
        {b.drift && (
          <Card k="Data drift" title="Models built on the 2013 snapshot, served on today’s data">
            <table className="tbl num"><thead><tr><th>Variant</th><th>Estimator</th><th className="r">Median</th><th className="r">p95</th><th className="r">p99</th></tr></thead>
              <tbody>{Object.entries(b.drift).flatMap(([variant, r]) => estOrder(r).map((nm, i) => (
                <tr key={variant + nm}><td>{i === 0 ? <b>{variant}</b> : ''}</td><td><EstName name={nm} /></td>
                  <td className="r">{fmtQ(r[nm].median)}</td><td className="r">{fmtQ(r[nm].p95)}</td><td className="r">{fmtQ(r[nm].p99)}</td></tr>)))}</tbody></table>
            <p className="note" style={{ marginTop: 8 }}>Refreshing the Bayesian networks is just recounting and takes about a second. Retraining the networks takes minutes. This table shows how much each step recovers.</p>
          </Card>
        )}
      </div>
    </section>
  )
}

function Robustness({ b }: { b: Bench }) {
  const props = estOrder(b.properties)
  return (
    <section className="section">
      <SectionHead eyebrow="Robustness" title={<>Guarantees, <em>tested</em></>}>
        Each learned model is an ensemble of seeds. When the seeds disagree more than a threshold calibrated on validation data, the answer falls back to FactorJoin. Every estimate is clamped to a provable upper bound.
      </SectionHead>
      <div className="grid g2">
        <Card k="Guards in action" title="How often each guard fired on the test set">
          <table className="tbl num"><thead><tr><th>Estimator</th><th className="r">Uncertainty fallback</th><th className="r">Clamped</th><th className="r">Seed spread (p99)</th></tr></thead>
            <tbody>{estOrder(b.accuracy).filter(nm => b.accuracy[nm].guards).map(nm => (
              <tr key={nm}><td><EstName name={nm} /></td><td className="r">{pct(b.accuracy[nm].guards!.fallback_rate, 1)}</td>
                <td className="r">{pct(b.accuracy[nm].guards!.clamp_rate, 1)}</td>
                <td className="r">{b.seeds[nm] ? `${fmtQ(b.seeds[nm].p99.mean)} ± ${b.seeds[nm].p99.std.toFixed(2)} (${b.seeds[nm].n_seeds} seeds)` : '–'}</td></tr>))}</tbody></table>
        </Card>
        <Card k="Property checks" title="Rules every estimate should obey">
          <table className="tbl num"><thead><tr><th>Estimator</th><th className="r">Order-invariant</th><th className="r">Monotone*</th><th className="r">Raw &gt; bound</th></tr></thead>
            <tbody>{props.map(nm => {
              const p = b.properties[nm]
              return <tr key={nm}><td><EstName name={nm} /></td>
                <td className="r"><span className={`tag ${p.order_invariant_rate === 1 ? 'good' : 'bad'}`}>{pct(p.order_invariant_rate, 1)}</span></td>
                <td className="r">{pct(1 - p.monotone_violation_rate, 1)}</td>
                <td className="r">{pct(p.raw_bound_violation_rate, 1)}</td></tr>
            })}</tbody></table>
          <p className="note" style={{ marginTop: 8 }}>*Share of queries where adding a filter did not raise the estimate. FactorJoin and PostgreSQL are monotone by construction. Learned models are not, and this shows how often they break the rule. "Raw &gt; bound" = estimates the clamp had to correct.</p>
        </Card>
      </div>
    </section>
  )
}

function TuningSection({ b }: { b: Bench }) {
  const models = (['treelstm', 'mlp', 'mscn'] as const).filter(m => b.tuning[m])
  const [name, setName] = useState<string>(models[0] ?? 'treelstm')
  const [t, setT] = useState<Tuning | null>(null)
  const [err, setErr] = useState('')
  const { show, hide, node } = useTip()
  useEffect(() => { setT(null); setErr(''); api<Tuning>(`/tuning/${name}`).then(setT).catch(e => setErr(e.message)) }, [name])
  if (!models.length) return null
  const W = 620, H = 220, pad = 44
  const vals = (t?.trials ?? []).filter(x => x.value != null).map(x => x.value!)
  const vmax = Math.max(...vals), vmin = Math.min(...vals)
  const sx = (i: number) => pad + (i / Math.max((t?.trials.length ?? 1) - 1, 1)) * (W - pad - 10)
  const sy = (v: number) => 12 + (1 - (v - vmin) / (vmax - vmin || 1)) * (H - 52)
  let best = Infinity
  const bestLine = t?.trials.map(x => { if (x.value != null && x.state === 'COMPLETE') best = Math.min(best, x.value); return best }) ?? []
  return (
    <section className="section">
      <SectionHead eyebrow="Hyperparameter tuning" title={<>Optuna search, <em>{EST_LABEL[name]}</em></>}>
        TPE sampler with median pruning. Objective: mean log Q-error on the validation split. The winning configuration is then trained with several seeds to form the ensemble.
      </SectionHead>
      <div style={{ marginBottom: 14 }}><Seg value={name} onChange={setName} options={models.map(m => [m, EST_LABEL[m]] as [string, string])} /></div>
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
                  <circle key={i} cx={sx(i)} cy={sy(x.value)} r={4.5} fill={x.state === 'PRUNED' ? '#fff' : EST_COLOR[name]} stroke={EST_COLOR[name]} strokeWidth={1.5}
                    onMouseMove={e => show(e, <><b>Trial {x.number}</b> · {x.state.toLowerCase()}<br />val {x.value!.toFixed(4)}<br />{Object.entries(x.params).map(([k, v]) => `${k}=${typeof v === 'number' ? +v.toPrecision(3) : v}`).join(', ')}</>)} onMouseLeave={hide} />
                ))}
                <text x={W - 10} y={H - 8} textAnchor="end">trial →</text>
              </svg>
              {node}
            </div>
            <div className="legend" style={{ marginTop: 6 }}>
              <span><span className="dot" style={{ background: EST_COLOR[name] }} />completed</span>
              <span><span className="dot" style={{ border: `1.5px solid ${EST_COLOR[name]}` }} />pruned</span>
              <span><span style={{ width: 16, height: 2, background: 'var(--maroon)', display: 'inline-block' }} />best so far</span>
            </div>
          </Card>
          <Card k="Winning configuration" title={t.ensemble ? `${t.ensemble.seeds}-seed ensemble` : 'Best trial'}>
            <table className="tbl"><tbody>
              {Object.entries(t.best_params).map(([k, v]) => <tr key={k}><td style={{ color: 'var(--muted)' }}>{k}</td><td className="r mono">{typeof v === 'number' ? (+v.toPrecision(3)).toString() : v}</td></tr>)}
            </tbody></table>
            {Object.keys(t.importance).length > 0 && <>
              <hr className="rule" />
              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '.14em', color: 'var(--faint)', marginBottom: 6 }}>WHAT MATTERED MOST (fANOVA)</div>
              <HBars labelWidth={110} fmt={v => pct(v)} max={1} rows={Object.entries(t.importance).slice(0, 6).map(([k, v]) => ({ key: k, label: <span className="mono">{k}</span>, value: v, color: 'var(--maroon)' }))} />
            </>}
            {t.ensemble && <p className="note" style={{ marginTop: 10 }}>Ensemble validation mean log-Q {t.ensemble.val_mean_log_q_ensemble.toFixed(3)} · fallback threshold τ = {t.ensemble.tau.toFixed(3)} · {(t.ensemble.train_seconds.reduce((a, c) => a + c, 0) / 60).toFixed(1)} min training</p>}
          </Card>
        </div>
      )}
    </section>
  )
}
