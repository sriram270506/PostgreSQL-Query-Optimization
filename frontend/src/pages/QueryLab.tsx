import { useEffect, useRef, useState } from 'react'
import { Play, Shuffle, Trophy } from 'lucide-react'
import { api, isAbort, EST_COLOR, EST_SHORT, fmtCompact, fmtInt, fmtMs, fmtQ, getDataset } from '../lib'
import type { RunResult } from '../types'
import { Card, ErrorBox, EstName, Seg, Skeleton } from '../components/ui'
import { PlanTree } from '../components/charts'

const STARTERS_STATS = [
  "SELECT COUNT(*) FROM comments c, posts p, postHistory ph, votes v, users u WHERE v.UserId = u.Id AND c.UserId = u.Id AND p.OwnerUserId = u.Id AND ph.UserId = u.Id AND p.ViewCount >= 0 AND p.AnswerCount >= 0 AND p.AnswerCount <= 5 AND ph.PostHistoryTypeId = 2 AND v.BountyAmount >= 0 AND v.BountyAmount <= 100 AND u.Views >= 0 AND u.Views <= 13",
  "SELECT COUNT(*) FROM posts p, postLinks pl WHERE p.Id = pl.RelatedPostId AND p.AnswerCount >= 0 AND p.CreationDate >= '2010-09-06 00:58:21'::timestamp",
  "SELECT u.Id, u.Reputation, b.Date FROM users u JOIN badges b ON b.UserId = u.Id WHERE u.Reputation > 5000 AND b.Date >= '2014-01-01 00:00:00'::timestamp LIMIT 50",
]
const STARTERS_HR = [
  "SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = 'Engineering' AND e.city = 'Bangalore'",
  "SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id JOIN assignments a ON e.emp_id = a.emp_id JOIN budgets b ON p.proj_id = b.proj_id WHERE e.city = 'Bangalore' AND p.budget_tier = 5 AND e.salary BETWEEN 82009 AND 142009",
  "SELECT e.name, e.city, p.priority, a.hours_per_week FROM employees e JOIN assignments a ON e.emp_id = a.emp_id JOIN projects p ON p.proj_id = a.proj_id WHERE p.priority = 'Critical' AND e.age < 30 LIMIT 50",
]
const CATS = ['1-table', '2-way join', '3-way join', '4+ way join']
const starters = () => (getDataset() === 'stats' ? STARTERS_STATS : STARTERS_HR)

export default function QueryLab() {
  const [sql, setSql] = useState(starters()[0])
  const [res, setRes] = useState<RunResult | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [samples, setSamples] = useState<{ sql: string; category: string }[]>([])
  const [cat, setCat] = useState('3-way join')

  const inflight = useRef<AbortController | null>(null)
  // only the latest request may update the page: a slow earlier query never overwrites a newer one
  const run = async (s = sql) => {
    inflight.current?.abort()
    const ac = new AbortController()
    inflight.current = ac
    setBusy(true); setErr('')
    try { setRes(await api<RunResult>('/run', { sql: s, row_limit: 100 }, ac.signal)) }
    catch (e) { if (isAbort(e)) return; setErr((e as Error).message); setRes(null) }
    finally { if (inflight.current === ac) setBusy(false) }
  }
  const loadSamples = (c = cat) => api<{ sql: string; category: string }[]>(`/workload/sample?category=${encodeURIComponent(c)}&n=4`).then(setSamples).catch(() => setSamples([]))
  useEffect(() => { run(); loadSamples() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const maxLogQ = res ? Math.max(...res.estimators.map(e => Math.log10(e.qerror)), 0.3) : 1
  return (
    <>
      <section style={{ paddingTop: 48 }}>
        <div className="eyebrow">Query Lab</div>
        <h1 className="display" style={{ fontSize: 'clamp(36px,4.4vw,54px)' }}>Run any query. <em>Score every estimator.</em></h1>
        <p className="lede">The query runs exactly on the embedded engine. At the same time each estimator predicts the row count, and is scored against the true count.</p>
      </section>

      <section className="section" style={{ paddingTop: 28 }}>
        <div className="grid g-side">
          <Card k="SQL" right={<span className="note">SELECT · JOIN … ON · WHERE (=, &lt;&gt;, &lt;, &gt;, BETWEEN, IN) · LIMIT</span>}>
            <label htmlFor="sql" className="sr-only" style={{ position: 'absolute', left: -9999 }}>SQL query</label>
            <textarea id="sql" className="editor" spellCheck={false} value={sql} onChange={e => setSql(e.target.value)}
              onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run() }} />
            <div style={{ display: 'flex', gap: 10, marginTop: 14, alignItems: 'center', flexWrap: 'wrap' }}>
              <button className="btn" onClick={() => run()} disabled={busy}><Play size={15} aria-hidden />{busy ? 'Running…' : 'Run query'}</button>
              <span className="note">Ctrl + Enter</span>
              <div className="chips" style={{ marginLeft: 'auto' }}>
                {starters().map((s, i) => <button key={i} className="chip" onClick={() => { setSql(s); run(s) }}>Example {i + 1}</button>)}
              </div>
            </div>
            {err && <div style={{ marginTop: 14 }}><ErrorBox error={err} onRetry={() => run()} /></div>}
          </Card>
          <Card k="From the test workload" right={<button className="chip" onClick={() => loadSamples()} aria-label="New samples"><Shuffle size={13} aria-hidden /></button>}>
            <div className="chips" style={{ marginBottom: 12 }}>
              {CATS.map(c => <button key={c} className={`chip ${c === cat ? 'on' : ''}`} onClick={() => { setCat(c); loadSamples(c) }}>{c}</button>)}
            </div>
            {samples.map((s, i) => (
              <button key={i} onClick={() => { setSql(s.sql); run(s.sql) }}
                style={{ display: 'block', width: '100%', textAlign: 'left', border: 0, borderTop: '1px solid var(--line)', background: 'transparent', padding: '10px 2px', color: 'var(--ink-2)' }}>
                <span className="mono" style={{ fontSize: 11.5, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>{s.sql}</span>
              </button>
            ))}
          </Card>
        </div>
      </section>

      {busy && !res && <section className="section"><Skeleton h={260} /></section>}
      {res && (
        <div aria-busy={busy} style={{ opacity: busy ? 0.45 : 1, transition: 'opacity .15s' }}>
          <section className="section" style={{ paddingTop: 36 }}>
            <div className="stats">
              <div className="stat"><div className="v num">{fmtInt(res.count)}</div><div className="l">True rows (exact)</div><div className="d">counted by the engine</div></div>
              <div className="stat"><div className="v num">{fmtMs(res.count_ms).replace(' ms', '')}<small>ms</small></div><div className="l">Exact count time</div><div className="d">{res.rows ? `+ ${fmtMs(res.fetch_ms)} to fetch rows` : 'COUNT(*) query'}</div></div>
              <div className="stat"><div className="v" style={{ fontSize: 24, paddingTop: 6 }}><EstName name={res.best} /></div><div className="l">Closest on this query</div>
                <div className="d">{fmtQ(res.estimators.find(e => e.name === res.best)!.qerror)} off · scored after execution; one query says little, see Benchmark for p95/p99</div></div>
              <div className="stat"><div className="v num">{res.scan.length}</div><div className="l">Tables</div><div className="d">{res.scan.reduce((n, s) => n + s.preds.length, 0)} filter conditions</div></div>
            </div>
          </section>

          <section className="section" style={{ paddingTop: 22 }}>
            <Card k="Shoot-out" title="Estimated vs true rows">
              <div className="scroll-x">
                <table className="tbl">
                  <thead><tr><th>Estimator</th><th className="r" title="Made at planning time, before the query runs">Estimate (pre-run)</th><th className="r" title="Exact count from running the query">True (post-run)</th><th style={{ width: '34%' }}>Q-error (log scale)</th><th className="r">Latency</th><th /></tr></thead>
                  <tbody>
                    {res.estimators.map(e => (
                      <tr key={e.name}>
                        <td><EstName name={e.name} /></td>
                        <td className="r num">{fmtInt(e.estimate)}</td>
                        <td className="r num" style={{ color: 'var(--muted)' }}>{fmtInt(res.count)}</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <div className="bar-track" style={{ flex: 1 }}><div className="bar-fill" style={{ width: `${Math.max(2, Math.log10(e.qerror) / maxLogQ * 100)}%`, background: EST_COLOR[e.name] }} /></div>
                            <b className="num" style={{ minWidth: 58, textAlign: 'right' }}>{fmtQ(e.qerror)}</b>
                          </div>
                        </td>
                        <td className="r num">{fmtMs(e.latency_ms)}</td>
                        <td style={{ whiteSpace: 'nowrap' }}>{e.name === res.best ? <span className="tag win"><Trophy size={12} aria-hidden />closest</span>
                          : <span className="tag">{e.under ? 'under' : 'over'}</span>}
                          {e.fallback && <span className="tag" title="Ensemble seeds disagreed beyond the calibrated threshold, so FactorJoin's estimate was used" style={{ marginLeft: 4 }}>fallback</span>}
                          {e.clamped && <span className="tag bad" title="Raw estimate exceeded the provable upper bound and was clamped" style={{ marginLeft: 4 }}>clamped</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note" style={{ marginTop: 10 }}>Estimates are what each estimator would hand the planner <b>before</b> the query runs; the true count only exists afterwards, so "closest" is a retrospective score, not something a planner could know.
                Provable upper bound for this query: {fmtCompact(res.upper_bound)} rows. No estimate is allowed above it.
                Learned models are {res.estimators.find(e => e.std !== undefined && e.std > 0) ? 'multi-seed ensembles; "fallback" means the seeds disagreed and FactorJoin answered' : 'served as ensembles'}.</p>
              {res.postgres_live_error && <p className="note" style={{ marginTop: 6 }}>Live PostgreSQL unavailable: {res.postgres_live_error}</p>}
            </Card>
          </section>

          <section className="section" style={{ paddingTop: 22 }}>
            <div className="grid g2">
              <Card k="Per-table filters" title="Rows left after each table’s WHERE conditions">
                <div className="scroll-x"><table className="tbl">
                  <thead><tr><th>Table</th><th className="r">Actual</th><th className="r">PostgreSQL</th><th className="r">BN</th></tr></thead>
                  <tbody>{res.scan.map(s => (
                    <tr key={s.alias}>
                      <td><b>{s.table}</b> <span className="note">({s.alias})</span>
                        {s.preds.map(p => <div key={p} className="mono" style={{ color: 'var(--muted)', fontSize: 11 }}>{p}</div>)}</td>
                      <td className="r num"><b>{fmtInt(s.actual)}</b><div className="note">of {fmtCompact(s.total)}</div></td>
                      <td className="r num">{s.pg == null ? '–' : fmtInt(s.pg)}</td>
                      <td className="r num">{fmtInt(s.factorjoin)}</td>
                    </tr>))}</tbody>
                </table></div>
              </Card>
              {res.plans ? <Plans plans={res.plans} /> : (
                <Card k="Join order" title="Plan quality">
                  <p className="note">Join-order comparison needs 3 to 6 tables. Try one of the examples above.</p>
                </Card>
              )}
            </div>
          </section>

          {res.rows && (
            <section className="section" style={{ paddingTop: 22 }}>
              <Card k="Result rows" title={`${fmtInt(res.rows.shown)} of ${fmtInt(res.count)} rows`} className="flush">
                <div className="scroll-x" style={{ maxHeight: 420, overflowY: 'auto', padding: '0 10px 10px' }}>
                  <table className="tbl mono">
                    <thead><tr>{res.rows.columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
                    <tbody>{res.rows.data.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{String(v ?? 'NULL')}</td>)}</tr>)}</tbody>
                  </table>
                </div>
              </Card>
            </section>
          )}
        </div>
      )}
    </>
  )
}

function Plans({ plans }: { plans: NonNullable<RunResult['plans']> }) {
  const names = Object.keys(plans.estimators)
  const [sel, setSel] = useState(names[names.length - 1])
  const p = plans.estimators[sel]
  return (
    <Card k="Join order" title="Join order chosen with each estimator, scored by true cost">
      <div style={{ marginBottom: 12 }}>
        {names.map(n => {
          const r = plans.estimators[n].ratio
          return (
            <div key={n} className="bar-row" style={{ gridTemplateColumns: '150px 1fr 90px' }}>
              <EstName name={n} />
              <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.min(100, 100 / r)}%`, background: EST_COLOR[n] }} /></div>
              <span className={`tag ${r < 1.0001 ? 'good' : r > 1.5 ? 'bad' : ''}`}>{r < 1.0001 ? 'optimal' : `${r.toFixed(2)}× cost`}</span>
            </div>
          )
        })}
      </div>
      <Seg value={sel} options={names.map(n => [n, EST_SHORT[n] ?? n] as [string, string])} onChange={setSel} />
      <div style={{ marginTop: 12 }}><PlanTree plan={p.plan} color={EST_COLOR[sel]} /></div>
      <p className="note">Cost (C<sub>out</sub>) = total rows produced by all intermediate joins. This plan: {fmtCompact(p.true_cost)} rows. Best possible: {fmtCompact(plans.optimal_cost)} rows.</p>
    </Card>
  )
}
