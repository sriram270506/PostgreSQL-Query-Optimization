import { useEffect, useState } from 'react'
import { ArrowRight, Database, GitBranch, Network, Sigma } from 'lucide-react'
import { api, fmtInt, fmtQ, pct } from '../lib'
import type { Bench } from '../types'
import { Card, EstName, SectionHead, Skeleton } from '../components/ui'

const STEPS = [
  { icon: Database, t: 'Embedded SQL engine', d: 'Parses SQL, runs it exactly in memory and emulates the PostgreSQL planner’s row estimates. No server needed.' },
  { icon: Sigma, t: 'Bayesian networks', d: 'One Chow-Liu tree per table, learned with MLE. Captures column correlations such as city ↔ department.' },
  { icon: GitBranch, t: 'FactorJoin', d: 'Passes per-key-bin row counts between tables to estimate joins without running them.' },
  { icon: Network, t: 'TreeLSTM + MLP', d: 'Reads the query tree, combines it with the statistics and learns a correction on top of FactorJoin.' },
]

export default function Overview() {
  const [b, setB] = useState<Bench | null>(null)
  const [err, setErr] = useState('')
  useEffect(() => { api<Bench>('/benchmark').then(setB).catch(e => setErr(e.message)) }, [])
  const best = 'treelstm'
  const a = b?.accuracy
  return (
    <>
      <section style={{ paddingTop: 56 }}>
        <div className="grid g-side" style={{ alignItems: 'end', gap: 40 }}>
          <div>
            <div className="eyebrow">Cardinality estimation · live</div>
            <h1 className="display">Count the rows <em>before</em><br />you read them.</h1>
            <p className="lede">A query optimizer picks a join order based on how many rows it expects at each step.
              We compare four estimators on the same SQL engine: PostgreSQL’s planner, FactorJoin, a fusion MLP and a TreeLSTM.
              Every query also runs exactly, so each estimate is checked against the true count.</p>
            <div style={{ display: 'flex', gap: 12, marginTop: 26, flexWrap: 'wrap' }}>
              <a className="btn" href="#/lab">Open the Query Lab <ArrowRight size={16} /></a>
              <a className="btn ghost" href="#/internals">See inside the models</a>
            </div>
          </div>
          <Card k="Held-out test set" title={b ? `${fmtInt(b.n_test)} unseen queries` : 'Loading…'}>
            {!b && !err && <Skeleton h={150} />}
            {err && <p className="note">Benchmark not available yet ({err}). Run <span className="mono">python -m scripts.benchmark</span>.</p>}
            {a && b.estimators.map(n => (
              <div key={n} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '9px 0', borderBottom: '1px solid var(--line)' }}>
                <EstName name={n} />
                <span className="num" style={{ fontSize: 13 }}>
                  <span style={{ color: 'var(--muted)' }}>p95</span> <b>{fmtQ(a[n].overall.p95)}</b>
                  <span style={{ color: 'var(--muted)', marginLeft: 12 }}>p99</span> <b>{fmtQ(a[n].overall.p99)}</b>
                </span>
              </div>
            ))}
            {a && <p className="note" style={{ marginTop: 12 }}>Q-error = how many times off the estimate is (1× = exact). Lower is better.</p>}
          </Card>
        </div>
      </section>

      {a && b && a[best] && (
        <section className="section">
          <div className="stats">
            <div className="stat"><div className="v num">{fmtQ(a[best].overall.p99)}</div><div className="l">TreeLSTM p99 Q-error</div>
              <div className="d">vs {fmtQ(a.postgres.overall.p99)} for PostgreSQL ({(a.postgres.overall.p99 / a[best].overall.p99).toFixed(1)}× lower)</div></div>
            <div className="stat"><div className="v num">{fmtQ(a[best].overall.max)}</div><div className="l">Worst-case Q-error</div>
              <div className="d">PostgreSQL’s worst: {fmtQ(a.postgres.overall.max)}</div></div>
            <div className="stat"><div className="v num">{pct(b.plans[best].optimal_rate, 1)}</div><div className="l">Optimal join orders</div>
              <div className="d">PostgreSQL: {pct(b.plans.postgres.optimal_rate, 1)} on 3+ table queries</div></div>
            <div className="stat"><div className="v num">{fmtQ(a[best].correlated.p99)}</div><div className="l">p99 on correlated filters</div>
              <div className="d">PostgreSQL: {fmtQ(a.postgres.correlated.p99)} on the same {a[best].correlated.n} queries</div></div>
          </div>
        </section>
      )}

      <section className="section">
        <SectionHead eyebrow="Pipeline" title={<>Four stages, <em>one engine</em></>}>
          Each stage passes its result to the next one. The final MLP adds a correction to the FactorJoin estimate, so it never starts from scratch.
        </SectionHead>
        <div className="grid g4">
          {STEPS.map((s, i) => (
            <div key={s.t} className="card" style={{ position: 'relative' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ width: 40, height: 40, borderRadius: '50%', border: '1px solid var(--maroon-line)', display: 'grid', placeItems: 'center', color: 'var(--maroon)' }}><s.icon size={18} aria-hidden /></span>
                <span style={{ font: '500 22px var(--serif)', color: 'var(--maroon-line)' }}>0{i + 1}</span>
              </div>
              <h3 className="sub" style={{ marginTop: 16 }}>{s.t}</h3>
              <p className="note">{s.d}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <SectionHead eyebrow="Why it matters" title={<>Bad counts make <em>bad plans</em></>} />
        <div className="grid g3">
          <Card k="Problem"><p className="note" style={{ color: 'var(--ink-2)', fontSize: 14.5 }}>PostgreSQL assumes columns are independent. On correlated data, such as Engineering staff being mostly in Bangalore, it can be off by 1000× and then choose a slow join order.</p></Card>
          <Card k="Our approach"><p className="note" style={{ color: 'var(--ink-2)', fontSize: 14.5 }}>Learn how columns depend on each other (Bayesian networks), combine tables over join keys (FactorJoin), then let a neural network learn the errors that remain.</p></Card>
          <Card k="How we check it"><p className="note" style={{ color: 'var(--ink-2)', fontSize: 14.5 }}>8,000 generated queries with exact true counts from our own engine. We report Q-error and the true cost of the join order each estimator leads to.</p></Card>
        </div>
      </section>
    </>
  )
}
