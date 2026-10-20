import { useState } from 'react'
import { EST_SHORT, fmtInt, fmtQ, pct } from '../lib'
import type { Bench, Cmp } from '../types'
import { Card, EstName, SectionHead, Seg } from './ui'

const ORDER = ['postgres_live', 'postgres_xstats', 'factorjoin', 'mscn', 'mlp', 'treelstm']
const BUCKET_ORDER = ['0 rows', '1-10', '11-1k', '1k-1M', '>1M']
const sortBuckets = (d: string, ks: string[]) =>
  d === 'true rows' ? BUCKET_ORDER.filter(k => ks.includes(k)) : [...ks].sort((a, b) => a.localeCompare(b, 'en', { numeric: true }))

const GUIDE: Record<string, string> = {
  tables: 'Error that grows with the number of tables points at join-correlation modelling: compare FactorJoin (independence across joins) with the learned models.',
  'join shape': 'Star and branching joins multiply fan-outs; if they lag chains, the join-key statistics (max fan-out, key overlap) are where to look.',
  predicates: 'Range-only errors point at the value-distribution (CDF) features; equality-only errors at rare or unseen values; mixed at correlated filters.',
  'true rows': 'Errors concentrated in "0 rows" / "1-10" mean the model overestimates empty results; in ">1M" it under-counts big fan-outs. These are the tails the training augmentation and tail loss target.',
}

/** p95 Q-error per slice: where the errors come from. */
export function Breakdown({ b }: { b: Bench }) {
  const dims = Object.keys(b.breakdown ?? {})
  const [dim, setDim] = useState(dims[0] ?? 'tables')
  if (!b.breakdown || !dims.length) return null
  const rows = b.breakdown[dim]
  const names = ORDER.filter(n => Object.values(rows).some(r => n in r))
  return (
    <section className="section">
      <SectionHead eyebrow="Diagnosis" title={<>Where the errors <em>come from</em></>}>
        The same test queries, sliced four ways. Each cell is the p95 Q-error of that estimator on that slice; the worst learned cell in each row is marked.
      </SectionHead>
      <Card k={`${fmtInt(b.n_test)} test queries`} title="p95 Q-error by slice" right={<Seg label="Slice by" value={dim} onChange={setDim} options={dims.map(d => [d, d[0].toUpperCase() + d.slice(1)] as [string, string])} />}>
        <div className="scroll-x">
          <table className="tbl num">
            <thead><tr><th scope="col">{dim}</th><th scope="col" className="r">Queries</th>{names.map(n => <th key={n} scope="col" className="r">{EST_SHORT[n] ?? n}</th>)}</tr></thead>
            <tbody>{sortBuckets(dim, Object.keys(rows)).map(k => {
              const r = rows[k]
              const learned = names.filter(n => ['mscn', 'mlp', 'treelstm'].includes(n))
              const worst = learned.reduce((w, n) => (r[n] && (!w || r[n].p95 > r[w].p95) ? n : w), '')
              return (
                <tr key={k}><th scope="row">{k}</th><td className="r">{fmtInt(r.n)}</td>
                  {names.map(n => <td key={n} className="r">{r[n] ? <span className={n === worst && r[n].p95 > 2 ? 'tag bad' : ''}>{fmtQ(r[n].p95)}</span> : '–'}</td>)}</tr>
              )
            })}</tbody>
          </table>
        </div>
        <p className="note" style={{ marginTop: 10 }}>{GUIDE[dim]}</p>
      </Card>
    </section>
  )
}

function Verdict({ c, a, bName }: { c: Cmp; a: string; bName: string }) {
  if (!c.ci95_mean) return <span className="tag">{c.verdict}</span>
  const cls = c.verdict === 'inconclusive' ? '' : 'good'
  const who = c.verdict === 'a better' ? EST_SHORT[a] ?? a : c.verdict === 'b better' ? EST_SHORT[bName] ?? bName : null
  return <span className={`tag ${cls}`}>{who ? `${who} better` : 'inconclusive'}</span>
}

const fmtD = (x: number) => (x >= 0 ? '+' : '') + x.toFixed(3)

export function CompareTable({ rows, title, k }: { rows: Record<string, Record<string, Cmp>>; title: string; k: string }) {
  return (
    <Card k={k} title={title}>
      <div className="scroll-x">
        <table className="tbl num">
          <thead><tr><th scope="col">Queries</th><th scope="col">Pair (a vs b)</th><th scope="col" className="r">Δ mean log-q</th><th scope="col" className="r">95% CI</th><th scope="col" className="r">Δ p95 log-q [CI]</th><th scope="col">Verdict</th></tr></thead>
          <tbody>{Object.entries(rows).flatMap(([slice, pairs]) => Object.entries(pairs).map(([pair, c], i) => {
            const [a, bName] = pair.split(' vs ')
            return (
              <tr key={slice + pair}>
                <th scope="row">{i === 0 ? slice : ''}</th>
                <td style={{ whiteSpace: 'nowrap' }}><EstName name={a} /> <span className="note">vs</span> <EstName name={bName} /></td>
                <td className="r">{c.mean_log_q_diff !== undefined ? fmtD(c.mean_log_q_diff) : '–'}</td>
                <td className="r">{c.ci95_mean ? `[${fmtD(c.ci95_mean[0])}, ${fmtD(c.ci95_mean[1])}]` : '–'}</td>
                <td className="r">{c.ci95_p95 ? `${fmtD(c.p95_log_diff!)} [${fmtD(c.ci95_p95[0])}, ${fmtD(c.ci95_p95[1])}]` : '–'}</td>
                <td><Verdict c={c} a={a} bName={bName} /></td>
              </tr>
            )
          }))}</tbody>
        </table>
      </div>
      <p className="note" style={{ marginTop: 10 }}>Paired bootstrap over queries (1,000 resamples): each query keeps both estimators' errors, so the CI is for the <i>difference</i>. Negative Δ = a is better. If the CI contains 0 the result is reported as inconclusive rather than as a win.</p>
    </Card>
  )
}

/** Matched comparisons + per-seed agreement. */
export function Comparisons({ b }: { b: Bench }) {
  if (!b.compare) return null
  const agree = b.seeds?.agreement as Record<string, number> | undefined
  const ood = Object.entries(b.ood).filter(([, v]) => v.compare)
  return (
    <section className="section">
      <SectionHead eyebrow="Matched comparisons" title={<>Is the difference <em>real?</em></>}>
        Same queries, paired errors, confidence intervals on the difference: overall, per slice and on every held-out split.
      </SectionHead>
      <CompareTable k="i.i.d. test set" title="TreeLSTM vs MLP vs FactorJoin" rows={b.compare} />
      <div className="grid g2" style={{ marginTop: 18 }}>
        {ood.length > 0 && <CompareTable k="Held-out splits" title="Same pairs on queries the models never saw"
          rows={Object.fromEntries(ood.map(([k, v]) => [OOD_LABEL[k] ?? k, v.compare!]))} />}
        {agree && (
          <Card k="Per-seed agreement" title="Does every seed tell the same story?">
            <table className="tbl num"><thead><tr><th scope="col">Pair (a vs b)</th><th scope="col" className="r">Seed pairs where a wins</th></tr></thead>
              <tbody>{Object.entries(agree).map(([pair, v]) => {
                const [a, bName] = pair.split(' vs ')
                return <tr key={pair}><td><EstName name={a} /> <span className="note">vs</span> <EstName name={bName} /></td>
                  <td className="r"><span className={`tag ${v === 1 || v === 0 ? 'good' : ''}`}>{pct(v, 0)}</span></td></tr>
              })}</tbody></table>
            <p className="note" style={{ marginTop: 8 }}>Each seed of a is compared with each seed of b on test mean log Q-error (FactorJoin and PostgreSQL are deterministic: one "seed"). 100% or 0% = the ranking holds for every seed; anything in between means the ranking depends on the random seed.</p>
          </Card>
        )}
      </div>
    </section>
  )
}

export const OOD_LABEL: Record<string, string> = {
  split_tmpl: 'Unseen query templates',
  split_shape: 'Unseen join sizes (trained on 1–3 tables, tested on 4–5)',
  split_join: 'Unseen join graphs',
  split_sel: 'Unseen cardinality range (tested on the bottom and top tails)',
}

/** Test-set calibration of ensemble disagreement. */
export function Calibration({ b }: { b: Bench }) {
  const c = b.calibration
  if (!c || !Object.keys(c).length) return null
  const names = ORDER.filter(n => n in c)
  const covs = c[names[0]].coverage_curve.map(x => x.coverage)
  return (
    <Card k="Uncertainty calibration (test)" title="Does seed disagreement predict error?">
      <div className="scroll-x">
        <table className="tbl num">
          <thead><tr><th scope="col">Estimator</th><th scope="col" className="r">Spearman</th>{covs.map(v => <th key={v} scope="col" className="r">{pct(v)} most sure</th>)}</tr></thead>
          <tbody>{names.map(n => <tr key={n}><td><EstName name={n} /></td><td className="r">{c[n].spearman_std_vs_error.toFixed(2)}</td>
            {c[n].coverage_curve.map(x => <td key={x.coverage} className="r">{fmtQ(x.p95_q)}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <p className="note" style={{ marginTop: 8 }}>Spearman = rank correlation between the seeds' spread and the actual error (higher = the spread is a useful warning). The other columns show p95 Q-error when only the x% of queries with the smallest spread are answered by the model; if it rises with x, the spread separates easy from hard queries. The fallback threshold τ is grid-searched on validation, never on these test numbers.</p>
    </Card>
  )
}

/** Never-trained special sets: empty / tiny results, rare or absent literals. */
export function SpecialSets({ b }: { b: Bench }) {
  if (!b.special) return null
  return (
    <>
      {Object.entries(b.special).map(([k, v]) => (
        <Card key={k} k={`${fmtInt(v.n)} queries · never trained on`} title={v.label[0].toUpperCase() + v.label.slice(1)}>
          <table className="tbl num"><thead><tr><th scope="col">Estimator</th><th scope="col" className="r">Median</th><th scope="col" className="r">p95</th><th scope="col" className="r">p99</th><th scope="col" className="r">Within 2×</th></tr></thead>
            <tbody>{ORDER.filter(n => n in v.accuracy).map(n => <tr key={n}><td><EstName name={n} /></td>
              <td className="r">{fmtQ(v.accuracy[n].overall.median)}</td><td className="r">{fmtQ(v.accuracy[n].overall.p95)}</td>
              <td className="r"><b>{fmtQ(v.accuracy[n].overall.p99)}</b></td><td className="r">{pct(v.accuracy[n].overall.within_2x, 0)}</td></tr>)}</tbody></table>
        </Card>
      ))}
    </>
  )
}
