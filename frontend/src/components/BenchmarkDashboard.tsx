import { useState, useEffect } from 'react'

const CATEGORY_COLORS: Record<string, string> = {
  'Category A': '#3b82f6',
  'Category B': '#8b5cf6',
  'Category C': '#06b6d4',
  'Category D': '#f59e0b',
  'Category E': '#f43f5e',
}

function getCatColor(cat: string) {
  for (const [k, v] of Object.entries(CATEGORY_COLORS)) {
    if (cat.includes(k)) return v
  }
  return '#3b82f6'
}

function LatencyBar({ label, val, maxVal, color }: { label: string; val: number | null; maxVal: number; color: string }) {
  const pct = val != null && maxVal > 0 ? Math.max((val / maxVal) * 100, 1) : 0
  return (
    <div className="latency-row">
      <div className="latency-label">{label}</div>
      <div className="latency-bar-track">
        <div className="latency-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="latency-val-out" style={{ color }}>
        {val != null ? `${val < 1 ? val.toFixed(4) : val.toFixed(2)}ms` : '—'}
      </div>
    </div>
  )
}

export default function BenchmarkDashboard() {
  const [bench, setBench] = useState<any>(null)
  const [queries, setQueries] = useState<any[]>([])
  const [catFilter, setCatFilter] = useState('all')
  const [queryPage, setQueryPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const PAGE_SIZE = 15

  useEffect(() => {
    Promise.all([
      fetch('/api/benchmark').then(r => r.json()),
      fetch(`/api/queries?limit=500`).then(r => r.json()),
    ]).then(([b, q]) => {
      setBench(b)
      setQueries(q.queries || [])
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  if (loading) return <div className="center-flex"><div className="loading-spin" /></div>
  if (!bench || bench.error) return (
    <div className="center-flex" style={{ color: 'var(--accent-rose)' }}>
      {bench?.error || 'Failed to load benchmark data'}
    </div>
  )

  const latencies = bench.latency_ms || {}
  const maxLat = Math.max(latencies.postgres_mean || 0, latencies.factorjoin_mean || 0, latencies.ml_fusion_mean || 0)
  const summary = bench.summary || {}
  const dist = bench.q_error_distribution || {}
  const catBreakdown = bench.category_breakdown || []
  const totalQ = bench.total_queries

  const filteredQueries = catFilter === 'all' ? queries : queries.filter(q => q.category?.includes(catFilter))
  const pagedQueries = filteredQueries.slice(queryPage * PAGE_SIZE, (queryPage + 1) * PAGE_SIZE)

  const speedup = latencies.postgres_mean && latencies.ml_fusion_mean
    ? (latencies.postgres_mean / latencies.ml_fusion_mean).toFixed(0)
    : null

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">▣ Benchmark Dashboard</div>
        <div className="page-subtitle">
          {totalQ} queries across 5 categories — PostgreSQL vs FactorJoin vs ML Fusion
        </div>
      </div>

      {/* Headline metrics */}
      <div className="stats-grid" style={{ marginBottom: 24 }}>
        {[
          { label: 'Queries Tested', value: totalQ, unit: 'across 5 categories', grad: 'var(--gradient-blue)' },
          { label: 'PG Median Q-error', value: summary.postgres_median_q?.toFixed(1) || '—', unit: 'PostgreSQL baseline', grad: 'var(--gradient-amber)' },
          { label: 'FJ Median Q-error', value: summary.factorjoin_median_q?.toFixed(2) || '—', unit: 'FactorJoin estimator', grad: 'var(--gradient-cyan)' },
          { label: 'ML Median Q-error', value: summary.ml_fusion_median_q?.toFixed(3) || '—', unit: 'Fusion MLP model', grad: 'var(--gradient-emerald)' },
        ].map((s, i) => (
          <div key={i} className="stat-card" style={{ '--accent-gradient': s.grad } as any}>
            <div className="stat-label">{s.label}</div>
            <div className="stat-value" style={{ background: s.grad, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
              {s.value}
            </div>
            <div className="stat-unit">{s.unit}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
        {/* Latency Comparison */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">⚡ Latency Comparison</div>
            {speedup && (
              <span className="badge badge-green">ML is {speedup}× faster than PG</span>
            )}
          </div>
          <div className="latency-bar-group">
            <LatencyBar label="PostgreSQL (sim.)" val={latencies.postgres_mean} maxVal={maxLat} color="#f43f5e" />
            <LatencyBar label="FactorJoin" val={latencies.factorjoin_mean} maxVal={maxLat} color="#3b82f6" />
            <LatencyBar label="ML Fusion" val={latencies.ml_fusion_mean} maxVal={maxLat} color="#10b981" />
          </div>
          <div style={{ marginTop: 16, padding: '10px 14px', background: 'rgba(16,185,129,0.07)', borderRadius: 8, border: '1px solid rgba(16,185,129,0.2)', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            <b style={{ color: '#34d399' }}>Key insight:</b> ML Fusion adds just ~{latencies.ml_fusion_mean?.toFixed(4)}ms overhead over FactorJoin features.
            PostgreSQL must parse, plan, and execute the query whereas both estimators run pure in-memory inference.
          </div>
        </div>

        {/* Q-error Distribution */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Q-error Distribution</div>
            <span className="badge badge-gray">{totalQ} queries</span>
          </div>
          <div style={{ marginBottom: 12 }}>
            <div className="section-title" style={{ marginBottom: 8 }}>PostgreSQL</div>
            <div className="q-dist-grid">
              {[
                { range: '< 2×', key: 'under_2', color: '#10b981' },
                { range: '2–10×', key: '2_to_10', color: '#f59e0b' },
                { range: '10–100×', key: '10_to_100', color: '#f43f5e' },
                { range: '> 100×', key: 'over_100', color: '#7f1d1d' },
              ].map(b => {
                const cnt = dist.postgres?.[b.key] || 0
                return (
                  <div key={b.key} className="q-dist-cell">
                    <div className="q-dist-range">{b.range}</div>
                    <div className="q-dist-count" style={{ color: b.color }}>{cnt}</div>
                    <div className="q-dist-pct">{totalQ > 0 ? ((cnt / totalQ) * 100).toFixed(0) : 0}%</div>
                  </div>
                )
              })}
            </div>
          </div>
          <div>
            <div className="section-title" style={{ marginBottom: 8 }}>FactorJoin</div>
            <div className="q-dist-grid">
              {[
                { range: '< 2×', key: 'under_2', color: '#10b981' },
                { range: '2–10×', key: '2_to_10', color: '#f59e0b' },
                { range: '10–100×', key: '10_to_100', color: '#f43f5e' },
                { range: '> 100×', key: 'over_100', color: '#7f1d1d' },
              ].map(b => {
                const cnt = dist.factorjoin?.[b.key] || 0
                return (
                  <div key={b.key} className="q-dist-cell">
                    <div className="q-dist-range">{b.range}</div>
                    <div className="q-dist-count" style={{ color: b.color }}>{cnt}</div>
                    <div className="q-dist-pct">{totalQ > 0 ? ((cnt / totalQ) * 100).toFixed(0) : 0}%</div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Per-category breakdown */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div className="card-title">Per-Category Breakdown</div>
          <span className="badge badge-gray">PostgreSQL vs FactorJoin Q-error</span>
        </div>
        {catBreakdown.map((cat: any, i: number) => {
          const color = getCatColor(cat.category)
          const maxQ = Math.max(cat.pg_median_q || 0, 1)
          const pgPct = cat.pg_median_q ? Math.min((cat.pg_median_q / maxQ) * 100, 100) : 0
          const fjPct = cat.fj_median_q ? Math.min((cat.fj_median_q / maxQ) * 100, 100) : 0
          return (
            <div key={i} style={{ padding: '14px 0', borderBottom: i < catBreakdown.length - 1 ? '1px solid var(--border)' : 'none' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
                  <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{cat.category}</span>
                  <span className="badge badge-gray">{cat.count} queries</span>
                </div>
                <div style={{ display: 'flex', gap: 12, fontSize: 12, color: 'var(--text-muted)' }}>
                  {cat.pg_latency_ms && <span>PG: {cat.pg_latency_ms.toFixed(1)}ms</span>}
                  {cat.fj_latency_ms && <span>FJ: {cat.fj_latency_ms.toFixed(3)}ms</span>}
                </div>
              </div>
              {/* PG Q-error bar */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={{ width: 80, fontSize: 11, color: 'var(--text-muted)' }}>PostgreSQL</span>
                <div className="mini-bar-track" style={{ flex: 1 }}>
                  <div className="mini-bar-fill" style={{ width: `${pgPct}%`, background: '#f43f5e' }} />
                </div>
                <span style={{ width: 50, fontSize: 11, fontWeight: 600, color: '#f43f5e', textAlign: 'right' }}>
                  {cat.pg_median_q?.toFixed(1)}×
                </span>
              </div>
              {/* FJ Q-error bar */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 80, fontSize: 11, color: 'var(--text-muted)' }}>FactorJoin</span>
                <div className="mini-bar-track" style={{ flex: 1 }}>
                  <div className="mini-bar-fill" style={{ width: `${fjPct}%`, background: '#3b82f6' }} />
                </div>
                <span style={{ width: 50, fontSize: 11, fontWeight: 600, color: '#3b82f6', textAlign: 'right' }}>
                  {cat.fj_median_q?.toFixed(2)}×
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Query workload table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">Query Workload</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {['all', 'A', 'B', 'C', 'D', 'E'].map(c => (
              <button key={c} className={`btn btn-ghost btn-sm ${catFilter === c ? 'active' : ''}`}
                style={catFilter === c ? { background: 'rgba(59,130,246,0.15)', color: '#3b82f6', borderColor: 'rgba(59,130,246,0.3)' } : {}}
                onClick={() => { setCatFilter(c); setQueryPage(0) }}>
                {c === 'all' ? 'All' : `Cat ${c}`}
              </button>
            ))}
          </div>
        </div>

        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Category</th>
                <th>SQL</th>
                <th>True Count</th>
                <th>PG Estimate</th>
                <th>FJ Estimate</th>
                <th>PG Q-err</th>
                <th>FJ Q-err</th>
                <th>FJ Latency</th>
                <th>PG Latency</th>
              </tr>
            </thead>
            <tbody>
              {pagedQueries.map((q: any) => {
                const color = getCatColor(q.category)
                const pgQ = q.postgres_q_error
                const fjQ = q.factorjoin_q_error
                const qColor = (q: number | null) => q == null ? 'var(--text-muted)' : q < 2 ? '#10b981' : q < 10 ? '#f59e0b' : '#f43f5e'
                return (
                  <tr key={q.id}>
                    <td style={{ color: 'var(--text-muted)' }}>{q.id}</td>
                    <td>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 7px', background: color + '18', border: `1px solid ${color}30`, borderRadius: 12, fontSize: 11, color, fontWeight: 600 }}>
                        {q.category.match(/Category \w/)?.[0] || q.category.slice(0, 8)}
                      </div>
                    </td>
                    <td className="mono" style={{ maxWidth: 280 }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 280, fontSize: 11 }}>{q.sql}</div>
                    </td>
                    <td className="mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{q.true_count?.toLocaleString()}</td>
                    <td className="mono" style={{ color: 'var(--text-secondary)' }}>{q.postgres_estimate?.toLocaleString()}</td>
                    <td className="mono" style={{ color: '#3b82f6' }}>{Number(q.factorjoin_estimate)?.toLocaleString()}</td>
                    <td className="mono" style={{ color: qColor(pgQ), fontWeight: 600 }}>{pgQ?.toFixed(1)}×</td>
                    <td className="mono" style={{ color: qColor(fjQ), fontWeight: 600 }}>{fjQ?.toFixed(2)}×</td>
                    <td className="mono" style={{ color: '#3b82f6' }}>{q.factorjoin_latency_ms?.toFixed(3)}ms</td>
                    <td className="mono" style={{ color: '#f43f5e' }}>{q.postgres_latency_ms?.toFixed(1)}ms</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div style={{ display: 'flex', gap: 8, padding: '12px 16px', borderTop: '1px solid var(--border)', alignItems: 'center' }}>
          <button className="btn btn-ghost btn-sm" disabled={queryPage <= 0} onClick={() => setQueryPage(p => p - 1)}>← Prev</button>
          <button className="btn btn-ghost btn-sm" disabled={(queryPage + 1) * PAGE_SIZE >= filteredQueries.length} onClick={() => setQueryPage(p => p + 1)}>Next →</button>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
            {queryPage * PAGE_SIZE + 1}–{Math.min((queryPage + 1) * PAGE_SIZE, filteredQueries.length)} of {filteredQueries.length}
          </span>
        </div>
      </div>
    </div>
  )
}
