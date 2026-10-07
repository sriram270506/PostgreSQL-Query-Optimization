import { useState, useEffect } from 'react'

const SAMPLE_QUERIES = [
  { label: 'Single-table (employees)', sql: `SELECT COUNT(*) FROM employees WHERE salary BETWEEN 60000 AND 120000` },
  { label: 'Correlated (city + salary)', sql: `SELECT COUNT(*) FROM employees WHERE city = 'Bangalore' AND salary BETWEEN 80000 AND 180000` },
  { label: '2-table join (city + tier)', sql: `SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id=p.dept_id WHERE e.city = 'Bangalore' AND p.budget_tier = 5` },
  { label: '3-table join', sql: `SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id=p.dept_id JOIN assignments a ON e.emp_id=a.emp_id WHERE e.city = 'Bangalore' AND p.budget_tier = 5 AND a.hours_per_week BETWEEN 20 AND 40` },
  { label: '4-table join (complex)', sql: `SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id=d.dept_id JOIN projects p ON e.dept_id=p.dept_id JOIN assignments a ON e.emp_id=a.emp_id JOIN budgets b ON p.proj_id=b.proj_id WHERE e.city = 'Bangalore' AND p.budget_tier = 5 AND e.salary BETWEEN 80000 AND 200000` },
  { label: 'Delhi + low budget', sql: `SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id=p.dept_id WHERE e.city = 'Delhi' AND p.budget_tier = 1` },
]

function QErrorBadge({ q }: { q: number | null }) {
  if (q == null) return <span className="badge badge-gray">—</span>
  if (q < 2) return <span className="badge badge-green">Q={q.toFixed(2)}</span>
  if (q < 10) return <span className="badge badge-amber">Q={q.toFixed(2)}</span>
  return <span className="badge badge-rose">Q={q.toFixed(2)}</span>
}

export default function FactorJoinSimulator() {
  const [sql, setSql] = useState(SAMPLE_QUERIES[2].sql)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<any[]>([])

  const runEstimate = () => {
    if (!sql.trim()) return
    setLoading(true)
    setError(null)
    fetch('/api/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sql }),
    })
      .then(r => r.json())
      .then(d => {
        setResult(d)
        setHistory(h => [{ sql, ...d, ts: new Date().toLocaleTimeString() }, ...h.slice(0, 9)])
        setLoading(false)
      })
      .catch(e => { setError(String(e)); setLoading(false) })
  }

  const loadSample = (q: typeof SAMPLE_QUERIES[0]) => {
    setSql(q.sql)
    setResult(null)
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">⚡ Estimator Lab</div>
        <div className="page-subtitle">
          Run a query and compare PostgreSQL estimate, FactorJoin, and ML Fusion — with live latency measurements.
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20, alignItems: 'start' }}>
        {/* Left: Editor + Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Sample queries */}
          <div className="card">
            <div className="card-title" style={{ marginBottom: 10 }}>Sample Queries</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {SAMPLE_QUERIES.map((q, i) => (
                <button key={i} className="btn btn-ghost btn-sm" onClick={() => loadSample(q)}>{q.label}</button>
              ))}
            </div>
          </div>

          {/* SQL Editor */}
          <div className="card">
            <div className="sql-editor-label">SQL Query</div>
            <textarea
              className="input"
              style={{ minHeight: 100 }}
              value={sql}
              onChange={e => setSql(e.target.value)}
              onKeyDown={e => { if (e.ctrlKey && e.key === 'Enter') runEstimate() }}
              spellCheck={false}
            />
            <div style={{ display: 'flex', gap: 10, marginTop: 10, alignItems: 'center' }}>
              <button className="btn btn-primary" onClick={runEstimate} disabled={loading}>
                {loading ? '…' : '⚡ Run Estimate'}
              </button>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Ctrl+Enter to run</span>
            </div>
            {error && (
              <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.3)', borderRadius: 8, color: '#fb7185', fontSize: 12 }}>
                {error}
              </div>
            )}
          </div>

          {/* Result */}
          {result && (
            <div className="card fade-in">
              <div className="card-header">
                <div className="card-title">Estimation Results</div>
                <QErrorBadge q={result.q_errors?.ml_fusion ?? result.q_errors?.factorjoin} />
              </div>

              {/* 3-way estimate comparison */}
              <div className="estimate-compare">
                <div className="est-box">
                  <div className="est-box-label">True Count</div>
                  <div className="est-box-value" style={{ color: '#94a3b8' }}>
                    {result.true_count != null ? result.true_count.toLocaleString() : '—'}
                  </div>
                  <div className="est-box-sub">Ground truth (Pandas)</div>
                </div>
                <div className="est-box" style={{ border: '1px solid rgba(59,130,246,0.3)', background: 'rgba(59,130,246,0.06)' }}>
                  <div className="est-box-label">FactorJoin</div>
                  <div className="est-box-value" style={{ color: '#3b82f6' }}>
                    {result.factorjoin_estimate != null ? Number(result.factorjoin_estimate).toLocaleString() : '—'}
                  </div>
                  <QErrorBadge q={result.q_errors?.factorjoin} />
                </div>
                <div className="est-box" style={{ border: '1px solid rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.06)' }}>
                  <div className="est-box-label">ML Fusion</div>
                  <div className="est-box-value" style={{ color: '#10b981' }}>
                    {result.ml_fusion_estimate != null ? Number(result.ml_fusion_estimate).toLocaleString() : '—'}
                  </div>
                  <QErrorBadge q={result.q_errors?.ml_fusion} />
                </div>
              </div>

              {/* Latency comparison */}
              <div style={{ marginTop: 20 }}>
                <div className="section-title">Inference Latency</div>
                <div className="latency-bar-group">
                  {[
                    { label: 'PostgreSQL (sim.)', val: result.latency?.postgres_ms, color: '#f43f5e' },
                    { label: 'FactorJoin', val: result.latency?.factorjoin_ms, color: '#3b82f6' },
                    { label: 'ML Fusion', val: result.latency?.ml_fusion_ms, color: '#10b981' },
                  ].map((l, i) => {
                    const maxVal = result.latency?.postgres_ms || 1
                    const pct = l.val != null ? Math.max((l.val / maxVal) * 100, 1) : 0
                    return (
                      <div key={i} className="latency-row">
                        <div className="latency-label">{l.label}</div>
                        <div className="latency-bar-track">
                          <div className="latency-bar-fill" style={{ width: `${pct}%`, background: l.color }} />
                        </div>
                        <div className="latency-val-out" style={{ color: l.color }}>
                          {l.val != null ? `${l.val < 1 ? l.val.toFixed(4) : l.val.toFixed(2)}ms` : '—'}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Features */}
              {result.features && (
                <div style={{ marginTop: 20 }}>
                  <div className="section-title">Extracted FactorJoin Features</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
                    {Object.entries(result.features).map(([k, v]: [string, any]) => (
                      <div key={k} style={{ padding: '8px 10px', background: 'var(--bg-glass)', borderRadius: 8, border: '1px solid var(--border)' }}>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>{k}</div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'JetBrains Mono, monospace' }}>
                          {typeof v === 'number' ? v.toFixed(4) : String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: History */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Query History</div>
          {history.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
              Run a query to see history
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {history.map((h, i) => (
                <div
                  key={i}
                  onClick={() => { setSql(h.sql); setResult(h) }}
                  style={{
                    padding: '10px 12px',
                    background: 'var(--bg-glass)',
                    border: '1px solid var(--border)',
                    borderRadius: 8,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-glass-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'var(--bg-glass)')}
                >
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{h.ts}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'JetBrains Mono', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {h.sql.slice(0, 60)}…
                  </div>
                  <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                    {h.true_count != null && <span className="badge badge-gray">T={h.true_count}</span>}
                    {h.factorjoin_estimate != null && <span className="badge badge-blue">FJ={Math.round(h.factorjoin_estimate)}</span>}
                    {h.ml_fusion_estimate != null && <span className="badge badge-green">ML={Math.round(h.ml_fusion_estimate)}</span>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
