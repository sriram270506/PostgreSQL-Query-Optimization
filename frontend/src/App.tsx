import { useState, useEffect } from 'react'
import './index.css'
import DataExplorer from './components/DataExplorer'
import BayesianNetworks from './components/BayesianNetworks'
import CPTInspector from './components/CPTInspector'
import FactorJoinSimulator from './components/FactorJoinSimulator'
import BenchmarkDashboard from './components/BenchmarkDashboard'

type Tab = 'overview' | 'data' | 'bayesian' | 'cpts' | 'simulator' | 'benchmark'

const NAV = [
  { id: 'overview', label: 'Overview', icon: '◈' },
  { id: 'data', label: 'Data Explorer', icon: '⊞' },
  { id: 'bayesian', label: 'Bayesian Networks', icon: '◉' },
  { id: 'cpts', label: 'CPT Inspector', icon: '≡' },
  { id: 'simulator', label: 'Estimator Lab', icon: '⚡' },
  { id: 'benchmark', label: 'Benchmark', icon: '▣' },
] as const

export default function App() {
  const [tab, setTab] = useState<Tab>('overview')
  const [health, setHealth] = useState<any>(null)

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(setHealth)
      .catch(() => {})
  }, [])

  const stats = health?.fusion_stats || {}

  return (
    <div className="app-shell">
      {/* ─── Sidebar ─── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-title">Cardinality<br />Research Platform</div>
          <div className="sidebar-logo-sub">PostgreSQL Optimization</div>
        </div>

        <div className="sidebar-section-label">Navigation</div>
        {NAV.map(n => (
          <div
            key={n.id}
            className={`nav-item ${tab === n.id ? 'active' : ''}`}
            onClick={() => setTab(n.id as Tab)}
          >
            <span style={{ fontSize: 16 }}>{n.icon}</span>
            <span>{n.label}</span>
          </div>
        ))}

        <div className="sidebar-footer">
          <div className="status-badge">
            <div className="status-dot" />
            <span>{health ? 'API Online' : 'Connecting…'}</span>
          </div>
          {health && (
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
              BN Tables: {health.bn_tables?.length || 0}/6
            </div>
          )}
        </div>
      </aside>

      {/* ─── Main Content ─── */}
      <main className="main-content">
        <div className="page-inner">
          {tab === 'overview' && <OverviewPage health={health} stats={stats} setTab={setTab} />}
          {tab === 'data' && <DataExplorer />}
          {tab === 'bayesian' && <BayesianNetworks />}
          {tab === 'cpts' && <CPTInspector />}
          {tab === 'simulator' && <FactorJoinSimulator />}
          {tab === 'benchmark' && <BenchmarkDashboard />}
        </div>
      </main>
    </div>
  )
}

function OverviewPage({ health, stats, setTab }: { health: any; stats: any; setTab: (t: Tab) => void }) {
  const cards = [
    { label: 'Tables', value: '6', unit: 'synthetic tables', color: 'var(--gradient-blue)', accentGrad: 'var(--gradient-blue)' },
    { label: 'Total Rows', value: '20.1K', unit: 'correlated records', color: 'var(--gradient-cyan)', accentGrad: 'var(--gradient-cyan)' },
    { label: 'Query Workload', value: '500', unit: 'queries across 5 categories', color: 'var(--gradient-purple)', accentGrad: 'var(--gradient-purple)' },
    { label: 'FJ Median Q-error', value: stats.median_q ? stats.median_q.toFixed(2) : '—', unit: 'ML Fusion model', color: 'var(--gradient-emerald)', accentGrad: 'var(--gradient-emerald)' },
  ]

  const latencies = [
    { label: 'PostgreSQL (sim.)', val: stats.pg_latency_ms, unit: 'ms', color: '#f43f5e', pct: 100 },
    { label: 'FactorJoin', val: stats.fj_latency_ms, unit: 'ms', color: '#3b82f6', pct: stats.pg_latency_ms ? (stats.fj_latency_ms / stats.pg_latency_ms) * 100 : 0 },
    { label: 'ML Fusion', val: stats.ml_latency_ms, unit: 'ms', color: '#10b981', pct: stats.pg_latency_ms ? (stats.ml_latency_ms / stats.pg_latency_ms) * 100 : 0 },
  ]

  const features = [
    { icon: '◈', title: 'Chow-Liu Trees', desc: 'Bayesian Networks learned from data using maximum spanning tree on mutual information, excluding non-correlated IDs and name fields.' },
    { icon: '⊞', title: '6-Table Schema', desc: 'Departments → Employees → Projects → Assignments → Locations → Budgets with injected same-table and cross-table correlations.' },
    { icon: '⚡', title: 'FactorJoin Estimator', desc: 'Per-table selectivity × conditional probability bridges computed using 2D contingency tables across foreign-key joins.' },
    { icon: '▣', title: 'ML Fusion MLP', desc: 'Sklearn MLP trained on FactorJoin features (128→64→32) with log-transformed targets for cardinality estimation.' },
  ]

  return (
    <div className="fade-in">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <div style={{
            background: 'var(--gradient-blue)',
            width: 40, height: 40,
            borderRadius: 10,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 20
          }}>◈</div>
          <div>
            <div className="page-title">PostgreSQL Cardinality Research Platform</div>
            <div className="page-subtitle">Bayesian Networks · FactorJoin · ML Fusion · Real-time Latency Benchmarking</div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        {cards.map((c, i) => (
          <div key={i} className="stat-card" style={{ '--accent-gradient': c.accentGrad } as any}>
            <div className="stat-label">{c.label}</div>
            <div className="stat-value" style={{ background: c.color, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
              {c.value}
            </div>
            <div className="stat-unit">{c.unit}</div>
          </div>
        ))}
      </div>

      {/* Two column: Latency + Arch */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Latency Comparison */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">⚡ Latency Comparison</div>
              <div className="card-subtitle">Mean per-query inference time</div>
            </div>
          </div>
          <div className="latency-bar-group">
            {latencies.map((l, i) => (
              <div key={i} className="latency-row">
                <div className="latency-label">{l.label}</div>
                <div className="latency-bar-track">
                  <div
                    className="latency-bar-fill"
                    style={{ width: `${Math.max(l.pct, 2)}%`, background: l.color }}
                  />
                </div>
                <div className="latency-val-out" style={{ color: l.color }}>
                  {l.val != null ? `${l.val < 1 ? l.val.toFixed(4) : l.val.toFixed(2)}ms` : '—'}
                </div>
              </div>
            ))}
          </div>
          {stats.pg_latency_ms && stats.ml_latency_ms && (
            <div style={{ marginTop: 16, padding: '10px 14px', background: 'rgba(16,185,129,0.08)', borderRadius: 8, border: '1px solid rgba(16,185,129,0.2)' }}>
              <span style={{ fontSize: 12, color: '#34d399', fontWeight: 600 }}>
                ML Fusion is {(stats.pg_latency_ms / stats.ml_latency_ms).toFixed(0)}× faster than PostgreSQL
              </span>
            </div>
          )}
        </div>

        {/* Model Performance */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">▣ Model Performance</div>
              <div className="card-subtitle">Q-error metrics (lower = better)</div>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[
              { name: 'PostgreSQL Baseline', q: 250, color: '#f43f5e', note: 'Extended statistics fail on 4+ table joins' },
              { name: 'FactorJoin', q: null, color: '#3b82f6', note: 'Chow-Liu BN + conditional probability bridges' },
              { name: 'ML Fusion MLP', q: stats.median_q, color: '#10b981', note: '128-64-32 MLP, log-space targets' },
            ].map((m, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: 'var(--bg-glass)', borderRadius: 8, border: '1px solid var(--border)' }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: m.color }}>{m.name}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{m.note}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: m.color }}>
                    {m.q != null ? m.q.toFixed(2) : '—'}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Median Q-err</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="grid-auto" style={{ marginBottom: 24 }}>
        {features.map((f, i) => (
          <div key={i} className="card" style={{ cursor: 'default' }}>
            <div style={{ fontSize: 24, marginBottom: 10 }}>{f.icon}</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>{f.title}</div>
            <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{f.desc}</div>
          </div>
        ))}
      </div>

      {/* Quick nav */}
      <div className="card">
        <div className="card-header"><div className="card-title">Quick Navigation</div></div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {NAV.filter(n => n.id !== 'overview').map(n => (
            <button key={n.id} className="btn btn-ghost" onClick={() => setTab(n.id as Tab)}>
              <span>{n.icon}</span> {n.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
