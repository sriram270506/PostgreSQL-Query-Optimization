import { useState, useEffect, useRef } from 'react'

const TABLE_COLOR: Record<string, string> = {
  departments: '#6366f1', employees: '#3b82f6', projects: '#10b981',
  assignments: '#f59e0b', locations: '#06b6d4', budgets: '#f43f5e',
}

function BNGraphSVG({ nodes, edges, color }: { nodes: any[]; edges: any[]; color: string }) {
  if (!nodes.length) return <div className="center-flex" style={{ padding: 40 }}>No graph data</div>

  const W = 520, H = 260
  const n = nodes.length

  // Find root (no parent)
  const childSet = new Set(edges.map((e: any) => e.target))
  const roots = nodes.filter(nd => !childSet.has(nd.id))

  // Layout: radial from root
  const positions: Record<string, { x: number; y: number }> = {}

  if (n <= 1) {
    positions[nodes[0].id] = { x: W / 2, y: H / 2 }
  } else {
    // BFS layout
    const adj: Record<string, string[]> = {}
    nodes.forEach(nd => { adj[nd.id] = [] })
    edges.forEach((e: any) => { adj[e.source]?.push(e.target) })

    const root = roots[0]?.id || nodes[0].id
    const visited = new Set<string>()
    const queue: { id: string; depth: number; idx: number; siblings: number }[] = [{ id: root, depth: 0, idx: 0, siblings: 1 }]
    const levels: Record<number, string[]> = {}

    // BFS
    const bfsQ = [root]
    visited.add(root)
    const depthMap: Record<string, number> = { [root]: 0 }
    while (bfsQ.length) {
      const cur = bfsQ.shift()!
      const d = depthMap[cur]
      if (!levels[d]) levels[d] = []
      levels[d].push(cur)
      for (const child of (adj[cur] || [])) {
        if (!visited.has(child)) {
          visited.add(child)
          depthMap[child] = d + 1
          bfsQ.push(child)
        }
      }
    }

    const maxDepth = Math.max(...Object.keys(levels).map(Number))
    const xStep = maxDepth > 0 ? (W - 80) / maxDepth : W / 2
    Object.entries(levels).forEach(([d, ids]) => {
      const depth = Number(d)
      const count = ids.length
      ids.forEach((id, i) => {
        positions[id] = {
          x: 50 + depth * xStep,
          y: count === 1 ? H / 2 : 30 + (i * (H - 60)) / (count - 1),
        }
      })
    })
  }

  const alpha = color + '44'

  return (
    <svg width={W} height={H} style={{ overflow: 'visible' }}>
      {/* Edges */}
      {edges.map((e: any, i: number) => {
        const s = positions[e.source], t = positions[e.target]
        if (!s || !t) return null
        const mx = (s.x + t.x) / 2
        const my = (s.y + t.y) / 2
        return (
          <g key={i}>
            <path
              d={`M ${s.x} ${s.y} Q ${mx} ${s.y} ${t.x} ${t.y}`}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
              strokeOpacity={0.5}
              strokeDasharray="4 3"
            />
            <circle cx={mx} cy={(s.y + t.y) / 2} r={3} fill={color} opacity={0.5} />
            {/* Arrow */}
            <polygon
              points={`${t.x - 8},${t.y - 4} ${t.x},${t.y} ${t.x - 8},${t.y + 4}`}
              fill={color}
              opacity={0.6}
            />
          </g>
        )
      })}

      {/* Nodes */}
      {nodes.map((nd: any) => {
        const pos = positions[nd.id]
        if (!pos) return null
        const isRoot = !edges.some((e: any) => e.target === nd.id)
        return (
          <g key={nd.id} transform={`translate(${pos.x},${pos.y})`}>
            <rect x={-38} y={-22} width={76} height={44} rx={8}
              fill={isRoot ? color + '22' : 'rgba(255,255,255,0.04)'}
              stroke={color}
              strokeWidth={isRoot ? 1.5 : 1}
              strokeOpacity={isRoot ? 0.8 : 0.35}
            />
            <text textAnchor="middle" dy={-4} fontSize={9.5} fill={color} fontWeight={600} fontFamily="Inter, sans-serif">
              {nd.id.replace(/_/g, ' ')}
            </text>
            {nd.bins_count && (
              <text textAnchor="middle" dy={9} fontSize={8} fill={color} opacity={0.6} fontFamily="Inter, sans-serif">
                {nd.bins_count} bins
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

export default function BayesianNetworks() {
  const [bnData, setBnData] = useState<any>(null)
  const [selected, setSelected] = useState<string>('employees')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/bayesian-networks')
      .then(r => r.json())
      .then(d => { setBnData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div className="center-flex"><div className="loading-spin" /></div>
  if (!bnData) return <div className="center-flex">Failed to load BN models</div>

  const tables = Object.keys(bnData)
  const current = bnData[selected]

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">◉ Bayesian Network Explorer</div>
        <div className="page-subtitle">
          Chow-Liu maximum-spanning-tree BNs learned per table. Non-correlated fields (emp_id, name, proj_id…) are excluded.
        </div>
      </div>

      {/* Table selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tables.map(t => (
          <button
            key={t}
            className={`btn ${selected === t ? 'btn-primary' : 'btn-ghost'}`}
            style={selected === t ? { background: TABLE_COLOR[t] || '#3b82f6' } : {}}
            onClick={() => setSelected(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {current && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20 }}>
          {/* Graph */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Chow-Liu Tree — {selected}</div>
                <div className="card-subtitle">
                  {current.nodes.length} variables · {current.edges.length} dependency edges
                </div>
              </div>
              <span className="badge badge-blue">{current.cols.length} features</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <BNGraphSVG
                nodes={current.nodes}
                edges={current.edges}
                color={TABLE_COLOR[selected] || '#3b82f6'}
              />
            </div>

            {/* Edges list */}
            {current.edges.length > 0 && (
              <div style={{ marginTop: 16, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                <div className="section-title">Dependency Edges (Learned Correlations)</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {current.edges.map((e: any, i: number) => (
                    <div key={i} style={{
                      display: 'inline-flex', alignItems: 'center', gap: 6,
                      padding: '4px 10px',
                      background: (TABLE_COLOR[selected] || '#3b82f6') + '15',
                      border: `1px solid ${(TABLE_COLOR[selected] || '#3b82f6')}30`,
                      borderRadius: 20,
                      fontSize: 12,
                    }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{e.source}</span>
                      <span style={{ color: TABLE_COLOR[selected] || '#3b82f6' }}>→</span>
                      <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{e.target}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* CPT Summary */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="card">
              <div className="card-title" style={{ marginBottom: 12 }}>CPT Summary</div>
              {Object.entries(current.cpt_summary || {}).map(([col, info]: [string, any]) => (
                <div key={col} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: TABLE_COLOR[selected] || '#3b82f6' }}>{col}</span>
                    <span className={`badge ${info.type === 'prior' ? 'badge-purple' : 'badge-blue'}`}>{info.type}</span>
                  </div>
                  {info.type === 'prior' ? (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {Object.entries(info.probs || {}).slice(0, 4).map(([k, v]: [string, any]) => (
                        <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis' }}>{k}</span>
                          <span style={{ fontWeight: 600 }}>{(v * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      <div>Parent: <b style={{ color: 'var(--text-secondary)' }}>{info.parent}</b></div>
                      <div>Parent states: <b style={{ color: 'var(--text-secondary)' }}>{info.n_parent_states}</b></div>
                      {info.sample_probs && (
                        <div style={{ marginTop: 4, padding: '6px 8px', background: 'var(--bg-glass)', borderRadius: 6 }}>
                          <div style={{ marginBottom: 3, color: 'var(--text-muted)', fontSize: 10 }}>
                            P({col} | {info.parent}={info.sample_parent_val?.slice(0, 20)}):
                          </div>
                          {Object.entries(info.sample_probs).slice(0, 3).map(([cv, p]: [string, any]) => (
                            <div key={cv} style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span style={{ color: 'var(--text-secondary)', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }}>{cv}</span>
                              <span style={{ fontWeight: 600 }}>{(p * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
