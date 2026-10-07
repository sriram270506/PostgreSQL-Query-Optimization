import { useState, useEffect } from 'react'

const TABLE_COLOR: Record<string, string> = {
  departments: '#6366f1', employees: '#3b82f6', projects: '#10b981',
  assignments: '#f59e0b', locations: '#06b6d4', budgets: '#f43f5e',
}

const ALL_TABLES = ['departments', 'employees', 'projects', 'assignments', 'locations', 'budgets']

export default function CPTInspector() {
  const [table, setTable] = useState('employees')
  const [cptData, setCptData] = useState<any>(null)
  const [selectedCol, setSelectedCol] = useState<string | null>(null)
  const [selectedParentVal, setSelectedParentVal] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setCptData(null)
    setSelectedCol(null)
    setSelectedParentVal(null)
    fetch(`/api/cpts/${table}`)
      .then(r => r.json())
      .then(d => {
        setCptData(d)
        const firstCol = Object.keys(d.cpts || {})[0]
        if (firstCol) setSelectedCol(firstCol)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [table])

  useEffect(() => {
    setSelectedParentVal(null)
  }, [selectedCol])

  const color = TABLE_COLOR[table] || '#3b82f6'
  const cpt = cptData?.cpts?.[selectedCol || '']
  const parentVals = cpt?.type === 'conditional' ? Object.keys(cpt.data || {}) : []

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">≡ CPT Inspector</div>
        <div className="page-subtitle">
          Conditional Probability Tables for each Chow-Liu tree node. Select table → variable → parent state.
        </div>
      </div>

      {/* Table selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {ALL_TABLES.map(t => (
          <button
            key={t}
            className={`btn ${table === t ? 'btn-primary' : 'btn-ghost'}`}
            style={table === t ? { background: TABLE_COLOR[t] } : {}}
            onClick={() => setTable(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {loading && <div className="center-flex"><div className="loading-spin" /></div>}

      {!loading && cptData && (
        <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 20, alignItems: 'start' }}>
          {/* Variable List */}
          <div className="card" style={{ padding: '12px 8px' }}>
            <div className="card-title" style={{ padding: '0 8px', marginBottom: 10 }}>Variables</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {Object.entries(cptData.cpts || {}).map(([col, info]: [string, any]) => (
                <div
                  key={col}
                  onClick={() => setSelectedCol(col)}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 8,
                    cursor: 'pointer',
                    background: selectedCol === col ? color + '18' : 'transparent',
                    border: `1px solid ${selectedCol === col ? color + '40' : 'transparent'}`,
                    transition: 'all 0.15s',
                  }}
                >
                  <div style={{ fontSize: 13, fontWeight: 500, color: selectedCol === col ? color : 'var(--text-secondary)' }}>
                    {col}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    {info.type === 'conditional' ? `P(${col} | ${info.parent})` : `P(${col})`}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* CPT Display */}
          <div>
            {selectedCol && cpt && (
              <div className="card fade-in">
                {/* Header */}
                <div className="card-header">
                  <div>
                    <div className="card-title" style={{ color }}>
                      {cpt.type === 'conditional' ? `P(${selectedCol} | ${cpt.parent})` : `P(${selectedCol})`}
                    </div>
                    <div className="card-subtitle">
                      {cpt.type === 'prior' ? 'Root node — marginal prior distribution' : `Conditional on parent: ${cpt.parent}`}
                    </div>
                  </div>
                  <span className={`badge ${cpt.type === 'conditional' ? 'badge-blue' : 'badge-purple'}`}>
                    {cpt.type}
                  </span>
                </div>

                {/* Prior */}
                {cpt.type === 'prior' && (
                  <div>
                    <div className="section-title">Marginal Distribution</div>
                    <div className="data-table-wrap">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>{selectedCol}</th>
                            <th>P(value)</th>
                            <th>Probability Bar</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(cpt.data || {})
                            .sort((a, b) => (b[1] as number) - (a[1] as number))
                            .map(([val, prob]: [string, any]) => (
                              <tr key={val}>
                                <td style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{val.slice(0, 40)}</td>
                                <td className="mono" style={{ color }}>{(prob * 100).toFixed(2)}%</td>
                                <td>
                                  <div style={{ width: 160, height: 10, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                                    <div style={{ width: `${prob * 100}%`, height: '100%', background: color, borderRadius: 4 }} />
                                  </div>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Conditional */}
                {cpt.type === 'conditional' && (
                  <div>
                    {/* Parent value selector */}
                    <div style={{ marginBottom: 16 }}>
                      <div className="section-title">Select Parent Value (P({selectedCol} | {cpt.parent} = ?)</div>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        {parentVals.slice(0, 20).map(pv => (
                          <button
                            key={pv}
                            className="btn btn-ghost btn-sm"
                            style={selectedParentVal === pv ? { background: color + '20', borderColor: color + '50', color } : {}}
                            onClick={() => setSelectedParentVal(pv === selectedParentVal ? null : pv)}
                          >
                            {pv.slice(0, 30)}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* CPT Table */}
                    <div className="data-table-wrap">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>{cpt.parent} (condition)</th>
                            {selectedParentVal
                              ? Object.keys(cpt.data[selectedParentVal] || {}).map(cv => <th key={cv}>{cv.slice(0, 25)}</th>)
                              : <th>{selectedCol} (sampled)</th>
                            }
                          </tr>
                        </thead>
                        <tbody>
                          {(selectedParentVal ? [selectedParentVal] : parentVals.slice(0, 15)).map(pv => {
                            const row = cpt.data[pv] || {}
                            const childVals = Object.entries(row).sort((a, b) => (b[1] as number) - (a[1] as number))
                            return (
                              <tr key={pv}>
                                <td style={{ color: 'var(--text-primary)', fontWeight: 600, maxWidth: 200 }}>
                                  <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 180 }}>{pv}</div>
                                </td>
                                {selectedParentVal
                                  ? childVals.map(([cv, p]: [string, any]) => (
                                    <td key={cv}>
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                        <div style={{ width: `${p * 100}%`, height: 4, background: color, borderRadius: 2, minWidth: 2 }} />
                                        <span className="mono" style={{ color }}>{(p * 100).toFixed(1)}%</span>
                                      </div>
                                    </td>
                                  ))
                                  : <td>
                                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                      {childVals.slice(0, 3).map(([cv, p]: [string, any]) => (
                                        <span key={cv} style={{
                                          fontSize: 11, padding: '2px 6px',
                                          background: color + '15', border: `1px solid ${color}30`,
                                          borderRadius: 4, color
                                        }}>
                                          {cv.slice(0, 20)}: {(p * 100).toFixed(1)}%
                                        </span>
                                      ))}
                                    </div>
                                  </td>
                                }
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
