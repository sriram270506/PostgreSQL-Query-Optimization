import { useState, useEffect } from 'react'

const ALL_TABLES = ['departments', 'employees', 'projects', 'assignments', 'locations', 'budgets']

const TABLE_COLORS: Record<string, string> = {
  departments: '#6366f1',
  employees: '#3b82f6',
  projects: '#10b981',
  assignments: '#f59e0b',
  locations: '#06b6d4',
  budgets: '#f43f5e',
}

export default function DataExplorer() {
  const [activeTable, setActiveTable] = useState('employees')
  const [tableInfo, setTableInfo] = useState<any>(null)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setPage(1)
  }, [activeTable])

  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams({ page: String(page), page_size: '25' })
    if (search) params.append('search', search)
    fetch(`/api/tables/${activeTable}?${params}`)
      .then(r => r.json())
      .then(d => { setTableInfo(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [activeTable, page, search])

  const accentColor = TABLE_COLORS[activeTable] || '#3b82f6'

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">⊞ Data Explorer</div>
        <div className="page-subtitle">Browse all 6 synthetic correlated tables with column statistics</div>
      </div>

      {/* Table Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {ALL_TABLES.map(t => (
          <button
            key={t}
            className={`btn ${activeTable === t ? 'btn-primary' : 'btn-ghost'}`}
            style={activeTable === t ? { background: TABLE_COLORS[t] } : {}}
            onClick={() => setActiveTable(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Search + info row */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
        <input
          className="input"
          style={{ maxWidth: 280 }}
          placeholder="Search rows…"
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(1) }}
        />
        {tableInfo && (
          <span className="badge badge-blue">
            {tableInfo.total_rows.toLocaleString()} rows
          </span>
        )}
        {tableInfo && (
          <span className="badge badge-gray">
            Page {tableInfo.page} / {tableInfo.total_pages}
          </span>
        )}
      </div>

      {/* Two-column: table data + stats */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 20, alignItems: 'start' }}>
        {/* Table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {loading ? (
            <div className="center-flex"><div className="loading-spin" /></div>
          ) : tableInfo?.rows ? (
            <>
              <div className="data-table-wrap" style={{ border: 'none', borderRadius: 0 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      {tableInfo.columns.map((c: any) => (
                        <th key={c.name}>
                          <div>{c.name}</div>
                          <div style={{ fontSize: 9, color: 'var(--text-disabled)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>{c.type}</div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tableInfo.rows.map((row: any, i: number) => (
                      <tr key={i}>
                        {tableInfo.columns.map((c: any) => (
                          <td key={c.name} className={typeof row[c.name] === 'number' ? 'mono' : ''}>
                            {String(row[c.name] ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div style={{ display: 'flex', gap: 8, padding: '12px 16px', borderTop: '1px solid var(--border)', alignItems: 'center' }}>
                <button className="btn btn-ghost btn-sm" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
                <button className="btn btn-ghost btn-sm" disabled={page >= tableInfo.total_pages} onClick={() => setPage(p => p + 1)}>Next →</button>
                <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
                  Showing {(page - 1) * 25 + 1}–{Math.min(page * 25, tableInfo.total_rows)} of {tableInfo.total_rows}
                </span>
              </div>
            </>
          ) : (
            <div className="center-flex">No data</div>
          )}
        </div>

        {/* Column Stats */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title" style={{ marginBottom: 12 }}>Column Statistics</div>
            {tableInfo?.stats && Object.entries(tableInfo.stats).map(([col, info]: [string, any]) => (
              <div key={col} style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: accentColor }}>{col}</span>
                  <span className={`badge ${info.type === 'numeric' ? 'badge-blue' : 'badge-purple'}`} style={{ fontSize: 10 }}>{info.type}</span>
                </div>
                {info.type === 'numeric' ? (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.8 }}>
                    <div>Min: <b style={{ color: 'var(--text-secondary)' }}>{Number(info.min).toLocaleString()}</b></div>
                    <div>Max: <b style={{ color: 'var(--text-secondary)' }}>{Number(info.max).toLocaleString()}</b></div>
                    <div>Mean: <b style={{ color: 'var(--text-secondary)' }}>{Number(info.mean).toLocaleString()}</b></div>
                    <div>Unique: <b style={{ color: 'var(--text-secondary)' }}>{info.unique}</b></div>
                  </div>
                ) : (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    <div style={{ marginBottom: 4 }}>Unique: <b style={{ color: 'var(--text-secondary)' }}>{info.unique}</b></div>
                    {Object.entries(info.top_values || {}).slice(0, 3).map(([v, cnt]: [string, any]) => (
                      <div key={v} style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{v}</span>
                        <span>{cnt}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
