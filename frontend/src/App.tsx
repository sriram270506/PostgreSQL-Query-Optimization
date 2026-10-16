import { useEffect, useState } from 'react'
import Overview from './pages/Overview'
import QueryLab from './pages/QueryLab'
import Benchmark from './pages/Benchmark'
import Internals from './pages/Internals'
import DataEngine from './pages/DataEngine'

const PAGES = [
  ['overview', 'Overview'],
  ['lab', 'Query Lab'],
  ['benchmark', 'Benchmark'],
  ['internals', 'Inside the Models'],
  ['data', 'Data & Engine'],
] as const
type Page = (typeof PAGES)[number][0]

function current(): Page {
  const h = window.location.hash.replace('#/', '').split('?')[0]
  return (PAGES.find(p => p[0] === h)?.[0] ?? 'overview') as Page
}

export default function App() {
  const [page, setPage] = useState<Page>(current)
  useEffect(() => {
    const on = () => { setPage(current()); window.scrollTo({ top: 0 }) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return (
    <>
      <header className="topbar">
        <div className="topbar-in">
          <a className="brand" href="#/overview" aria-label="Cardinality Lab home">
            <span className="brand-mark">C</span>
            <span className="brand-copy"><strong>Cardinality Lab</strong><small>LEARNED QUERY OPTIMIZATION</small></span>
          </a>
          <nav className="nav" aria-label="Main">
            {PAGES.map(([id, label]) => <a key={id} href={`#/${id}`} className={page === id ? 'on' : ''} aria-current={page === id ? 'page' : undefined}>{label}</a>)}
          </nav>
        </div>
      </header>
      <main className="shell fade-in" key={page}>
        {page === 'overview' && <Overview />}
        {page === 'lab' && <QueryLab />}
        {page === 'benchmark' && <Benchmark />}
        {page === 'internals' && <Internals />}
        {page === 'data' && <DataEngine />}
        <footer className="footer">
          <span>AIE-A · Group 17 · Bhuvaneswaran S · Sriram S</span>
          <span>Embedded SQL engine · Chow-Liu BN · FactorJoin · TreeLSTM + MLP</span>
        </footer>
      </main>
    </>
  )
}
