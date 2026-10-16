export const EST = ['postgres', 'factorjoin', 'mlp', 'treelstm'] as const
export type EstName = (typeof EST)[number] | 'postgres_live'

export const EST_LABEL: Record<string, string> = {
  postgres: 'PostgreSQL planner',
  factorjoin: 'FactorJoin (BN)',
  mlp: 'Fusion MLP',
  treelstm: 'TreeLSTM + MLP',
  postgres_live: 'PostgreSQL (live)',
}
export const EST_COLOR: Record<string, string> = {
  postgres: 'var(--c-postgres)',
  factorjoin: 'var(--c-factorjoin)',
  mlp: 'var(--c-mlp)',
  treelstm: 'var(--c-treelstm)',
  postgres_live: 'var(--c-postgres)',
}

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, body === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try { msg = (await res.json()).detail ?? msg } catch { /* non-JSON error */ }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return res.json()
}

export const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US')
export function fmtCompact(n: number): string {
  if (!isFinite(n)) return '–'
  const a = Math.abs(n)
  if (a >= 1e9) return (n / 1e9).toFixed(a >= 1e10 ? 0 : 1) + 'B'
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M'
  if (a >= 1e4) return (n / 1e3).toFixed(0) + 'K'
  return fmtInt(n)
}
export const fmtQ = (q: number) => (q >= 100 ? fmtCompact(q) : q >= 10 ? q.toFixed(1) : q.toFixed(2)) + '×'
export const fmtMs = (ms: number) => (ms < 1 ? ms.toFixed(2) : ms < 100 ? ms.toFixed(1) : fmtInt(ms)) + ' ms'
export const pct = (x: number, d = 0) => (x * 100).toFixed(d) + '%'

/** sequential maroon ramp, t in [0,1] (light -> dark) */
export function seq(t: number): string {
  const s = Math.max(0, Math.min(1, t))
  const a = [248, 239, 242], b = [107, 15, 43]
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * s))
  return `rgb(${c.join(',')})`
}
/** diverging: negative -> blue, 0 -> neutral gray, positive -> maroon */
export function div(t: number): string {
  const s = Math.max(-1, Math.min(1, t))
  const mid = [240, 236, 235]
  const end = s < 0 ? [45, 92, 150] : [138, 21, 56]
  const c = mid.map((v, i) => Math.round(v + (end[i] - v) * Math.abs(s)))
  return `rgb(${c.join(',')})`
}
