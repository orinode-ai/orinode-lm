import RunCard from '../components/RunCard'
import { useRuns } from '../hooks/useRuns'

export default function Dashboard() {
  const { runs, loading, error } = useRuns(3000)

  const active = runs.filter((r) => r.status === 'running')
  const bestWer = runs
    .filter((r) => r.wer !== null)
    .reduce<number | null>((best, r) => {
      if (r.wer === null) return best
      return best === null || r.wer < best ? r.wer : best
    }, null)

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-100 mb-6">Dashboard</h1>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <Stat label="Active runs" value={active.length} />
        <Stat label="Total runs" value={runs.length} />
        <Stat
          label="Best WER"
          value={bestWer !== null ? bestWer.toFixed(3) : '—'}
        />
      </div>

      {/* Recent runs */}
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-3">
        Recent runs
      </h2>

      {loading && <p className="text-gray-500 text-sm">Loading…</p>}
      {error && <p className="text-red-400 text-sm">{error}</p>}

      {!loading && runs.length === 0 && (
        <p className="text-gray-600 text-sm">
          No runs yet. Start training with <code>make smoke-test</code>.
        </p>
      )}

      <div className="space-y-3">
        {runs.slice(0, 8).map((r) => (
          <RunCard key={r.run_id} run={r} />
        ))}
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-2xl font-bold text-gray-100">{value}</div>
    </div>
  )
}
