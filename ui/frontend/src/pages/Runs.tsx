import { Link } from 'react-router-dom'
import { useRuns } from '../hooks/useRuns'
import type { RunSummary } from '../types'

const STATUS_COLORS: Record<string, string> = {
  running: 'text-green-400',
  completed: 'text-blue-400',
  error: 'text-red-400',
}

function Cell({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-3 py-2 text-sm ${className}`}>{children}</td>
}

function Row({ run }: { run: RunSummary }) {
  const color = STATUS_COLORS[run.status] ?? 'text-gray-400'
  return (
    <tr className="border-b border-gray-800 hover:bg-gray-900 transition-colors">
      <Cell>
        <Link to={`/training/runs/${run.run_id}`} className="text-brand-400 hover:underline">
          {run.run_id}
        </Link>
      </Cell>
      <Cell className="text-gray-400">{run.stage}</Cell>
      <Cell className={color}>{run.status}</Cell>
      <Cell className="text-gray-300 tabular-nums">
        {run.step.toLocaleString()}/{run.total_steps.toLocaleString()}
      </Cell>
      <Cell className="text-gray-300 tabular-nums">
        {run.train_loss !== null ? run.train_loss.toFixed(4) : '—'}
      </Cell>
      <Cell className="text-gray-300 tabular-nums">
        {run.val_loss !== null ? run.val_loss.toFixed(4) : '—'}
      </Cell>
      <Cell className="text-gray-300 tabular-nums">
        {run.wer !== null ? run.wer.toFixed(3) : '—'}
      </Cell>
    </tr>
  )
}

export default function Runs() {
  const { runs, loading, error } = useRuns()

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-100 mb-6">Runs</h1>

      {loading && <p className="text-gray-500 text-sm">Loading…</p>}
      {error && <p className="text-red-400 text-sm">{error}</p>}

      {!loading && runs.length === 0 && (
        <p className="text-gray-600 text-sm">No runs yet.</p>
      )}

      {runs.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-gray-800">
          <table className="w-full text-left">
            <thead className="bg-gray-900 border-b border-gray-800">
              <tr>
                {['Run ID', 'Stage', 'Status', 'Step', 'Train Loss', 'Val Loss', 'WER'].map(
                  (h) => (
                    <th
                      key={h}
                      className="px-3 py-2 text-xs text-gray-500 font-semibold uppercase tracking-wide"
                    >
                      {h}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <Row key={r.run_id} run={r} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
