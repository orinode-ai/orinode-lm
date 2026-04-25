import { Link } from 'react-router-dom'
import type { RunSummary } from '../types'

const STATUS_COLORS: Record<string, string> = {
  running: 'bg-green-500',
  completed: 'bg-blue-500',
  error: 'bg-red-500',
}

function fmt(n: number | null, decimals = 4): string {
  if (n === null) return '—'
  return n.toFixed(decimals)
}

function pct(step: number, total: number): string {
  if (total === 0) return '0%'
  return `${Math.round((step / total) * 100)}%`
}

export default function RunCard({ run }: { run: RunSummary }) {
  const dot = STATUS_COLORS[run.status] ?? 'bg-gray-500'

  return (
    <Link
      to={`/training/runs/${run.run_id}`}
      className="block bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-brand-600 transition-colors"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-gray-100 truncate">{run.run_id}</span>
        <span className="flex items-center gap-1.5 text-xs text-gray-400">
          <span className={`inline-block w-2 h-2 rounded-full ${dot}`} />
          {run.status}
        </span>
      </div>
      <div className="text-xs text-gray-500 mb-3">{run.stage}</div>
      <div className="flex gap-4 text-xs text-gray-400">
        <span>
          step{' '}
          <span className="text-gray-200">
            {run.step.toLocaleString()}/{run.total_steps.toLocaleString()}
          </span>{' '}
          ({pct(run.step, run.total_steps)})
        </span>
        <span>
          loss <span className="text-gray-200">{fmt(run.train_loss)}</span>
        </span>
        {run.wer !== null && (
          <span>
            WER <span className="text-gray-200">{fmt(run.wer, 3)}</span>
          </span>
        )}
      </div>
    </Link>
  )
}
