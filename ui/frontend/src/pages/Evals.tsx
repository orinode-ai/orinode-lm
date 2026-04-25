import { useRuns } from '../hooks/useRuns'
import type { RunSummary } from '../types'

const LANGS = ['en', 'ha', 'ig', 'yo', 'pcm', 'overall']

function evalCell(run: RunSummary, lang: string): string {
  if (lang === 'overall') return run.wer !== null ? run.wer.toFixed(3) : '—'
  return '—'
}

export default function Evals() {
  const { runs, loading } = useRuns()
  const completed = runs.filter((r) => r.wer !== null)

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-100 mb-6">Eval Results</h1>

      {loading && <p className="text-gray-500 text-sm">Loading…</p>}

      {!loading && completed.length === 0 && (
        <p className="text-gray-600 text-sm">
          No eval results yet. Run <code>make eval RUN=&lt;run_id&gt;</code>.
        </p>
      )}

      {completed.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-gray-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-gray-900 border-b border-gray-800">
              <tr>
                <th className="px-3 py-2 text-xs text-gray-500 font-semibold uppercase">Run</th>
                <th className="px-3 py-2 text-xs text-gray-500 font-semibold uppercase">Stage</th>
                {LANGS.map((l) => (
                  <th
                    key={l}
                    className="px-3 py-2 text-xs text-gray-500 font-semibold uppercase"
                  >
                    WER/{l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {completed.map((run) => (
                <tr key={run.run_id} className="border-b border-gray-800 hover:bg-gray-900">
                  <td className="px-3 py-2 text-brand-400">{run.run_id}</td>
                  <td className="px-3 py-2 text-gray-400">{run.stage}</td>
                  {LANGS.map((l) => (
                    <td key={l} className="px-3 py-2 text-gray-300 tabular-nums">
                      {evalCell(run, l)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
