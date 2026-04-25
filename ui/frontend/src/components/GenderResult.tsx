import type { GenderResponse } from '../api_v1'

export default function GenderResult({ result }: { result: GenderResponse }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <span className="text-2xl font-bold text-gray-100 capitalize">{result.prediction}</span>
        <span className="text-sm text-gray-400">{(result.confidence * 100).toFixed(1)}% confidence</span>
        <span className="text-xs text-gray-500">{result.latency_ms.toFixed(0)} ms</span>
      </div>

      {result.per_speaker && result.per_speaker.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-2">Per-speaker breakdown</p>
          <div className="space-y-1.5">
            {result.per_speaker.map((s) => (
              <div
                key={`${s.speaker}-${s.start}`}
                className="flex items-center gap-3 text-sm text-gray-300"
              >
                <span className="w-20 font-mono text-xs text-gray-400">{s.speaker}</span>
                <span className="capitalize">{s.prediction}</span>
                <span className="text-gray-500 text-xs">{(s.confidence * 100).toFixed(0)}%</span>
                <span className="text-gray-600 text-xs">
                  {s.start}s – {s.end}s
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
