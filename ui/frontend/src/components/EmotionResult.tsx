import type { EmotionResponse } from '../api_v1'
import PreviewBadge from './PreviewBadge'

const EMOTION_COLOR: Record<string, string> = {
  happy: 'bg-yellow-500',
  angry: 'bg-red-500',
  sad: 'bg-blue-500',
  neutral: 'bg-gray-500',
}

export default function EmotionResult({ result }: { result: EmotionResponse }) {
  const sorted = Object.entries(result.confidences).sort(([, a], [, b]) => b - a)

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-2xl font-bold text-gray-100 capitalize">{result.top_prediction}</span>
        <PreviewBadge />
        <span className="text-xs text-gray-500">{result.latency_ms.toFixed(0)} ms</span>
      </div>

      <div className="space-y-2">
        {sorted.map(([emotion, conf]) => (
          <div key={emotion} className="flex items-center gap-3">
            <span className="w-16 text-xs text-gray-400 capitalize">{emotion}</span>
            <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${EMOTION_COLOR[emotion] ?? 'bg-brand-500'} transition-all`}
                style={{ width: `${(conf * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="w-12 text-right text-xs text-gray-400">{(conf * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {result.segment_timeline && result.segment_timeline.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-2">Segment timeline</p>
          <div className="flex gap-1 flex-wrap">
            {result.segment_timeline.map((seg) => (
              <div
                key={seg.start}
                title={`${seg.start}s–${seg.end}s: ${seg.top}`}
                className={`text-xs px-2 py-1 rounded capitalize ${EMOTION_COLOR[seg.top] ?? 'bg-gray-700'} text-white`}
              >
                {seg.start}s
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-gray-600 border-t border-gray-800 pt-3">{result.disclaimer}</p>
    </div>
  )
}
