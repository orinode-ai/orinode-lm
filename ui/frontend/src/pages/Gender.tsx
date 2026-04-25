import { useState } from 'react'
import { fileToB64, v1 } from '../api_v1'
import type { GenderResponse } from '../api_v1'
import AudioInputArea from '../components/AudioInputArea'
import FeedbackWidget from '../components/FeedbackWidget'
import GenderResult from '../components/GenderResult'

export default function Gender() {
  const [perSpeaker, setPerSpeaker] = useState(false)
  const [result, setResult] = useState<GenderResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const run = async (blob: Blob) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const b64 = await fileToB64(blob)
      const res = await v1.gender(b64, perSpeaker)
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-100 mb-1">Gender Classification</h1>
        <p className="text-sm text-gray-500">
          Predict speaker gender from audio. Enable per-speaker mode for multi-speaker recordings.
        </p>
      </div>

      <div className="space-y-4">
        <AudioInputArea onAudio={run} disabled={loading} />

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={perSpeaker}
            onChange={(e) => setPerSpeaker(e.target.checked)}
            className="rounded border-gray-600 bg-gray-800 text-brand-500 focus:ring-brand-500"
          />
          <span className="text-sm text-gray-400">Per-speaker mode (requires pyannote)</span>
        </label>
      </div>

      {loading && (
        <div className="text-sm text-gray-500 animate-pulse">Classifying gender…</div>
      )}

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded text-red-300 text-sm">
          {error.includes('503')
            ? 'No gender model loaded. Run: make train-gender'
            : error}
        </div>
      )}

      {result && (
        <div className="p-4 bg-gray-900 border border-gray-800 rounded space-y-4">
          <GenderResult result={result} />
          <FeedbackWidget task="gender" modelVersion={result.model_version} />
        </div>
      )}
    </div>
  )
}
