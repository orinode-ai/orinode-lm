import { useState } from 'react'
import { fileToB64, v1 } from '../api_v1'
import type { EmotionResponse } from '../api_v1'
import AudioInputArea from '../components/AudioInputArea'
import EmotionResult from '../components/EmotionResult'
import FeedbackWidget from '../components/FeedbackWidget'
import PreviewBadge from '../components/PreviewBadge'

export default function Emotion() {
  const [result, setResult] = useState<EmotionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const run = async (blob: Blob) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const b64 = await fileToB64(blob)
      const res = await v1.emotion(b64)
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl space-y-6">
      <div className="flex items-start gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-xl font-bold text-gray-100">Emotion Detection</h1>
            <PreviewBadge />
          </div>
          <p className="text-sm text-gray-500">
            Classify speech as happy, angry, sad, or neutral. Initial model trained on
            English transfer data (IEMOCAP + RAVDESS).
          </p>
        </div>
      </div>

      <AudioInputArea onAudio={run} disabled={loading} />

      {loading && (
        <div className="text-sm text-gray-500 animate-pulse">Analysing emotion…</div>
      )}

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded text-red-300 text-sm">
          {error === 'Service Unavailable' || error.includes('503')
            ? 'No emotion model loaded. Run: make train-emotion'
            : error}
        </div>
      )}

      {result && (
        <div className="p-4 bg-gray-900 border border-gray-800 rounded space-y-4">
          <EmotionResult result={result} />
          <FeedbackWidget task="emotion" modelVersion={result.model_version} />
        </div>
      )}
    </div>
  )
}
