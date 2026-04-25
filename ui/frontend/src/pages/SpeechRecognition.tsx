import { useState } from 'react'
import { fileToB64, v1 } from '../api_v1'
import type { TranscribeV1Response } from '../api_v1'
import AudioInputArea from '../components/AudioInputArea'
import FeedbackWidget from '../components/FeedbackWidget'
import LanguageBadge from '../components/LanguageBadge'

const LANGUAGES = ['auto', 'en', 'ha', 'ig', 'yo', 'pcm']

export default function SpeechRecognition() {
  const [language, setLanguage] = useState('auto')
  const [result, setResult] = useState<TranscribeV1Response | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const run = async (blob: Blob) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const b64 = await fileToB64(blob)
      const res = await v1.transcribe(b64, language)
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
        <h1 className="text-xl font-bold text-gray-100 mb-1">Speech Recognition</h1>
        <p className="text-sm text-gray-500">
          Transcribe audio in Hausa, Yorùbá, Igbo, Pidgin, or English.
        </p>
      </div>

      <div className="space-y-4">
        <AudioInputArea onAudio={run} disabled={loading} />

        <label className="block">
          <span className="text-sm text-gray-400 mb-1 block">Language</span>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm
                       text-gray-200 focus:outline-none focus:border-brand-500"
          >
            {LANGUAGES.map((l) => (
              <option key={l} value={l}>
                {l === 'auto' ? 'Auto-detect' : l.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
      </div>

      {loading && (
        <div className="text-sm text-gray-500 animate-pulse">Transcribing…</div>
      )}

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded text-red-300 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <LanguageBadge lang={result.language} />
            <span className="text-xs text-gray-500">{result.latency_ms.toFixed(0)} ms</span>
            <span className="text-xs text-gray-600">{result.model_version}</span>
          </div>
          <div className="p-4 bg-gray-900 border border-gray-800 rounded text-gray-200 text-sm whitespace-pre-wrap">
            {result.text}
          </div>
          <FeedbackWidget task="transcribe" modelVersion={result.model_version} />
        </div>
      )}
    </div>
  )
}
