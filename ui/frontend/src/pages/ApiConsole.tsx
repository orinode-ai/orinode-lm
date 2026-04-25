import { useState } from 'react'
import CodeBlock from '../components/CodeBlock'

interface Endpoint {
  id: string
  method: 'GET' | 'POST'
  path: string
  description: string
  exampleRequest?: string
  exampleResponse: string
}

const ENDPOINTS: Endpoint[] = [
  {
    id: 'stats',
    method: 'GET',
    path: '/api/v1/stats',
    description: 'Return aggregate stats about the model workspace.',
    exampleResponse: JSON.stringify(
      { total_hours: 42.5, languages: ['en', 'ha', 'yo', 'ig', 'pcm'], checkpoints_count: 3, runs_count: 5 },
      null,
      2,
    ),
  },
  {
    id: 'transcribe',
    method: 'POST',
    path: '/api/v1/transcribe',
    description: 'Transcribe audio (WAV/FLAC) encoded as base64.',
    exampleRequest: JSON.stringify({ audio_b64: '<base64>', language: 'auto' }, null, 2),
    exampleResponse: JSON.stringify(
      {
        text: 'I dey happy to meet you today',
        language: 'pcm',
        languages_detected: ['pcm'],
        word_tags: [],
        latency_ms: 312.4,
        model_version: 'speech_llm/v1',
      },
      null,
      2,
    ),
  },
  {
    id: 'emotion',
    method: 'POST',
    path: '/api/v1/emotion',
    description: 'Classify emotion in audio as happy / angry / sad / neutral.',
    exampleRequest: JSON.stringify({ audio_b64: '<base64>' }, null, 2),
    exampleResponse: JSON.stringify(
      {
        top_prediction: 'happy',
        confidences: { happy: 0.72, angry: 0.1, sad: 0.08, neutral: 0.1 },
        segment_timeline: null,
        model_version: 'aux_emotion/v1',
        disclaimer: 'Preview model trained on English transfer data.',
        latency_ms: 120.1,
      },
      null,
      2,
    ),
  },
  {
    id: 'gender',
    method: 'POST',
    path: '/api/v1/gender',
    description: 'Classify speaker gender from audio.',
    exampleRequest: JSON.stringify({ audio_b64: '<base64>', per_speaker: false }, null, 2),
    exampleResponse: JSON.stringify(
      {
        prediction: 'female',
        confidence: 0.91,
        model_version: 'aux_gender/v1',
        per_speaker: null,
        latency_ms: 85.2,
      },
      null,
      2,
    ),
  },
  {
    id: 'analyze',
    method: 'POST',
    path: '/api/v1/analyze',
    description: 'Run transcription + emotion + gender in a single call.',
    exampleRequest: JSON.stringify({ audio_b64: '<base64>', language: 'auto' }, null, 2),
    exampleResponse: JSON.stringify(
      {
        transcription: { text: 'Hello from Nigeria!', language: 'en' },
        emotion: { top_prediction: 'happy' },
        gender: { prediction: 'male', confidence: 0.87 },
        total_latency_ms: 540.3,
      },
      null,
      2,
    ),
  },
  {
    id: 'checkpoints',
    method: 'GET',
    path: '/api/v1/checkpoints',
    description: 'List all available model checkpoints.',
    exampleResponse: JSON.stringify(
      [{ id: 'stage4_step8000', stage: 'stage4', created_at: 1700000000, size_bytes: 4200000000, path: 'stage4/stage4_step8000.pt' }],
      null,
      2,
    ),
  },
  {
    id: 'feedback',
    method: 'POST',
    path: '/api/v1/feedback',
    description: 'Submit a rating (1–5) for a model output.',
    exampleRequest: JSON.stringify(
      { task: 'transcribe', rating: 4, comment: 'Mostly correct', model_version: 'speech_llm/v1' },
      null,
      2,
    ),
    exampleResponse: JSON.stringify({ status: 'recorded' }, null, 2),
  },
]

function buildCurl(ep: Endpoint): string {
  if (ep.method === 'GET') {
    return `curl http://127.0.0.1:7860${ep.path}`
  }
  return `curl -X POST http://127.0.0.1:7860${ep.path} \\
  -H "Content-Type: application/json" \\
  -d '${ep.exampleRequest ?? '{}'}'`
}

export default function ApiConsole() {
  const [selected, setSelected] = useState<string>('stats')
  const [liveResponse, setLiveResponse] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [liveError, setLiveError] = useState<string | null>(null)

  const ep = ENDPOINTS.find((e) => e.id === selected)!

  const tryLive = async () => {
    if (ep.method !== 'GET') return
    setLoading(true)
    setLiveResponse(null)
    setLiveError(null)
    try {
      const res = await fetch(`/api/v1/${ep.id === 'stats' ? 'stats' : ep.id}`)
      const json = await res.json()
      setLiveResponse(JSON.stringify(json, null, 2))
    } catch (e: unknown) {
      setLiveError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-100 mb-1">API Console</h1>
        <p className="text-sm text-gray-500">
          Base URL: <code className="text-brand-400">http://127.0.0.1:7860</code>
        </p>
      </div>

      <div className="grid sm:grid-cols-3 gap-4">
        {/* Endpoint list */}
        <div className="space-y-1">
          {ENDPOINTS.map((e) => (
            <button
              key={e.id}
              onClick={() => {
                setSelected(e.id)
                setLiveResponse(null)
                setLiveError(null)
              }}
              className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
                e.id === selected
                  ? 'bg-brand-700 text-white'
                  : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'
              }`}
            >
              <span
                className={`text-xs font-mono mr-2 ${
                  e.method === 'GET' ? 'text-green-400' : 'text-yellow-400'
                }`}
              >
                {e.method}
              </span>
              <span className="font-mono">{e.id}</span>
            </button>
          ))}
        </div>

        {/* Detail */}
        <div className="sm:col-span-2 space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span
                className={`text-xs font-mono font-bold ${
                  ep.method === 'GET' ? 'text-green-400' : 'text-yellow-400'
                }`}
              >
                {ep.method}
              </span>
              <code className="text-sm text-gray-300">{ep.path}</code>
            </div>
            <p className="text-sm text-gray-500">{ep.description}</p>
          </div>

          {ep.exampleRequest && (
            <div>
              <p className="text-xs text-gray-500 mb-1">Request body</p>
              <CodeBlock code={ep.exampleRequest} language="json" />
            </div>
          )}

          <div>
            <p className="text-xs text-gray-500 mb-1">Example response</p>
            <CodeBlock code={ep.exampleResponse} language="json" />
          </div>

          <div>
            <p className="text-xs text-gray-500 mb-1">cURL</p>
            <CodeBlock code={buildCurl(ep)} language="bash" />
          </div>

          {ep.method === 'GET' && (
            <div>
              <button
                onClick={tryLive}
                disabled={loading}
                className="text-sm px-4 py-2 rounded bg-brand-600 hover:bg-brand-500 text-white
                           disabled:opacity-50 transition-colors"
              >
                {loading ? 'Fetching…' : 'Try live ▶'}
              </button>
              {liveError && (
                <p className="mt-2 text-xs text-red-400">{liveError}</p>
              )}
              {liveResponse && (
                <div className="mt-3">
                  <p className="text-xs text-green-400 mb-1">Live response</p>
                  <CodeBlock code={liveResponse} language="json" />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
