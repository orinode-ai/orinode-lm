import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { v1 } from '../api_v1'
import type { StatsResponse } from '../api_v1'
import LanguageBadge from '../components/LanguageBadge'

const FEATURES = [
  {
    title: 'Speech Recognition',
    desc: 'Transcribe Nigerian speech across 5 languages with a single API call.',
    href: '/speech',
    icon: '🎙',
  },
  {
    title: 'Emotion Detection',
    desc: 'Classify audio into happy, angry, sad, or neutral with per-segment timelines.',
    href: '/emotion',
    icon: '😊',
  },
  {
    title: 'Gender Classification',
    desc: 'Predict speaker gender from raw audio, with optional per-speaker diarization.',
    href: '/gender',
    icon: '👤',
  },
  {
    title: 'API Console',
    desc: 'Explore every endpoint interactively with live request/response examples.',
    href: '/api-console',
    icon: '⚡',
  },
]

export default function Home() {
  const [stats, setStats] = useState<StatsResponse | null>(null)

  useEffect(() => {
    v1.stats().then(setStats).catch(() => null)
  }, [])

  return (
    <div className="max-w-3xl space-y-10">
      {/* Hero */}
      <div>
        <h1 className="text-3xl font-bold text-gray-100 mb-3">Orinode Speech AI</h1>
        <p className="text-gray-400 text-base leading-relaxed">
          Open-source speech processing built for Nigerian languages — Hausa, Yorùbá, Igbo, Pidgin,
          and English. Trained on NaijaVoices, AfriSpeech, and community-sourced audio.
        </p>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: 'Hours of audio', value: `${stats.total_hours.toFixed(0)}h` },
            { label: 'Languages', value: stats.languages.length },
            { label: 'Checkpoints', value: stats.checkpoints_count },
            { label: 'Training runs', value: stats.runs_count },
          ].map(({ label, value }) => (
            <div key={label} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <p className="text-2xl font-bold text-brand-400">{value}</p>
              <p className="text-xs text-gray-500 mt-1">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Languages */}
      <div>
        <p className="text-sm text-gray-500 mb-3">Supported languages</p>
        <div className="flex gap-2 flex-wrap">
          {['en', 'ha', 'yo', 'ig', 'pcm'].map((l) => (
            <LanguageBadge key={l} lang={l} />
          ))}
        </div>
      </div>

      {/* Feature cards */}
      <div className="grid sm:grid-cols-2 gap-4">
        {FEATURES.map(({ title, desc, href, icon }) => (
          <Link
            key={href}
            to={href}
            className="block bg-gray-900 border border-gray-800 hover:border-brand-700
                       rounded-xl p-5 transition-colors group"
          >
            <div className="text-2xl mb-2">{icon}</div>
            <h2 className="font-semibold text-gray-100 group-hover:text-brand-400 transition-colors mb-1">
              {title}
            </h2>
            <p className="text-sm text-gray-500">{desc}</p>
          </Link>
        ))}
      </div>

      {/* Quick start */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="font-semibold text-gray-200 mb-3">Quick start</h2>
        <pre className="bg-gray-950 rounded p-3 text-xs text-gray-400 overflow-x-auto font-mono">
          {`curl -X POST http://127.0.0.1:7860/api/v1/transcribe \\
  -H "Content-Type: application/json" \\
  -d '{"audio_b64": "<base64>", "language": "auto"}'`}
        </pre>
        <p className="text-xs text-gray-600 mt-3">
          Full API reference →{' '}
          <Link to="/api-console" className="text-brand-400 hover:underline">
            API Console
          </Link>
        </p>
      </div>
    </div>
  )
}
