import { useState } from 'react'
import { v1 } from '../api_v1'

interface Props {
  task: string
  modelVersion?: string
}

export default function FeedbackWidget({ task, modelVersion = '' }: Props) {
  const [rating, setRating] = useState(0)
  const [comment, setComment] = useState('')
  const [sent, setSent] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    if (!rating) return
    try {
      await v1.feedback(task, rating, comment, modelVersion)
      setSent(true)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  if (sent)
    return <p className="text-xs text-green-400 mt-3">Thanks for your feedback!</p>

  return (
    <div className="mt-4 border-t border-gray-800 pt-4 space-y-2">
      <p className="text-xs text-gray-500">Rate this result</p>
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            onClick={() => setRating(n)}
            className={`text-lg transition-colors ${n <= rating ? 'text-yellow-400' : 'text-gray-700'}`}
          >
            ★
          </button>
        ))}
      </div>
      <input
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="Optional comment…"
        className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm
                   text-gray-300 placeholder-gray-600 focus:outline-none focus:border-brand-500"
      />
      {error && <p className="text-xs text-red-400">{error}</p>}
      <button
        onClick={submit}
        disabled={!rating}
        className="text-xs px-3 py-1.5 rounded bg-brand-700 hover:bg-brand-600 text-white
                   disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        Submit
      </button>
    </div>
  )
}
