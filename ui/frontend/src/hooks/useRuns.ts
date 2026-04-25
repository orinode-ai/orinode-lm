import { useEffect, useState } from 'react'
import { fetchRuns } from '../api'
import type { RunSummary } from '../types'

export function useRuns(pollMs = 5000) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    const load = () => {
      fetchRuns()
        .then((data) => {
          if (!cancelled) {
            setRuns(data)
            setError(null)
            setLoading(false)
          }
        })
        .catch((err: unknown) => {
          if (!cancelled) {
            setError(err instanceof Error ? err.message : String(err))
            setLoading(false)
          }
        })
    }

    load()
    const id = setInterval(load, pollMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [pollMs])

  return { runs, error, loading }
}
