import { useEffect, useRef, useState } from 'react'
import type { TrainingEvent } from '../types'

export function useRunEvents(runId: string | null): TrainingEvent[] {
  const [events, setEvents] = useState<TrainingEvent[]>([])
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!runId) return
    ws.current?.close()

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.hostname
    const port = import.meta.env.DEV ? '7860' : window.location.port
    const url = `${proto}://${host}:${port}/ws/runs/${runId}`

    const socket = new WebSocket(url)
    ws.current = socket

    socket.onmessage = (e: MessageEvent) => {
      try {
        const ev = JSON.parse(e.data as string) as TrainingEvent
        setEvents((prev) => [...prev, ev])
      } catch {
        // ignore malformed frames
      }
    }

    socket.onerror = () => socket.close()

    return () => socket.close()
  }, [runId])

  return events
}
