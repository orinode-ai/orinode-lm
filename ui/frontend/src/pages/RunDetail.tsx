import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { fetchRun } from '../api'
import LossChart from '../components/LossChart'
import { useRunEvents } from '../hooks/useWebSocket'
import type { RunDetail as RunDetailType, TrainingEvent } from '../types'

function EventRow({ ev }: { ev: TrainingEvent }) {
  const t = new Date(ev.ts * 1000).toLocaleTimeString()
  return (
    <div className="flex gap-3 text-xs border-b border-gray-800 py-1.5">
      <span className="text-gray-600 shrink-0 w-20">{t}</span>
      <span className="text-brand-400 shrink-0 w-24">{ev.type}</span>
      <span className="text-gray-400 truncate">{JSON.stringify(ev.data)}</span>
    </div>
  )
}

function formatWer(ev: TrainingEvent | undefined, fallback: number | null): string {
  if (ev) {
    const wer = ev.data.wer as Record<string, number> | undefined
    if (wer && typeof wer === 'object') {
      const parts = Object.entries(wer)
        .filter(([, v]) => typeof v === 'number')
        .map(([k, v]) => `${k.toUpperCase()}: ${v.toFixed(2)}`)
      if (parts.length) return parts.join(' | ')
    }
  }
  return fallback != null ? fallback.toFixed(3) : '—'
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const [run, setRun] = useState<RunDetailType | null>(null)
  const [error, setError] = useState<string | null>(null)

  const liveEvents = useRunEvents(runId ?? null)

  useEffect(() => {
    if (!runId) return
    fetchRun(runId)
      .then(setRun)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : String(e)),
      )
  }, [runId])

  if (error) return <p className="text-red-400">{error}</p>
  if (!run) return <p className="text-gray-500 text-sm">Loading…</p>

  const allEvents = [...run.events, ...liveEvents]

  // Bug 4: derive live metrics from most recent step / eval events
  const lastStep = [...allEvents].reverse().find((e) => e.type === 'step')
  const lastEval = [...allEvents].reverse().find((e) => e.type === 'eval')

  const trainLoss =
    lastStep && typeof lastStep.data.loss === 'number'
      ? (lastStep.data.loss as number).toFixed(4)
      : run.train_loss != null ? run.train_loss.toFixed(4) : '—'

  const valLoss =
    lastEval && typeof lastEval.data.eval_loss === 'number'
      ? (lastEval.data.eval_loss as number).toFixed(4)
      : run.val_loss != null ? run.val_loss.toFixed(4) : '—'

  const werDisplay = formatWer(lastEval, run.wer)

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-100 mb-1">{run.run_id}</h1>
      <p className="text-sm text-gray-500 mb-6">
        {run.stage} · {run.status} · step {run.step.toLocaleString()}/
        {run.total_steps.toLocaleString()}
      </p>

      {/* Metrics grid — Bug 4: wired to live event data */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        <div className="bg-gray-900 border border-gray-800 rounded p-3">
          <div className="text-xs text-gray-500 mb-0.5">Train loss</div>
          <div className="text-lg font-bold">{trainLoss}</div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded p-3">
          <div className="text-xs text-gray-500 mb-0.5">Val loss</div>
          <div className="text-lg font-bold">{valLoss}</div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded p-3">
          <div className="text-xs text-gray-500 mb-0.5">WER</div>
          <div className="text-sm font-bold leading-tight break-all">{werDisplay}</div>
        </div>
      </div>

      {/* Loss chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-3">Loss</h2>
        <LossChart events={allEvents} />
      </div>

      {/* Event log */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-gray-400 mb-3">
          Event log ({allEvents.length})
        </h2>
        <div className="max-h-80 overflow-y-auto">
          {allEvents.length === 0 && (
            <p className="text-gray-600 text-xs">No events yet.</p>
          )}
          {[...allEvents].reverse().map((ev, i) => (
            <EventRow key={i} ev={ev} />
          ))}
        </div>
      </div>
    </div>
  )
}
