import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { TrainingEvent } from '../types'

interface Point {
  step: number
  train_loss?: number
  val_loss?: number
}

function buildSeries(events: TrainingEvent[]): Point[] {
  const byStep = new Map<number, Point>()

  for (const ev of events) {
    const step = (ev.data.step as number | undefined) ?? 0
    if (ev.type === 'step' && typeof ev.data.loss === 'number') {
      const p = byStep.get(step) ?? { step }
      p.train_loss = ev.data.loss as number
      byStep.set(step, p)
    }
    if (ev.type === 'eval' && typeof ev.data.eval_loss === 'number') {
      const p = byStep.get(step) ?? { step }
      p.val_loss = ev.data.eval_loss as number
      byStep.set(step, p)
    }
  }

  return [...byStep.values()].sort((a, b) => a.step - b.step)
}

interface Props {
  events: TrainingEvent[]
  height?: number
}

export default function LossChart({ events, height = 260 }: Props) {
  const data = buildSeries(events)

  if (data.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-gray-600 text-sm"
      >
        No loss data yet
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="step" stroke="#9ca3af" tick={{ fontSize: 11 }} />
        <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} width={48} />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #374151' }}
          labelStyle={{ color: '#e5e7eb' }}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="train_loss"
          name="train"
          stroke="#0ea5e9"
          dot={false}
          strokeWidth={1.5}
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="val_loss"
          name="val"
          stroke="#f59e0b"
          dot={false}
          strokeWidth={1.5}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
