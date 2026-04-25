import { useEffect, useState } from 'react'
import { fetchRun } from '../api'
import LossChart from '../components/LossChart'
import { useRuns } from '../hooks/useRuns'
import type { RunDetail } from '../types'

export default function Compare() {
  const { runs } = useRuns()
  const [leftId, setLeftId] = useState('')
  const [rightId, setRightId] = useState('')
  const [left, setLeft] = useState<RunDetail | null>(null)
  const [right, setRight] = useState<RunDetail | null>(null)

  useEffect(() => {
    if (leftId) fetchRun(leftId).then(setLeft).catch(() => setLeft(null))
  }, [leftId])

  useEffect(() => {
    if (rightId) fetchRun(rightId).then(setRight).catch(() => setRight(null))
  }, [rightId])

  const runIds = runs.map((r) => r.run_id)

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-100 mb-6">Compare Runs</h1>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <Select label="Run A" value={leftId} options={runIds} onChange={setLeftId} />
        <Select label="Run B" value={rightId} options={runIds} onChange={setRightId} />
      </div>

      {(left || right) && (
        <div className="grid grid-cols-2 gap-4">
          <Panel run={left} />
          <Panel run={right} />
        </div>
      )}

      {!left && !right && (
        <p className="text-gray-600 text-sm">Select two runs to compare.</p>
      )}
    </div>
  )
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
}) {
  return (
    <label className="block">
      <span className="text-sm text-gray-400 mb-1 block">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm
                   text-gray-200 focus:outline-none focus:border-brand-500"
      >
        <option value="">— select —</option>
        {options.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
    </label>
  )
}

function Panel({ run }: { run: RunDetail | null }) {
  if (!run) return <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 text-gray-600 text-sm">—</div>

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">{run.run_id}</h2>
      <p className="text-xs text-gray-500 mb-3">{run.stage} · {run.status}</p>
      <div className="grid grid-cols-3 gap-2 mb-4">
        <Metric k="Train loss" v={run.train_loss?.toFixed(4) ?? '—'} />
        <Metric k="Val loss" v={run.val_loss?.toFixed(4) ?? '—'} />
        <Metric k="WER" v={run.wer?.toFixed(3) ?? '—'} />
      </div>
      <LossChart events={run.events} height={180} />
    </div>
  )
}

function Metric({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="text-xs text-gray-600">{k}</div>
      <div className="text-sm font-bold">{v}</div>
    </div>
  )
}
