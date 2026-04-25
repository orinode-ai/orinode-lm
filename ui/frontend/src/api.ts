import type { RunDetail, RunSummary, TranscribeResponse } from './types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { credentials: 'include' })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const fetchRuns = (): Promise<RunSummary[]> =>
  get<RunSummary[]>('/runs')

export const fetchRun = (runId: string): Promise<RunDetail> =>
  get<RunDetail>(`/runs/${runId}`)

export async function transcribeAudio(
  audioB64: string,
  language = 'auto',
  maxNewTokens = 256,
): Promise<TranscribeResponse> {
  const res = await fetch(`${BASE}/transcribe`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio_b64: audioB64,
      language,
      max_new_tokens: maxNewTokens,
    }),
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<TranscribeResponse>
}
