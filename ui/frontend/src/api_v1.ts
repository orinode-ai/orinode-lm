const BASE = '/api/v1'

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail?.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { credentials: 'include' })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export interface WordTag {
  word: string
  start: number
  end: number
  confidence: number
}

export interface TranscribeV1Response {
  text: string
  language: string
  languages_detected: string[]
  word_tags: WordTag[]
  latency_ms: number
  model_version: string
}

export interface EmotionResponse {
  top_prediction: string
  confidences: Record<string, number>
  segment_timeline: Array<{
    start: number
    end: number
    top: string
    confidences: Record<string, number>
  }> | null
  model_version: string
  disclaimer: string
  latency_ms: number
}

export interface GenderResponse {
  prediction: string
  confidence: number
  model_version: string
  per_speaker: Array<{
    speaker: string
    start: number
    end: number
    prediction: string
    confidence: number
  }> | null
  latency_ms: number
}

export interface AnalyzeResponse {
  transcription: { text: string; language: string }
  emotion: EmotionResponse | null
  gender: GenderResponse | null
  total_latency_ms: number
}

export interface CheckpointInfo {
  id: string
  stage: string
  created_at: number
  size_bytes: number
  path: string
}

export interface StatsResponse {
  total_hours: number
  languages: string[]
  checkpoints_count: number
  runs_count: number
}

export interface SampleMeta {
  id: string
  title: string
  language: string
  duration_sec: number
  emotion?: string
  gender?: string
  transcript?: string
}

export const v1 = {
  transcribe: (audioB64: string, language = 'auto') =>
    post<TranscribeV1Response>('/transcribe', { audio_b64: audioB64, language }),

  emotion: (audioB64: string) =>
    post<EmotionResponse>('/emotion', { audio_b64: audioB64 }),

  gender: (audioB64: string, perSpeaker = false) =>
    post<GenderResponse>('/gender', { audio_b64: audioB64, per_speaker: perSpeaker }),

  analyze: (audioB64: string, language = 'auto') =>
    post<AnalyzeResponse>('/analyze', {
      audio_b64: audioB64,
      language,
      include_emotion: true,
      include_gender: true,
    }),

  checkpoints: () => get<CheckpointInfo[]>('/checkpoints'),

  stats: () => get<StatsResponse>('/stats'),

  samples: () => get<SampleMeta[]>('/samples'),

  feedback: (task: string, rating: number, comment = '', modelVersion = '') =>
    post<{ status: string }>('/feedback', { task, rating, comment, model_version: modelVersion }),
}

export async function fileToB64(file: File | Blob): Promise<string> {
  const buf = await file.arrayBuffer()
  const bytes = new Uint8Array(buf)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}
