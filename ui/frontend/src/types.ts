export interface EventData {
  [key: string]: unknown
}

export interface TrainingEvent {
  ts: number
  run_id: string
  type: string
  data: EventData
}

export interface RunSummary {
  run_id: string
  stage: string
  status: 'running' | 'completed' | 'error'
  step: number
  total_steps: number
  train_loss: number | null
  val_loss: number | null
  wer: number | null
  created_at: number
  updated_at: number
}

export interface RunDetail extends RunSummary {
  events: TrainingEvent[]
}

export interface TranscribeResponse {
  text: string
  language: string
  latency_ms: number
}
