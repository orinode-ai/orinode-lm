import { useEffect, useRef, useState } from 'react'

interface Props {
  onAudio: (blob: Blob) => void
  disabled?: boolean
}

function fmt(s: number) {
  const m = Math.floor(s / 60)
  const ss = s % 60
  return `${m}:${ss.toString().padStart(2, '0')}`
}

export default function AudioRecorder({ onAudio, disabled }: Props) {
  const [state, setState] = useState<'idle' | 'recording' | 'error'>('idle')
  const [seconds, setSeconds] = useState(0)
  const [errMsg, setErrMsg] = useState('')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => {
      timerRef.current && clearInterval(timerRef.current)
      recorderRef.current?.stream.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const start = async () => {
    setErrMsg('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrMsg('Recording requires HTTPS — use file upload instead')
      setState('error')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg'].find(
        (m) => MediaRecorder.isTypeSupported(m),
      ) ?? ''
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      chunksRef.current = []

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || 'audio/webm' })
        onAudio(blob)
      }

      mr.start(100) // collect data every 100 ms for smooth blobs
      recorderRef.current = mr
      setState('recording')
      setSeconds(0)
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setErrMsg(msg.includes('Permission') || msg.includes('NotAllowed')
        ? 'Microphone access denied — allow it in your browser settings'
        : `Mic error: ${msg}`)
      setState('error')
    }
  }

  const stop = () => {
    timerRef.current && clearInterval(timerRef.current)
    recorderRef.current?.stop()
    setState('idle')
  }

  if (state === 'recording') {
    return (
      <div className="flex items-center gap-3">
        <button
          onClick={stop}
          className="flex items-center gap-2 px-4 py-2 rounded text-sm font-medium
                     bg-red-700 hover:bg-red-600 text-white transition-colors"
        >
          <span className="w-2 h-2 rounded-full bg-red-300 animate-pulse" />
          Stop recording — {fmt(seconds)}
        </button>
        <span className="text-xs text-gray-500">Recording…</span>
      </div>
    )
  }

  return (
    <div className="space-y-1">
      <button
        onClick={start}
        disabled={disabled}
        className="flex items-center gap-2 px-4 py-2 rounded text-sm font-medium
                   bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700
                   disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm-1 17.93V21h2v-2.07A8 8 0 0 0 20 11h-2a6 6 0 0 1-12 0H4a8 8 0 0 0 7 7.93z"/>
        </svg>
        Record from microphone
      </button>
      {state === 'error' && (
        <p className="text-xs text-red-400">{errMsg}</p>
      )}
    </div>
  )
}
