import AudioInput from './AudioInput'
import AudioRecorder from './AudioRecorder'

interface Props {
  onAudio: (blob: Blob) => void
  disabled?: boolean
  accept?: string
}

export default function AudioInputArea({ onAudio, disabled, accept }: Props) {
  return (
    <div className="space-y-3">
      <AudioInput onFile={onAudio} disabled={disabled} accept={accept} />

      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-gray-800" />
        <span className="text-xs text-gray-600 shrink-0">or</span>
        <div className="flex-1 h-px bg-gray-800" />
      </div>

      <AudioRecorder onAudio={onAudio} disabled={disabled} />
    </div>
  )
}
